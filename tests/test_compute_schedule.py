"""Output batching/shared tau weights preserve recurrence and stochastic masks."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from benchmark_compute import RepeatedOutputReference, check_equivalence
from liquid_lm.layers import LiquidCell, _TRITON_AVAILABLE
from liquid_lm.model import LiquidLanguageModel, detach_hidden


def models(**overrides):
    config=dict(vocab_size=23,embedding_dim=16,hidden_size=16,num_layers=2,
                dropout=0.,zoneout=0.)
    config.update(overrides)
    torch.manual_seed(44)
    ref=RepeatedOutputReference(**config)
    current=LiquidLanguageModel(**config)
    current.load_state_dict(ref.state_dict())
    return ref,current


class ComputeScheduleTests(unittest.TestCase):
    def test_outputs_states_and_all_gradients_match_the_previous_schedule(self):
        tokens=torch.randint(0,23,(2,9))
        targets=torch.randint(0,23,tokens.shape)
        for config in ({},{'multi_tau':False,'ode_unfolds':1},
                       {'gate_mode':'cfc','wiring':'dense','ffn_mult':0.5},
                       {'norm_type':'rms','innovation_memory_dim':4},
                       {'residual':False,'mlp_head':True,'multi_tau':False,'embedding_dim':12,'tie_weights':False},
                       {'dropout':0.15,'zoneout':0.2,'use_checkpoint':True}):
            with self.subTest(config=config):
                ref,current=models(**config)
                self.assertEqual(list(ref.state_dict()),list(current.state_dict()))
                checked=check_equivalence((ref,current),tokens,targets,torch.device('cpu'),'fp32',123)
                self.assertEqual(checked['checked_parameter_tensors'],len(list(current.parameters())))

    def test_preparations_keep_native_amp_gradients_and_batch_normalization(self):
        model=models()[1]
        tokens=torch.randint(0,23,(2,8))
        shapes=[]
        handle=model.final_norm.register_forward_hook(lambda m,a,o:shapes.append(tuple(a[0].shape)))
        with patch('liquid_lm.model.F.softmax',wraps=F.softmax) as softmax:
            logits,_=model(tokens)
            self.assertEqual(softmax.call_count,model.num_layers)
        self.assertEqual(shapes,[(2,8,16)])
        logits.square().sum().backward()
        for cell in model.cells:
            self.assertIsNotNone(cell.tau_mix.grad)
            self.assertTrue(torch.isfinite(cell.tau_mix.grad).all())
        shapes.clear()
        last,_=model(tokens,last_token_only=True)
        self.assertEqual(shapes,[(2,1,16)])
        torch.testing.assert_close(last,logits[:,-1:])
        handle.remove()
        # Guard the actual AMP-rounding regression observed with a joint GEMM:
        # input-gradient contributions must round per head before their FP32 sum.
        reference_cell=LiquidCell(17,19)
        prepared_cell=copy.deepcopy(reference_cell)
        source=torch.randn(3,17)
        probes=[torch.randn(3,19) for _ in range(2)]
        inputs=[]
        for index,cell in enumerate((reference_cell,prepared_cell)):
            x=source.clone().requires_grad_()
            inputs.append(x)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                weights=cell.input_projection_weights() if index else None
                if index:
                    self.assertEqual(len(weights),3)
                heads=cell.precompute_x(x,weights)
                loss=sum((head.float()*probe).sum() for head,probe in zip(heads,probes))
            loss.backward()
        torch.testing.assert_close(inputs[0].grad,inputs[1].grad,rtol=0,atol=0)
        for (name,a),(_,b) in zip(reference_cell.named_parameters(),prepared_cell.named_parameters()):
            if a.grad is not None:
                torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=name)

    def test_optimizer_updates_and_carried_state_use_fresh_tau_weights(self):
        pair=models(dropout=0.1,zoneout=0.15,norm_type='rms',use_checkpoint=True)
        optimizers=[torch.optim.AdamW(m.parameters(),lr=0.002) for m in pair]
        tokens=torch.randint(0,23,(2,12))
        hidden=[None,None]
        for step in range(2):
            for index,(model,optimizer) in enumerate(zip(pair,optimizers)):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(800+step)
                out,state=model(tokens[:,step*6:(step+1)*6],hidden[index])
                F.cross_entropy(out.flatten(0,1),tokens[:,step*6:(step+1)*6].flatten()).backward()
                optimizer.step()
                hidden[index]=detach_hidden(state)
            for (name,a),(_,b) in zip(pair[0].named_parameters(),pair[1].named_parameters()):
                torch.testing.assert_close(a,b,rtol=3e-5,atol=3e-6,msg=name)

    def test_compiled_step_signature_and_fullgraph_backward(self):
        current=models()[1]
        ref=copy.deepcopy(current)
        current._step_fn=torch.compile(current._step_impl,backend='eager',fullgraph=True)
        tokens=torch.randint(0,23,(2,4))
        expected=ref(tokens)[0]
        actual=current(tokens)[0]
        torch.testing.assert_close(actual,expected)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in current.parameters()))


@unittest.skipUnless(torch.cuda.is_available() and _TRITON_AVAILABLE,'CUDA/Triton required')
class ComputeScheduleGpuTests(unittest.TestCase):
    def test_all_amp_gradients_and_regularized_checkpointing(self):
        old_tf32=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False
        try:
            for precision in ('fp32','bf16','fp16'):
                for config in ({},{'dropout':.1,'zoneout':.2,'use_checkpoint':True,'wiring':'dense','norm_type':'rms'},
                               {'innovation_memory_dim':4,'ffn_mult':.5},
                               {'gate_mode':'cfc','residual':False,'embedding_dim':12,'tie_weights':False}):
                    with self.subTest(precision=precision,config=config):
                        pair=[m.cuda() for m in models(**config)]
                        if config.get('gate_mode')!='cfc':
                            for m in pair:
                                self.assertEqual(m.set_kernel('triton'),'triton')
                        tokens=torch.randint(0,23,(2,8),device='cuda')
                        targets=torch.randint(0,23,tokens.shape,device='cuda')
                        checked=check_equivalence(pair,tokens,targets,torch.device('cuda'),precision,72)
                        self.assertEqual(checked['checked_parameter_tensors'],len(list(pair[1].parameters())))
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_tf32


if __name__=='__main__':
    unittest.main()

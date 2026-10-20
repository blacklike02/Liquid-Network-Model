"""Guard benchmark labels, counterfactual controls and held-out query tasks."""

import unittest

import torch

from benchmark_memory_validation import make_models, multiquery_batch, overwrite_batch


class MemoryBenchmarkTests(unittest.TestCase):
    def test_overwrite_queries_the_new_value_and_keeps_the_old_different(self):
        tokens, target, old = overwrite_batch(torch.Generator().manual_seed(1), 128, 8)
        self.assertTrue(torch.all(target != old))
        for row, new, previous in zip(tokens, target, old):
            writes = row[:15].reshape(5, 3)
            query_key = row[-1]
            matching = writes[writes[:, 1] == query_key, 2]
            self.assertEqual(matching.numel(), 2)
            self.assertEqual(matching[0], previous)
            self.assertEqual(matching[-1], new)
            self.assertEqual(row[-2], 2)
            self.assertTrue(torch.all((row[15:-2] >= 19) & (row[15:-2] < 27)))

    def test_multiple_query_labels_come_only_from_preceding_writes(self):
        tokens, targets, positions = multiquery_batch(torch.Generator().manual_seed(2), 128, 8)
        for row, answers in zip(tokens, targets):
            writes = row[:12].reshape(4, 3)
            for position, answer in zip(positions, answers):
                self.assertEqual(row[position - 1], 2)
                matching = writes[writes[:, 1] == row[position], 2]
                self.assertEqual(matching.numel(), 1)
                self.assertEqual(matching[0], answer)
            self.assertTrue(torch.all((row[20:] >= 2) & (row[20:] < 11)))

    def test_ablation_initialization_is_identical_and_delta_gate_is_exact(self):
        models = make_models(11)
        for name, model in models.items():
            for key, value in models['baseline'].state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0, msg=name)
        memory = models['innovation'].innovation_memory
        for name in ('delta_control', 'scalar_decay'):
            for key, value in memory.state_dict().items():
                torch.testing.assert_close(value, models[name].innovation_memory.state_dict()[key], rtol=0, atol=0)
        error = torch.tensor([[0., 0.], [1., 2.]])
        torch.testing.assert_close(models['delta_control'].innovation_memory._write_novelty(error), torch.ones(2, 1))
        rates = models['scalar_decay'].innovation_memory._decay_rate(torch.zeros(2, 18))
        self.assertEqual(rates.shape, (2, 1))


if __name__ == '__main__':
    unittest.main()

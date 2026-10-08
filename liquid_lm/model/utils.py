"""Small helpers for working with the model and its state."""

from __future__ import annotations


def get_base_model(model):
    return getattr(model, "_orig_mod", model)


def describe_wiring(model) -> str:
    base = get_base_model(model)
    if getattr(base, "wiring", "dense") != "ncp":
        return "dense"
    frac = base.cells[0].recurrent_param_fraction
    if frac is None:
        return "dense"
    return f"ncp (рекуррентная часть ~{frac * 100:.0f}% от dense по параметрам/FLOPs)"


def detach_hidden(hidden):
    return [h.detach() for h in hidden]


def clone_float_state(model):
    return {
        k: v.detach().clone()
        for k, v in get_base_model(model).state_dict().items()
        if v.dtype.is_floating_point
    }


def restore_float_state(model, backup):
    msd = get_base_model(model).state_dict()
    for k, v in backup.items():
        msd[k].copy_(v)

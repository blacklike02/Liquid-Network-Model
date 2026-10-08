"""Architecture config keys, model construction and compatible weight loading."""

from __future__ import annotations

from .liquid_lm import LiquidLanguageModel


ARCH_KEYS = (
    "embedding_dim", "hidden_size", "num_layers", "ode_unfolds",
    "dropout", "zoneout", "tie_weights", "mlp_head", "residual",
    "multi_tau", "local_conv", "gate_mode", "ode_solver",
    "wiring", "ncp_command_frac", "ncp_motor_frac", "memory_slots",
)


LEGACY_ARCH_DEFAULTS = {
    "zoneout": 0.0, "tie_weights": False, "mlp_head": False,
    "residual": False, "multi_tau": False, "local_conv": False,
    "gate_mode": "ltc", "ode_solver": "euler",
    "wiring": "dense",
    "ncp_command_frac": 0.4, "ncp_motor_frac": 0.2,
    "memory_slots": 0,
}


def _arch_value(config, key):
    if key in config:
        return config[key]
    return LEGACY_ARCH_DEFAULTS.get(key)


def create_model_from_config(vocab_size, config, device):
    return LiquidLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=int(config["embedding_dim"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        ode_unfolds=int(config["ode_unfolds"]),
        dropout=float(config.get("dropout", 0.05)),
        zoneout=float(_arch_value(config, "zoneout") or 0.0),
        tie_weights=bool(_arch_value(config, "tie_weights")),
        mlp_head=bool(_arch_value(config, "mlp_head")),
        residual=bool(_arch_value(config, "residual")),
        multi_tau=bool(_arch_value(config, "multi_tau")),
        local_conv=bool(_arch_value(config, "local_conv")),
        use_checkpoint=False,
        gate_mode=str(_arch_value(config, "gate_mode") or "ltc"),
        ode_solver=str(_arch_value(config, "ode_solver") or "euler"),
        wiring=str(_arch_value(config, "wiring") or "dense"),
        ncp_command_frac=float(_arch_value(config, "ncp_command_frac") or 0.4),
        ncp_motor_frac=float(_arch_value(config, "ncp_motor_frac") or 0.2),
        memory_slots=int(_arch_value(config, "memory_slots") or 0),
    ).to(device)


def load_model_state_compat(model, state):
    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Неожиданные ключи в checkpoint: {unexpected}")
    allowed_missing = {f"h0.{i}" for i in range(model.num_layers)}
    if model.multi_tau:
        allowed_missing.add("tau_mix")
    if model.local_conv_layer is not None:
        allowed_missing.add("local_conv_layer.weight")
        allowed_missing.add("local_conv_layer.bias")
    bad_missing = [k for k in missing if k not in allowed_missing]
    if bad_missing:
        raise RuntimeError(f"Checkpoint is missing weights: {bad_missing}")
    print("[CHECKPOINT] Compatible weights loaded (new features initialized).")

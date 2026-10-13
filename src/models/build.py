"""Configurable DeepONet construction and exact checkpoint reconstruction."""
from .deeponet import DeepONet

def apply_variant(cfg):
    if cfg.operator != "deeponet" or cfg.variant not in ("baseline", "custom"):
        raise ValueError("This study supports DeepONet baseline/custom only.")
    if cfg.deeponet_activation not in ("tanh", "gelu"):
        raise ValueError("Activation must be tanh or gelu.")
    for widths in (cfg.branch_hidden, cfg.trunk_hidden):
        if not widths or any(int(w) != w or w < 1 for w in widths):
            raise ValueError("Hidden widths must be positive integers.")

def build_model(cfg, ny, nx):
    return DeepONet(ny * nx, cfg.branch_hidden, cfg.trunk_hidden,
                   cfg.latent_dim, cfg.deeponet_activation)

def model_from_checkpoint(checkpoint):
    if checkpoint["model_kind"] != "deeponet":
        raise ValueError("Not a DeepONet checkpoint.")
    return DeepONet(**checkpoint["architecture"])

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())

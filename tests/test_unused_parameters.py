"""Guard against network branches that never receive a gradient.

DDP refuses to reduce a trainable parameter that the loss backward pass never
reaches, which is why the trainer had to fall back to
`ddp_find_unused_parameters_true`. Dead branches are invisible in single-GPU runs
and only blow up (or silently cost a full graph traversal per step) once the warm
start is unfrozen under DDP, so they need a test rather than a code review.

Scope: this checks that every trainable parameter of the network is reachable from
its own outputs. It does not check that every output is supervised by the loss,
which is a per-task property of `configs/losses` and the `keep_input_*` flags.
"""

import os

import hydra
import pyrootutils
import pytest
import torch
from hydra import compose, initialize_config_dir

# Puts the repo root on sys.path so Hydra discovers `hydra_plugins/resolvers.py`,
# which registers the custom OmegaConf resolvers the model configs rely on.
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

# Shapes of a synthetic sample, small enough to fit next to a real training run.
BATCH, N_CAMS = 1, 6
IMG_H, IMG_W = 448, 800


def _synthetic_batch(cfg, device):
    """Build the tensors `BEVPredFormerPredictor.forward` reads, without touching NuScenes."""
    t = len(cfg.data.cam_T_P)
    eye = torch.eye(4, device=device)
    intrins = torch.tensor(
        [[800.0, 0.0, IMG_W / 2], [0.0, 800.0, IMG_H / 2], [0.0, 0.0, 1.0]],
        device=device,
    )
    return {
        "imgs": torch.randn(BATCH, t, N_CAMS, 3, IMG_H, IMG_W, device=device),
        "rots": torch.eye(3, device=device).expand(BATCH, t, N_CAMS, 3, 3).contiguous(),
        "trans": torch.zeros(BATCH, t, N_CAMS, 3, 1, device=device),
        "intrins": intrins.expand(BATCH, t, N_CAMS, 3, 3).contiguous(),
        "bev_aug": eye.expand(BATCH, t, 4, 4).contiguous(),
        "egoTin_to_seq": eye.expand(BATCH, t, 4, 4).contiguous(),
        "future_egomotion": torch.zeros(
            BATCH, len(cfg.data.bev_T_P), 6, device=device
        ),
    }


def _output_tensors(out):
    if isinstance(out, torch.Tensor):
        return [out] if out.is_floating_point() and out.requires_grad else []
    if isinstance(out, dict):
        return [t for v in out.values() for t in _output_tensors(v)]
    if isinstance(out, (list, tuple)):
        return [t for v in out for t in _output_tensors(v)]
    return []


def _compose(overrides):
    os.environ.setdefault("PROJECT_ROOT", os.path.abspath(CONFIG_DIR + "/.."))
    with initialize_config_dir(
        version_base="1.3", config_dir=os.path.abspath(CONFIG_DIR)
    ):
        return compose(config_name="train.yaml", overrides=overrides)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="deformable attention and spconv are CUDA-only",
)
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param([], id="defaults"),
        pytest.param(
            ["model.net.autoencoder.with_proj_residual=True"],
            id="proj_residual",
        ),
        pytest.param(
            ["model.net.view_transform.n_layers=6"],
            id="multi_layer_view_transform",
        ),
    ],
)
def test_every_trainable_parameter_gets_a_gradient(overrides):
    cfg = _compose(overrides)
    device = torch.device("cuda")
    net = hydra.utils.instantiate(cfg.model.net).to(device)
    net.train()

    # Same precision as the training runs, otherwise a full-resolution sample in
    # fp32 does not fit next to anything else on the GPU.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = _output_tensors(net(**_synthetic_batch(cfg, device)))
    assert outputs, "network produced no differentiable output"

    loss = sum(t.float().pow(2).mean() for t in outputs)
    loss.backward()

    unused = [n for n, p in net.named_parameters() if p.requires_grad and p.grad is None]

    # Each case builds a full network, so release it before the next one runs.
    del outputs, loss, net
    torch.cuda.empty_cache()

    assert not unused, (
        f"{len(unused)} trainable parameters got no gradient, DDP would report them "
        f"as unused: {unused[:10]}"
    )

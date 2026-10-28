"""Calibration-noise robustness evaluation on nuScenes val (inference only).

Standalone: it reuses val.yaml and the regular validation pipeline
(bevpredformer/val.py), and only perturbs the camera extrinsics the model
receives. Each camera's sensor-to-ego rotation and translation (`rots`, `trans`
returned by TemporalNuScenesDataset.__getitem__) are perturbed before the model
builds its ego-to-camera projection (CamProjector.from_spatial_to_cams).
Intrinsics are unchanged. GT is unaffected: the BEV labels never use the camera
extrinsics, and the perspective labels use their own unperturbed copy.

Noise model:
- Rotation: R' = R_delta @ R, with R_delta a rotation of fixed angle `rot_deg`
  about a uniformly random axis.
- Translation: t' = t + trans_m * u, with u a uniformly random unit vector.
- Sampled once per camera per val sample (cameras are independent) and shared
  by all input timesteps of the sample: a calibration error is a constant offset.
- The RNG is seeded with (seed, sample index), so runs are reproducible and do
  not depend on the number of dataloader workers.

Usage (same overrides as val.py):
    uv run python scripts/eval_calib_noise.py \
        +calib_noise.rot_deg=1.0 +calib_noise.trans_m=0.05 +calib_noise.seed=0 \
        +out_json=<path>
"""

import json
from typing import Optional

import hydra
import numpy as np
import pyrootutils
import torch
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from bevpredformer import utils
from bevpredformer.data.dataset.nuscenes_temporal import TemporalNuScenesDataset
from bevpredformer.val import val

log = utils.get_pylogger(__name__)

NOISE = {"rot_deg": 0.0, "trans_m": 0.0, "seed": 0}


def random_unit_vectors(rng, n):
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def axis_angle_to_matrix(axes, angle):
    """Rodrigues formula for unit axes (n, 3) and a shared angle (rad)."""
    K = np.zeros((len(axes), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axes[:, 2], axes[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axes[:, 2], -axes[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axes[:, 1], axes[:, 0]
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


# --------------------------------------------------------------------------- #
# Dataset patch: perturb the extrinsics fed to the model.
# --------------------------------------------------------------------------- #
_orig_getitem = TemporalNuScenesDataset.__getitem__


def __getitem__(self, index):
    out = _orig_getitem(self, index)
    rots, trans = out["rots"], out["trans"]  # (T, N, 3, 3), (T, N, 3, 1)
    n = rots.shape[1]

    rng = np.random.default_rng([NOISE["seed"], index])
    r_delta = axis_angle_to_matrix(random_unit_vectors(rng, n), np.deg2rad(NOISE["rot_deg"]))
    t_delta = NOISE["trans_m"] * random_unit_vectors(rng, n)

    r_delta = torch.from_numpy(r_delta).to(rots.dtype)
    t_delta = torch.from_numpy(t_delta).to(trans.dtype).unsqueeze(-1)
    out["rots"] = r_delta.unsqueeze(0) @ rots
    out["trans"] = trans + t_delta.unsqueeze(0)
    return out


TemporalNuScenesDataset.__getitem__ = __getitem__


@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    noise_cfg = cfg.get("calib_noise", {}) or {}
    NOISE.update({k: noise_cfg[k] for k in NOISE if k in noise_cfg})
    log.info(f"Calibration noise: {NOISE}")

    utils.modif_config_based_on_flags(cfg)
    utils.extras(cfg)
    metric_dict, _ = val(cfg)
    metrics = {k: float(v) for k, v in metric_dict[0].items()}

    results = {
        "calib_noise": dict(NOISE),
        "ckpt": cfg.ckpt.path,
        "grid": {"xbound": list(cfg.data.grid.xbound), "ybound": list(cfg.data.grid.ybound)},
        "VPQ": metrics["val_vpq_metric"],
        "RQ": metrics["val_rq_metric"],
        "SQ": metrics["val_sq_metric"],
        "IoU": metrics["val_bev_metric_iou_Time_Avg"],
        "metrics": metrics,
    }
    print(json.dumps(results, indent=2))
    if cfg.get("out_json"):
        with open(cfg.out_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

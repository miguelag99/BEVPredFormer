"""Ego-pose noise robustness evaluation on nuScenes val (inference only).

Standalone: it reuses val.yaml and the regular validation pipeline
(bevpredformer/val.py), and only perturbs the ego-pose inputs of the model.
The model consumes the ego pose through two tensors, both built from the
LIDAR_TOP ego poses:
- `egoTin_to_seq` (T_in, 4, 4): ego at input frame t -> present ego. Places the
  voxel reference points of past frames (CamProjector.from_spatial_to_seqaug).
- `future_egomotion` (S, 6): consecutive transform t -> t+1 as a 6-dof vector.
  The predictor appends its first T_in - 1 rows to the BEV queries.
GT (labels, flow, instances) does not use these tensors and is unaffected.

Noise model: each past input frame t gets a planar error N_t in the present ego
frame (translation of `trans_m` in a uniformly random xy direction, yaw of
`yaw_deg` with random sign), drawn independently per frame. Its pose relative
to the present becomes M'_t = N_t @ M_t; the present frame is unperturbed.
Both tensors are rebuilt from the perturbed poses, so they stay consistent:
egoTin_to_seq'_t = M'_t and the consecutive transform t -> t+1 is
inv(M'_{t+1}) @ M'_t. The RNG is seeded with (seed, sample index).

Usage (same overrides as val.py):
    uv run python scripts/eval_egopose_noise.py \
        +egopose_noise.trans_m=0.25 +egopose_noise.yaw_deg=1.0 +egopose_noise.seed=0 \
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
from bevpredformer.utils.geom import mat2pose_vec
from bevpredformer.val import val

log = utils.get_pylogger(__name__)

NOISE = {"trans_m": 0.0, "yaw_deg": 0.0, "seed": 0}


def planar_noise(rng, trans_m, yaw_deg):
    phi = rng.uniform(0.0, 2 * np.pi)
    yaw = np.deg2rad(yaw_deg) * rng.choice([-1.0, 1.0])
    mat = np.eye(4)
    mat[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    mat[:2, 3] = trans_m * np.array([np.cos(phi), np.sin(phi)])
    return mat


# --------------------------------------------------------------------------- #
# Dataset patch: perturb the ego-pose inputs fed to the model.
# --------------------------------------------------------------------------- #
_orig_getitem = TemporalNuScenesDataset.__getitem__


def __getitem__(self, index):
    out = _orig_getitem(self, index)
    # Rebuilding future_egomotion differs from the dataset's by float round-off
    # (~1e-4), so the clean level keeps the original tensors untouched.
    if NOISE["trans_m"] == 0 and NOISE["yaw_deg"] == 0:
        return out
    assert self.mode_ref_cam_T == "present"
    # Input frames must be consecutive records ending at the present.
    assert np.array_equal(self.cam_T_index, np.arange(len(self.cam_T_index)))
    assert self.cam_T_index[-1] == self.present_index[0]

    records = [self.ixes[i] for i in self.indices[index]]
    present = records[self.present_index[0]]
    poses = [self._get_egoTin_to_egoTout(records[i], present) for i in self.cam_T_index]

    rng = np.random.default_rng([NOISE["seed"], index])
    poses = [
        planar_noise(rng, NOISE["trans_m"], NOISE["yaw_deg"]) @ m for m in poses[:-1]
    ] + [poses[-1]]

    out["egoTin_to_seq"] = torch.from_numpy(np.stack(poses)).float()
    for t in range(len(poses) - 1):
        step = torch.from_numpy(np.linalg.inv(poses[t + 1]) @ poses[t])
        out["future_egomotion"][t] = mat2pose_vec(step).float()
    return out


TemporalNuScenesDataset.__getitem__ = __getitem__


@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    noise_cfg = cfg.get("egopose_noise", {}) or {}
    NOISE.update({k: noise_cfg[k] for k in NOISE if k in noise_cfg})
    log.info(f"Ego-pose noise: {NOISE}")

    utils.modif_config_based_on_flags(cfg)
    utils.extras(cfg)
    metric_dict, _ = val(cfg)
    metrics = {k: float(v) for k, v in metric_dict[0].items()}

    results = {
        "egopose_noise": dict(NOISE),
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

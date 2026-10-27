"""Speed-stratified VPQ / IoU evaluation on nuScenes val.

Standalone: it reuses val.yaml and the model/datamodule, but does not touch the
regular evaluation pipeline. Each GT vehicle instance gets a speed (norm of the
nuScenes annotation velocity, nusc.box_velocity, at the present frame) and is
assigned to a bin. VPQ and IoU are then computed per bin with the paper protocol
(frames [:, 1:], valid_binimg mask, global TP/FP/FN accumulation, IoU > 0.5
matching, unique_id_mapping consistency).

Per-bin rules:
- VPQ: only GT instances of the bin are scored. Predictions matched to a GT
  instance of another bin are ignored. Unmatched predictions are assigned to a
  bin by their predicted speed (mean predicted backward flow over the instance
  pixels * resolution / dt). With +fp_bin_rule=overlap, an unmatched prediction
  that overlaps GT instances takes the bin of the one with the largest pixel
  overlap; only non-overlapping ones fall back to the predicted speed.
- IoU: GT pixels of instances from other bins are ignored.
GT instances without a valid velocity (single-annotation tracks) are ignored in
every bin. The "all" row is the unstratified metric and must match the model's
own metrics, which are computed in the same run as a sanity check.

Usage (same overrides as val.py):
    uv run python scripts/eval_speed_stratified.py \
        data/augs@data.img_params=scale_0_5 ckpt.path=checkpoints/<ckpt> \
        +out_json=<path>
"""

import json
from collections import defaultdict

import hydra
import numpy as np
import pyrootutils
import torch
from omegaconf import DictConfig
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from bevpredformer import utils
from bevpredformer.data.dataset.nuscenes_temporal import TemporalNuScenesDataset
from bevpredformer.utils.instance_pred import predict_instance_segmentation

torch.set_float32_matmul_precision("high")

DT = 0.5  # seconds between nuScenes keyframes
MAX_IDS = 2048
BINS = {"static": (0.0, 0.5), "slow": (0.5, 5.0), "fast": (5.0, float("inf"))}
BIN_NAMES = list(BINS)
UNKNOWN = -1


def speed_to_bin(speed):
    """Map speeds (tensor, m/s) to bin index; NaN -> UNKNOWN."""
    out = torch.full_like(speed, UNKNOWN, dtype=torch.long)
    for i, (lo, hi) in enumerate(BINS.values()):
        out[(speed >= lo) & (speed < hi)] = i
    return out


# --------------------------------------------------------------------------- #
# Dataset patch: attach per-instance-id GT speed to every sample.
# --------------------------------------------------------------------------- #
_orig_get_outputs_bev = TemporalNuScenesDataset._get_outputs_bev
_orig_getitem = TemporalNuScenesDataset.__getitem__


def _get_outputs_bev(self, bev_records_T, *args, **kwargs):
    data_bev, inst_map = _orig_get_outputs_bev(self, bev_records_T, *args, **kwargs)
    self._speed_ctx = (bev_records_T, dict(inst_map))
    return data_bev, inst_map


def _instance_to_ann(nusc, rec):
    return {nusc.get("sample_annotation", a)["instance_token"]: a for a in rec["anns"]}


def __getitem__(self, index):
    out = _orig_getitem(self, index)
    bev_records_T, inst_map = self._speed_ctx
    assert len(inst_map) < MAX_IDS

    # Search order: present frame first, then future frames, then past.
    present = int(np.where(self.bev_T == 0)[0][0])
    order = [present] + [i for i in range(len(self.bev_T)) if self.bev_T[i] > 0] \
        + [i for i in range(len(self.bev_T)) if self.bev_T[i] < 0]
    lookups = [_instance_to_ann(self.nusc, bev_records_T[i]) for i in order]

    speed = torch.full((MAX_IDS,), float("nan"))
    from_present = torch.zeros(MAX_IDS, dtype=torch.bool)
    token_hash = torch.zeros(MAX_IDS, dtype=torch.long)
    for token, inst_id in inst_map.items():
        if token == "ego":
            continue
        token_hash[inst_id] = int(token[:15], 16)
        for k, lut in enumerate(lookups):
            if token in lut:
                v = self.nusc.box_velocity(lut[token])
                speed[inst_id] = float(np.linalg.norm(v[:2]))
                from_present[inst_id] = k == 0
                break
    out["inst_speed"] = speed
    out["inst_speed_from_present"] = from_present
    out["inst_token_hash"] = token_hash
    return out


TemporalNuScenesDataset._get_outputs_bev = _get_outputs_bev
TemporalNuScenesDataset.__getitem__ = __getitem__


# --------------------------------------------------------------------------- #
# Stratified metrics.
# --------------------------------------------------------------------------- #
class StratifiedMetrics:
    """Rows: one per bin + "all" (unstratified, must equal the standard metric)."""

    def __init__(self, res, fp_bin_rule="flow"):
        assert fp_bin_rule in ("flow", "overlap")
        self.res = res
        self.fp_bin_rule = fp_bin_rule
        # Unmatched-prediction FPs per (bin, rule); rule is "overlap" or "flow".
        self.fp_rule = defaultdict(int)
        self.rows = BIN_NAMES + ["all"]
        self.vpq = {r: defaultdict(float) for r in self.rows}
        self.pix = {r: defaultdict(int) for r in self.rows}
        self.inst_seq = defaultdict(int)  # (sequence, gt instance) pairs
        self.inst_frames = defaultdict(int)  # (frame, gt instance) pairs
        self.inst_tokens = defaultdict(set)  # unique nuScenes instances
        self.n_from_present = 0
        self.n_not_from_present = 0

    @staticmethod
    def _pred_speeds(pred_inst, flow, n_pred, res):
        """Speed per predicted id from the mean backward centripetal flow (px/frame)."""
        ids = pred_inst.view(-1)
        cnt = torch.bincount(ids, minlength=n_pred).float().clamp(min=1)
        fx = torch.bincount(ids, weights=flow[0].reshape(-1).float(), minlength=n_pred) / cnt
        fy = torch.bincount(ids, weights=flow[1].reshape(-1).float(), minlength=n_pred) / cnt
        return torch.sqrt(fx**2 + fy**2) * res / DT

    def update_vpq(self, pred_instance, gt_instance, flow, gt_bin, b_tokens, from_present):
        """pred_instance, gt_instance: (s, h, w); flow: (s, 2, h, w); gt_bin: (MAX_IDS,)."""
        unique_id_mapping = {}
        seen = set()
        for t in range(gt_instance.shape[0]):
            pred, gt = pred_instance[t].long(), gt_instance[t].long()
            n_gt = int(gt.max().item()) + 1
            n_pred = int(pred.max().item()) + 1
            conf = torch.bincount(
                (gt * n_pred + pred).view(-1), minlength=n_gt * n_pred
            ).view(n_gt, n_pred)
            gt_area, pred_area = conf.sum(1), conf.sum(0)
            union = gt_area[:, None] + pred_area[None, :] - conf
            iou = torch.where(
                union > 0, (conf.float() + 1e-9) / (union.float() + 1e-9), torch.zeros_like(union).float()
            )
            iou[0, :] = 0
            iou[:, 0] = 0
            matches = (iou > 0.5).nonzero(as_tuple=False).tolist()
            pred_speed_bin = speed_to_bin(self._pred_speeds(pred, flow[t], n_pred, self.res)).tolist()
            iou_l = iou.cpu()
            gt_present = (gt_area > 0).nonzero().flatten().tolist()
            pred_present = (pred_area > 0).nonzero().flatten().tolist()
            gbin = gt_bin[:n_gt]
            gt_ov = conf[1:]
            best_gt = (gt_ov.argmax(0) + 1).tolist() if n_gt > 1 else [0] * n_pred
            has_ov = (gt_ov.max(0).values > 0).tolist() if n_gt > 1 else [False] * n_pred

            def add(g_bin, key, val=1.0):
                self.vpq["all"][key] += val
                if g_bin != UNKNOWN:
                    self.vpq[BIN_NAMES[g_bin]][key] += val

            matched_gt, matched_pred = set(), set()
            for g, p in matches:
                matched_gt.add(g)
                matched_pred.add(p)
                if g in unique_id_mapping and unique_id_mapping[g] != p:
                    add(gbin[g], "fn")
                    add(gbin[g], "fp")
                    unique_id_mapping[g] = p
                    continue
                add(gbin[g], "tp")
                add(gbin[g], "iou", iou_l[g, p].item())
                unique_id_mapping[g] = p

            for g in gt_present:
                if g == 0:
                    continue
                if g not in matched_gt:
                    add(gbin[g], "fn")
                self.inst_frames[gbin[g]] += 1
                if g not in seen:
                    seen.add(g)
                    self.inst_seq[gbin[g]] += 1
                    self.inst_tokens[gbin[g]].add(b_tokens[g])
                    if from_present[g]:
                        self.n_from_present += 1
                    else:
                        self.n_not_from_present += 1

            for p in pred_present:
                if p == 0 or p in matched_pred:
                    continue
                if self.fp_bin_rule == "overlap" and has_ov[p]:
                    # Bin of the GT instance with the largest pixel overlap.
                    p_bin, rule = gbin[best_gt[p]], "overlap"
                else:
                    p_bin, rule = pred_speed_bin[p], "flow"
                add(p_bin, "fp")
                self.fp_rule[(p_bin, rule)] += 1

    def update_iou(self, pred_cls, target, gt_instance, gt_bin):
        """pred_cls, target: (s, h, w) in {0,1}; gt_instance: (s, h, w)."""
        pred_cls, target = pred_cls.bool(), target.bool()
        pix_bin = gt_bin.to(gt_instance.device)[gt_instance.long()]
        is_inst = gt_instance > 0
        self.pix["all"]["tgt_without_instance"] += int((target & ~is_inst).sum())
        for r in self.rows:
            if r == "all":
                keep = torch.ones_like(target)
            else:
                keep = ~is_inst | (pix_bin == BIN_NAMES.index(r))
            self.pix[r]["tp"] += int((pred_cls & target & keep).sum())
            self.pix[r]["fp"] += int((pred_cls & ~target & keep).sum())
            self.pix[r]["fn"] += int((~pred_cls & target & keep).sum())

    def compute(self):
        out = {}
        for r in self.rows:
            v, p = self.vpq[r], self.pix[r]
            den = max(v["tp"] + v["fp"] / 2 + v["fn"] / 2, 1)
            out[r] = {
                "VPQ": v["iou"] / den,
                "SQ": v["iou"] / max(v["tp"], 1),
                "RQ": v["tp"] / den,
                "IoU": p["tp"] / max(p["tp"] + p["fp"] + p["fn"], 1),
                "TP": v["tp"], "FP": v["fp"], "FN": v["fn"],
            }
            idx = BIN_NAMES.index(r) if r in BIN_NAMES else None
            if idx is not None:
                out[r]["n_instance_sequences"] = self.inst_seq[idx]
                out[r]["n_instance_frames"] = self.inst_frames[idx]
                out[r]["n_unique_instances"] = len(self.inst_tokens[idx])
        out["fp_bin_rule"] = self.fp_bin_rule
        out["unmatched_fp_by_rule"] = {
            name: {rule: self.fp_rule[(b, rule)] for rule in ("overlap", "flow")}
            for name, b in [*((n, i) for i, n in enumerate(BIN_NAMES)), ("unknown", UNKNOWN)]
        }
        out["unmatched_fp_by_rule"]["all"] = {
            rule: sum(v for (_, r), v in self.fp_rule.items() if r == rule) for rule in ("overlap", "flow")
        }
        out["unknown_speed"] = {
            "n_instance_sequences": self.inst_seq[UNKNOWN],
            "n_instance_frames": self.inst_frames[UNKNOWN],
        }
        out["speed_source"] = {
            "present_frame": self.n_from_present,
            "other_frame": self.n_not_from_present,
        }
        out["tgt_pixels_without_instance"] = self.pix["all"]["tgt_without_instance"]
        return out


@torch.no_grad()
def evaluate(cfg: DictConfig):
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("only_val")
    model = hydra.utils.instantiate(cfg.model)
    model = utils.load_state_model(
        model, utils.get_ckpt_from_path(cfg.ckpt.path), cfg.ckpt.model.freeze, cfg.ckpt.model.load, verbose=1
    )
    model = model.cuda().eval()

    res = cfg.data.grid.xbound[2]
    strat = StratifiedMetrics(res, cfg.get("fp_bin_rule", "flow"))
    loader = datamodule.val_dataloader()
    limit = cfg.get("limit_batches", None)

    for i, batch in enumerate(tqdm(loader, total=limit or len(loader))):
        if limit is not None and i >= limit:
            break
        speed = batch.pop("inst_speed")
        from_present = batch.pop("inst_speed_from_present")
        token_hash = batch.pop("inst_token_hash")
        batch = {k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch = datamodule.on_after_batch_transfer(batch, 0)

        preds = model(batch)
        # Standard metrics (reference values, identical code path to val.py).
        model._common_step_metrics(preds, batch, "val")

        pred_instance_seg = predict_instance_segmentation(
            {
                "instance_flow": preds["bev"]["flow"],
                "segmentation": preds["bev"]["binimg"],
                "centerness": preds["bev"]["centerness"] if model.with_centr_offs else None,
            },
            spatial_extent=(model.grid["xbound"][1], model.grid["ybound"][1]),
            conf_threshold=model._conf_threshold,
            nms_kernel_size=model._nms_kernel_size,
        )
        valid = batch["valid_binimg"]
        pred_inst = (pred_instance_seg[:, 1:] * valid[:, 1:].squeeze(2)).long()
        gt_inst = (batch["instance"][:, 1:] * valid[:, 1:]).squeeze(2).long()
        cls_pred = torch.argmax(preds["bev"]["binimg"].sigmoid().contiguous(), 2, keepdims=True)
        cls_pred = (cls_pred[:, 1:] * valid[:, 1:]).squeeze(2)
        target = (batch["binimg"][:, 1:] * valid[:, 1:]).squeeze(2)

        for b in range(gt_inst.shape[0]):
            gt_bin = speed_to_bin(speed[b])
            strat.update_vpq(
                pred_inst[b], gt_inst[b], preds["bev"]["flow"][b, 1:],
                gt_bin.tolist(), token_hash[b].tolist(), from_present[b].tolist(),
            )
            strat.update_iou(cls_pred[b], target[b], gt_inst[b], gt_bin)

    results = strat.compute()
    vpq = model.metric_vpq_val.compute()
    results["reference_model_metrics"] = {
        "VPQ": vpq["pq"][1].item(),
        "IoU": model.metric_iou_Time_Avg_val.compute()[1].item(),
    }
    return results


def print_table(results):
    print(f"{'bin':<8}{'VPQ':>8}{'SQ':>8}{'RQ':>8}{'IoU':>8}{'TP':>9}{'FP':>9}{'FN':>9}{'#inst-seq':>11}{'#inst-frm':>11}{'#uniq':>8}")
    for r in BIN_NAMES + ["all"]:
        v = results[r]
        print(
            f"{r:<8}{100*v['VPQ']:>8.2f}{100*v['SQ']:>8.2f}{100*v['RQ']:>8.2f}{100*v['IoU']:>8.2f}"
            f"{int(v['TP']):>9}{int(v['FP']):>9}{int(v['FN']):>9}"
            f"{v.get('n_instance_sequences', ''):>11}{v.get('n_instance_frames', ''):>11}"
            f"{v.get('n_unique_instances', ''):>8}"
        )
    for k in ["fp_bin_rule", "unmatched_fp_by_rule", "unknown_speed", "speed_source", "tgt_pixels_without_instance", "reference_model_metrics"]:
        print(f"{k}: {results[k]}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig):
    results = evaluate(cfg)
    print_table(results)
    if cfg.get("out_json"):
        with open(cfg.out_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

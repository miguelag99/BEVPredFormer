"""
Collapse the view transform of a checkpoint to a single layer.

The training loop in DefAttnVT never rebound its running query, so only the last
of the n_layers ever reached the output and the rest stayed at their random
initialisation. These weights are therefore a 1-layer BEVFormer already. This
script rewrites a checkpoint to say so:

  - view_transform.<group>.<last>.*  ->  view_transform.<group>.0.*
  - every other view_transform layer index is dropped
  - backbone stage 3 and the sparse encoder projection shortcuts are dropped,
    since the fixed model no longer builds them
  - optimizer / scheduler / loop state is dropped, because its parameter indices
    are positional and would silently land on the wrong tensors after the above

The result is a warm-start checkpoint for a model configured with
model.net.view_transform.n_layers=1. The BEV outputs are unchanged.

The input file is never modified or deleted; the output path must not exist.

Usage (inside Docker container):
    uv run scripts/migrate_checkpoint.py checkpoints/BEVPredFormer_scale05.ckpt
    uv run scripts/migrate_checkpoint.py checkpoints/a.ckpt -o checkpoints/b.ckpt
    uv run scripts/migrate_checkpoint.py checkpoints/*.ckpt
"""

import argparse
import os
from pathlib import Path

import torch

# Torch > 2.6 defaults to weights_only loads, which these checkpoints predate.
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

VT_PREFIX = "net.view_transform."
VT_GROUPS = [
    "sa_layers",
    "sa_norm_layers",
    "ca_layers",
    "ca_norm_layers",
    "mlp_layers",
    "last_norm_layers",
]

# Branches the fixed model no longer builds.
DROP_PREFIXES = (
    "net.backbone.model.stages_3.",
    "net.decoder.encoder.layer_2.0.downsample.",
    "net.decoder.encoder.layer_3.0.downsample.",
)

# Training state keyed by parameter position, invalid once tensors are removed.
DROP_TOP_LEVEL = ("optimizer_states", "lr_schedulers", "loops", "callbacks")


def _split_vt_key(key):
    """Return (group, layer_index, remainder) for a view transform key, else None."""
    if not key.startswith(VT_PREFIX):
        return None
    rest = key[len(VT_PREFIX) :]
    group, _, rest = rest.partition(".")
    if group not in VT_GROUPS:
        return None
    idx, _, remainder = rest.partition(".")
    if not idx.isdigit():
        return None
    return group, int(idx), remainder


def collapse_state_dict(state_dict):
    """Keep only the last view transform layer, renamed to index 0."""
    indices = set()
    for key in state_dict:
        parsed = _split_vt_key(key)
        if parsed is not None:
            indices.add(parsed[1])
    if not indices:
        raise ValueError("no view_transform layers found; is this a BEVPredFormer checkpoint?")
    keep = max(indices)

    out, dropped = {}, {"vt_layers": 0, "dead_branches": 0}
    for key, value in state_dict.items():
        if key.startswith(DROP_PREFIXES):
            dropped["dead_branches"] += 1
            continue
        parsed = _split_vt_key(key)
        if parsed is None:
            out[key] = value
            continue
        group, idx, remainder = parsed
        if idx != keep:
            dropped["vt_layers"] += 1
            continue
        out[f"{VT_PREFIX}{group}.0.{remainder}"] = value
    return out, keep, sorted(indices), dropped


def migrate(src: Path, dst: Path):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError(f"{src}: no state_dict, not a Lightning checkpoint")

    before = len(ckpt["state_dict"])
    state_dict, kept, indices, dropped = collapse_state_dict(ckpt["state_dict"])
    ckpt["state_dict"] = state_dict

    removed_top = [k for k in DROP_TOP_LEVEL if k in ckpt]
    for key in removed_top:
        del ckpt[key]

    # The collapse is only sound if exactly one layer survives, under index 0.
    survivors = {_split_vt_key(k)[1] for k in state_dict if _split_vt_key(k)}
    assert survivors == {0}, f"expected only layer 0 to survive, got {sorted(survivors)}"
    assert not any(k.startswith(DROP_PREFIXES) for k in state_dict)

    torch.save(ckpt, dst)

    print(f"{src.name} -> {dst.name}")
    print(f"  view transform layers {indices}, kept {kept} as layer 0")
    print(f"  state_dict {before} -> {len(state_dict)} tensors "
          f"({dropped['vt_layers']} view transform, "
          f"{dropped['dead_branches']} dead branch)")
    print(f"  dropped training state: {', '.join(removed_top) or 'none present'}")
    print(f"  {src.stat().st_size / 2**20:.0f} MiB -> "
          f"{dst.stat().st_size / 2**20:.0f} MiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output path, only valid with a single input "
             "(default: <name>_1layer.ckpt next to the input)",
    )
    args = parser.parse_args()

    if args.output is not None and len(args.checkpoints) > 1:
        parser.error("-o/--output takes a single input checkpoint")

    for src in args.checkpoints:
        if not src.is_file():
            parser.error(f"{src} does not exist")
        dst = args.output or src.with_name(f"{src.stem}_1layer{src.suffix}")
        if dst.resolve() == src.resolve():
            parser.error(f"{dst} would overwrite the input checkpoint")
        if dst.exists():
            parser.error(f"{dst} already exists, refusing to overwrite")
        migrate(src, dst)


if __name__ == "__main__":
    main()

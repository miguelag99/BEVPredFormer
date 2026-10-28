import itertools
import os
import time

from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import lightning as L
import torch
from omegaconf import DictConfig
from lightning import LightningDataModule
from torch.utils.flop_counter import FlopCounterMode

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from bevpredformer import utils

log = utils.get_pylogger(__name__)


def _spconv_num_pairs(m, x, out):
    """Number of (input site, output site) kernel pairs a spconv conv computes on, from the active
    site indices. Each pair is one Cin x Cout matmul-accumulate."""
    ksize, stride, pad, dil = m.kernel_size, m.stride, m.padding, m.dilation
    if m.inverse:  # same pairs as the strided op (conv or pool) sharing its indice_key, reversed
        d = x.indice_dict[m.indice_key]
        ksize, stride, pad, dil = d.ksize, d.stride, d.padding, d.dilation
        x, out = out, x
    b = x.indices[:, 0].long()
    coords = x.indices[:, 1:].long()
    pad = torch.tensor(pad, device=coords.device)
    dil = torch.tensor(dil, device=coords.device)
    stride = torch.tensor(stride, device=coords.device)
    if m.subm:
        in_shape = torch.tensor(x.spatial_shape, device=coords.device)
        occ = torch.zeros((x.batch_size, *x.spatial_shape), dtype=torch.bool, device=coords.device)
        occ[(b, *coords.T)] = True
    else:
        out_shape = torch.tensor(out.spatial_shape, device=coords.device)
    n = 0
    for off in itertools.product(*[range(k) for k in ksize]):
        off = torch.tensor(off, device=coords.device)
        if m.subm:  # output sites are the input sites; pair if the neighbour is active
            nb = coords + off * dil - pad
            valid = ((nb >= 0) & (nb < in_shape)).all(1)
            n += occ[(b[valid], *nb[valid].T)].sum().item()
        else:  # output site (c + pad - off * dil) / stride, if integral and in bounds
            oc = coords + pad - off * dil
            valid = ((oc % stride == 0) & (oc >= 0) & (oc // stride < out_shape)).all(1)
            n += valid.sum().item()
    return n


def count_flops(net, batch):
    """FLOPs (2 per multiply-add) of one forward pass. Aten ops are counted by FlopCounterMode;
    spconv convs bypass aten, so they are counted from their kernel pairs via forward hooks.
    Not counted: the sampling inside the custom deformable-attention CUDA kernel."""
    from spconv.pytorch.conv import SparseConvolution

    spconv_flops = [0]

    def hook(m, inputs, out):
        n = _spconv_num_pairs(m, inputs[0], out)
        spconv_flops[0] += 2 * n * m.in_channels * m.out_channels // m.groups

    handles = [m.register_forward_hook(hook) for m in net.modules() if isinstance(m, SparseConvolution)]
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        net(**batch)
    for h in handles:
        h.remove()
    return fc.get_total_flops(), spconv_flops[0]

def train(cfg: DictConfig) -> Tuple[dict, dict]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    cfg.data.version = 'mini'
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataloader = datamodule.val_dataloader()
    iter_dataloader = iter(dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model).to(device)
    
    def move_to_device(batch, device):
        """Move batch of inputs to device if they are tensor."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: move_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [move_to_device(v, device) for v in batch]
        elif isinstance(batch, tuple):
            return tuple(move_to_device(v, device) for v in batch)
        else:
            return batch

    # Get a batch and move it to GPU
    batch = next(iter_dataloader)
    batch = move_to_device(batch, device)
    
    model.eval()
    # Warm up the model
    with torch.no_grad():
        for _ in range(cfg.get("num_warmup_iters", 10)):
            _ = model(batch)
    torch.cuda.synchronize()

    # Reset after warm-up so cudnn.benchmark's algorithm search workspaces don't inflate the peak.
    torch.cuda.reset_peak_memory_stats()
    static_mb = torch.cuda.memory_allocated() / 1024**2  # weights + batch

    with torch.no_grad():
        # Run the model on the batch multiple times and compute average
        num_samples = cfg.get("num_iters", 50)
        elapsed_times = []
        
        for _ in range(num_samples):
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            output = model(batch)
            t2.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded
            elapsed_times.append(t1.elapsed_time(t2))
        
        avg_elapsed_time = sum(elapsed_times) / num_samples
        log.info(f"Average elapsed time for model inference over {num_samples} runs: {avg_elapsed_time:.2f} ms")
        log.info(f"Standard deviation: {torch.tensor(elapsed_times).std().item():.2f} ms")
        log.info(f"Batch size: {batch['imgs'].shape[0]}")
        log.info(f"Static memory (weights + batch): {static_mb:.0f} MB")
        log.info(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
        log.info(f"Peak reserved VRAM: {torch.cuda.max_memory_reserved() / 1024**2:.0f} MB")

    log.info(f"Params: {sum(p.numel() for p in model.net.parameters()) / 1e6:.2f} M")
    aten_flops, spconv_flops = count_flops(model.net, batch)
    log.info(f"GFLOPs: {(aten_flops + spconv_flops) / 1e9:.1f} (aten {aten_flops / 1e9:.1f} + spconv {spconv_flops / 1e9:.1f})")
                
    
    return 0, 0
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="val.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.modif_config_based_on_flags(cfg)
    
    utils.extras(cfg)

    train(cfg)



if __name__ == "__main__":
    main()

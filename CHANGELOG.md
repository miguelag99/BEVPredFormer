# BEVPredFormer Changelog

- **v2.0.0** (09-2026)
  - Upgrade to PyTorch 2.8 + CUDA 12.9, Lightning 2.5 and torchmetrics 1.8 (codebase synced with TGRIP).
  - Environment managed with uv (`pyproject.toml` + `uv.lock`); Docker image based on CUDA 12.9 / Ubuntu 24.04.
  - Multi-Scale Deformable Attention op with fp16/bf16 support (`trainer.precision=bf16-mixed`).
  - Removed unused sparse grid sample op (`ops/gs`) and `GridSampleVT`.
  - Default `min_visibility` set to 2 in data configs.
  - Instance post-processing from TGRIP: centerness added to the center scores, configurable `conf_threshold` and `nms_kernel_size` (`model.postproc_kwargs`).
  - VPQ is computed only over the valid (visible) region, as in TGRIP.
  - Model fixes from TGRIP: the view transform now rebinds its running query, so `n_layers` actually stacks. The released checkpoints only ever used their last layer, so the defaults move to `n_layers: 1`; convert existing checkpoints with `scripts/migrate_checkpoint.py`.
  - Dropped branches that never received a gradient (EfficientViT stage 4, sparse encoder projection shortcuts), so DDP no longer needs `find_unused_parameters`.
- **v1.1.0** (07-2025)
  - Multiple validation and plotting scripts.
  - Uploaded BEVPredFormer weights.
- **v1.0.0** (06-2025)
  - Initial release of BEVPredFormer.

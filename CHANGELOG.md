# BEVPredFormer Changelog

- **v1.2.0** (09-2026)
  - Upgrade to PyTorch 2.8 + CUDA 12.9, Lightning 2.5 and torchmetrics 1.8 (codebase synced with TGRIP).
  - Environment managed with uv (`pyproject.toml` + `uv.lock`); Docker image based on CUDA 12.9 / Ubuntu 24.04.
  - Multi-Scale Deformable Attention op with fp16/bf16 support; validation runs in bf16-mixed.
  - Removed unused sparse grid sample op (`ops/gs`) and `GridSampleVT`.
  - Default `min_visibility` set to 2 in data configs.
- **v1.1.0** (07-2025)
  - Multiple validation and plotting scripts.
  - Uploaded BEVPredFormer weights.
- **v1.0.0** (06-2025)
  - Initial release of BEVPredFormer.

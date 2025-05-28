# BEVPredFormer

Implementation of BEVPredFormer, a transformer-based model for predicting future trajectories in bird's-eye view (BEV) space.

1. Image feature extraction using EfficientViT or other models.
2. BEVFormer as the 3D projection module.
3. Sparse UNet for BEV feature processing.
4. Temporal module based on Predformer and DiffGuided module.
5. Multi-scale prediction heads.

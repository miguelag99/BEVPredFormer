# BEVPredFormer

<div align=center>
  <a href="https://github.com/miguelag99/BEVPredFormer/blob/main/CHANGELOG.md">
    <img src="https://img.shields.io/badge/Changelog-v1.1.0-2ea44f?style=for-the-badge" alt="CHANGELOG">
  </a>
  <a href="https://pytorch.org">
    <img src="https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?style=for-the-badge&logo=pytorch" alt="pytorch">
  </a>
  <a href="https://lightning.ai/docs/pytorch/stable/">
    <img src="https://img.shields.io/badge/Lightning-1.9.5-purple?style=for-the-badge&logo=lightning" alt="Lightning">
  </a>
  <a href="https://wandb.ai/">
    <img src="https://img.shields.io/badge/Wandb-gray?style=for-the-badge&logo=weightsandbiases" alt="wandb">
  </a>
  <a href="https://www.docker.com">
    <img src="https://img.shields.io/badge/Docker-gray?style=for-the-badge&logo=docker&logoColor=white&labelColor=%23007FFF" alt="Docker">
  </a>
</div>

Implementation of BEVPredFormer, a transformer-based model for predicting future trajectories in bird's-eye view (BEV) space.

1. Image feature extraction using EfficientViT or other models.
2. BEVFormer as the 3D projection module.
3. Sparse UNet for BEV feature processing.
4. Temporal module based on Predformer and DiffGuided module.
5. Multi-scale prediction heads.

## 1. NuScenes Dataset

Download the NuScenes dataset from the [official website](https://www.nuscenes.org/download) and extract the files in a folder with the following structure:

```bash
  nuscenes/
    ├──── maps/
    ├──── samples/
    ├──── sweeps/
    ├──── v1.0-trainval/
    └──── v1.0-mini/
```

Configure the path to the NuScenes dataset in the Makefile:

```bash
NUSCENES_PATH = /path/to/nuscenes
```

## 2. Installation and Usage

Build the Docker image with the following command:

```bash
make build
```

You can configure the following parameters of the image in the Makefile:

- `IMAGE_NAME`: Name of the generated Docker image.
- `TAG_NAME`: Tag of the generated Docker image.
- `USER_NAME`: Name of the user inside the Docker container.
- `NUSCENES_PATH`: Path to the NuScenes dataset.

Once the image is built, you can run the container with the following command:

```bash
make run
```

This command will run a bash inside the container and mount the current directory and dataset inside the container.

### 2.1 Training

To train any version of BEVPredFormer, you can use the following command inside the Docker container:

```bash
python bevpredformer/train.py
```

The different configuration parameters can be tuned in the different yaml files located in the *configs* directory.

It is recommended to use some of the pretrained models available:

- BEVPredformer_Backbone_05.ckpt: Pretrained model with EfficientViT backbone for semantic segmentation (no prediction head). Recommended to use as a freezed backbone to train prediction models. Keys to load and freeze in train.yaml: `'net.backbone', 'net.neck', 'net.view_transform', 'net.decoder', 'net.coord_selector' and 'net.query_gen'`.

## 3. Checkpoints

### 3.1 Segmentation Checkpoints

We provide several checkpoints for BEV semantic segmentation. The model uses three frame as input and predicts the semantic segmentation map for the current frame. These checkpoints are used later to train the prediction models.

| Checkpoint Name | Description | IoU | Download Link |
|-----------------|-------------|-----|--------------|
| BEVPredformer_Backbone_effvitL2_03.ckpt  | Long range with img resolution of 224x480 | 41.90 | [Download](https://github.com/miguelag99/BEVPredFormer/releases/download/v1.0.0/BEVPredformer_Backbone_effvitL2_03.ckpt) |
| BEVPredformer_Backbone_effvitL2_05.ckpt | Long range with img resolution of 448x800  | 44.11 | [Download](https://github.com/miguelag99/BEVPredFormer/releases/download/v1.0.0/BEVPredformer_Backbone_effvitL2_05.ckpt) |
| BEVPredformer_Backbone_effvitL2_1.ckpt  | Long range with img resolution of 640x1600 | 44.17 | [Download](https://github.com/miguelag99/BEVPredFormer/releases/download/v1.0.0/BEVPredformer_Backbone_effvitL2_1.ckpt)  |
| BEVPredformer_Backbone_effvitL2_05_short_range.ckpt  | Short range with img resolution of 448x800 | 70.25 | [Download](https://github.com/miguelag99/BEVPredFormer/releases/download/v1.0.0/BEVPredformer_Backbone_effvitL2_05_short_range.ckpt) |

### 3.2 Prediction Checkpoints

| Checkpoint Name | Description | IoU | VPQ | Download Link |
|-----------------|-------------|-----|-----|---------------|
| effvit_SpUnet_2TripletTST_256_ps4_scale05.ckpt | | 40.9 | 33.2 | [Download](https://github.com/miguelag99/BEVPredFormer/releases/download/v1.1.0/effvit_SpUnet_2TripletTST_256_ps4_scale05.ckpt) |
| effvit_SpUnet_2TripletTST_256_ps4_scale03.ckpt | | 38.8 | 31.0 | [Download](https://github.com/miguelag99/BEVPredFormer/releases/download/v1.1.0/effvit_SpUnet_2TripletTST_256_ps4_scale03.ckpt) |

## Contact

[![Static Badge](https://img.shields.io/badge/ORCID-0009--0008--5627--5325-green?style=flat&logo=orcid)
](https://orcid.org/0009-0008-5627-5325)

If you have any questions, feel free to contact me at [miguel.antunes@uah.es](mailto:miguel.antunes@uah.es).

# BEVPredFormer: Spatio-temporal Attention for BEV Instance Prediction in Autonomous Driving

<p align="center">
    <a href="https://www.miguelantunes.eu/">Miguel Antunes-García</a><sup>1</sup>,
    <a href="https://www.santimontiel.eu/">Santiago Montiel-Marín</a><sup>1</sup>,
    <a href="https://www.linkedin.com/in/fabio-sanchez-garcia/">Fabio Sánchez-García</a><sup>1</sup>,
</p>
<p align="center">
    <a href="https://rodrigogutierrezm.github.io/">Rodrigo Gutiérrez-Moreno</a><sup>1</sup>,
    <a href="https://scholar.google.es/citations?hl=es&user=IktmiSAAAAAJ">Rafael Barea</a><sup>1</sup>, and
    <a href="http://www.robesafe.uah.es/personal/bergasa/">Luis M. Bergasa</a><sup>1</sup>
</p>
<p align="center" style="font-size: 0.9em; font-style: italic;">
  <sup>1</sup> Universidad de Alcalá
</p>

<div align=center>
    <img src="https://img.shields.io/badge/Changelog-v1.2.0-2ea44f?style=for-the-badge" alt="CHANGELOG">
    <img src="https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg?style=for-the-badge&logo=pytorch" alt="pytorch">
    <img src="https://img.shields.io/badge/Lightning-2.5.4-purple?style=for-the-badge&logo=lightning" alt="Lightning">
</div>
<div align=center>
    <img src="https://img.shields.io/badge/Wandb-gray?style=for-the-badge&logo=weightsandbiases" alt="wandb">
    <img src="https://img.shields.io/badge/CUDA-12.9-76B900?style=for-the-badge&logo=nvidia" alt="CUDA">
    <img src="https://img.shields.io/badge/uv-gray?style=for-the-badge&logo=uv" alt="uv">
    <img src="https://img.shields.io/badge/Docker-gray?style=for-the-badge&logo=docker&logoColor=white&labelColor=%23007FFF" alt="Docker">
    <a href="https://arxiv.org/abs/2604.02930">
      <img src="https://img.shields.io/badge/arxiv-black?style=for-the-badge&logo=arxiv" alt="arxiv">
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

The environment is managed with [uv](https://docs.astral.sh/uv/) and uses PyTorch 2.8 with CUDA 12.9. All dependencies are defined in `pyproject.toml` and pinned in `uv.lock`.

### 2.1 Docker (recommended)

Build the Docker image (CUDA 12.9 devel + uv) with the following command:

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

This command will run a bash inside the container and mount the current directory and dataset inside the container. The first time the container is launched, the entrypoint runs `uv sync`, which creates the `.venv` inside the repository and compiles the Multi-Scale Deformable Attention CUDA op. Following runs reuse the existing `.venv`.

> **Note:** the `.venv` is created for the container's Python. If you also create one outside the container, delete it (`rm -rf .venv`) before launching the container so it can be rebuilt.

### 2.2 Local installation (without Docker)

Requirements: [uv](https://docs.astral.sh/uv/getting-started/installation/), an NVIDIA driver compatible with CUDA 12.9 and the CUDA 12.9 toolkit (`nvcc`) to compile the deformable attention op.

```bash
uv sync
```

If you modify the CUDA sources of the deformable attention op, `uv sync` will not rebuild it. Force the rebuild with:

```bash
uv sync --reinstall-package multiscaledeformableattention
```

### 2.3 Training

To train any version of BEVPredFormer, you can use the following command inside the Docker container:

```bash
uv run python bevpredformer/train.py
```

### 2.4 Validation

To evaluate a checkpoint (path configured in `configs/val.yaml`):

```bash
uv run python bevpredformer/val.py
```

Validation runs with `bf16-mixed` precision by default. Use `trainer.precision=32` to evaluate in full precision.

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

## Citation
Please, consider citing thiw work with:

```bibtex
@misc{antunesgarcía2026bevpredformerspatiotemporalattentionbev,
      title={BEVPredFormer: Spatio-temporal Attention for BEV Instance Prediction in Autonomous Driving}, 
      author={Miguel Antunes-García and Santiago Montiel-Marín and Fabio Sánchez-García and Rodrigo Gutiérrez-Moreno and Rafael Barea and Luis M. Bergasa},
      year={2026},
      eprint={2604.02930},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.02930}, 
}
```

## Contact

[![Static Badge](https://img.shields.io/badge/ORCID-0009--0008--5627--5325-green?style=flat&logo=orcid)
](https://orcid.org/0009-0008-5627-5325)

If you have any questions, feel free to contact me at [miguel.antunes@uah.es](mailto:miguel.antunes@uah.es).

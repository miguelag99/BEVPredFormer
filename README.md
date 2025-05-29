# BEVPredFormer

<div align=center>
  <a href="https://github.com/miguelag99/BEVPredFormer/blob/main/CHANGELOG.md">
    <img src="https://img.shields.io/badge/Changelog-v0.0.0-2ea44f?style=for-the-badge" alt="CHANGELOG">
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

## NuScenes Dataset

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

## Installation and Usage

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

## Contact

[![Static Badge](https://img.shields.io/badge/ORCID-0009--0008--5627--5325-green?style=flat&logo=orcid)
](https://orcid.org/0009-0008-5627-5325)

If you have any questions, feel free to contact me at [miguel.antunes@uah.es](mailto:miguel.antunes@uah.es).

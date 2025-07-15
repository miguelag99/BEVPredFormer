import cv2
import numpy as np
import hydra
import shutil
import pyrootutils
import pytorch_lightning as L
import torch
import imageio.v3 as imageio
import matplotlib.pyplot as plt

from typing import Optional, Dict
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from tqdm.auto import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from bevpredformer import utils
from visualize_batch import generate_gt_instance_pred

log = utils.get_pylogger(__name__)

EGO_DIMS = (4.087,1.562,1.787)

def visualise(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    cfg.data.prefetch_factor = 0
    cfg.data.num_workers = 1
    cfg.data.batch_size = 1
    cfg.data.normalize_img = True
    # cfg.data.version = 'mini'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataset = datamodule.val_dataloader().dataset
    
    ckpt = utils.get_ckpt_from_path(cfg.ckpt.path)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = utils.load_state_model(
        model,
        ckpt,
        cfg.ckpt.model.freeze,
        cfg.ckpt.model.load,
        verbose=1,
    ).to(device)
    
    # Remove the default output directory if it exists
    output_path = Path(cfg.paths.output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    # Save it in the default visualization directory
    output_path = Path(*output_path.parts[:-1])
    
    log.info(f"Output path: {output_path}")

    for scene_id in tqdm(cfg.scene_ids):
        (output_path / f"scene_{scene_id}").mkdir(parents=True, exist_ok=True)
        
        x = dataset[scene_id]
        
        instance_pred_map = generate_gt_instance_pred(
            bev_bounds= cfg.data.grid,
            batch=x
        )

        # Save instance prediction map using cv2
        instance_pred_path = output_path / f"scene_{scene_id}" / "gt.png"
        imageio.imwrite(str(instance_pred_path), instance_pred_map)
        
        
        # Move to device and add batch dimension
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                x[k] = v.to(device).unsqueeze(0)
                
        with torch.inference_mode():
            model.eval()
            output = model(x)

        prediction = generate_val_instance_pred(
            bev_bounds=cfg.data.grid,
            batch=x,
            output=output['bev']
        )
                
        inst_pred_path = output_path / f"scene_{scene_id}" / "prediction.png"
        imageio.imwrite(str(inst_pred_path), prediction)


def generate_val_instance_pred(
    bev_bounds: DictConfig,
    batch: Dict,
    output: Dict
) -> np.ndarray:
    # Bird's-eye view parameters
    bev_resolution = torch.tensor(
        [row[2] for row in [bev_bounds.xbound, bev_bounds.ybound, bev_bounds.zbound]]
    )
    bev_start_position = torch.tensor(
        [
            row[0] + row[2] / 2.0
            for row in [bev_bounds.xbound, bev_bounds.ybound, bev_bounds.zbound]
        ]
    )
    bev_dimension = torch.tensor(
        [
            (row[1] - row[0]) / row[2]
            for row in [bev_bounds.xbound, bev_bounds.ybound, bev_bounds.zbound]
        ],
        dtype=torch.long,
    )
    
    bev_resolution, bev_start_position, bev_dimension = (
        bev_resolution.numpy(), bev_start_position.numpy(), bev_dimension.numpy()
    )
    
    ## Predicted instance prediction
    output['binimg'] = output['binimg'][0,1:,:,:].unsqueeze(0)
    output['flow'] = output['flow'][0,1:,:,:].unsqueeze(0)

    data = {
        'segmentation': output['binimg'],
        'instance_flow': output['flow'],
        'centerness': output['centerness']
    }

    consistent_instance_seg, matched_centers = utils.predict_instance_segmentation(
        data, compute_matched_centers=True,
        spatial_extent=(bev_bounds.xbound[1], bev_bounds.ybound[1])
    )

    first_instance_seg = consistent_instance_seg[0, 1]

    unique_ids = torch.unique(first_instance_seg).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = utils.generate_instance_colours(instance_map)
    vis_image = utils.plot_instance_map(first_instance_seg.cpu().numpy(), instance_map)
    trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    for instance_id in unique_ids:
        path = matched_centers[instance_id]
        for t in range(len(path) - 1):
            color = instance_colours[instance_id].tolist()
            cv2.line(trajectory_img, tuple(map(int,path[t])),
                        tuple(map(int,path[t+1])), color, 4)

    # Overlay arrows
    temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 0.0)
    mask = ~ np.all(trajectory_img == 0, axis=2)
    vis_image[mask] = temp_img[mask]
    
    # Plot ego pose at the center of the image with cv2 circle       
    pts = np.array([[EGO_DIMS[1]/2, EGO_DIMS[0]/2],
                    [EGO_DIMS[1]/2, -EGO_DIMS[0]/2],
                    [-EGO_DIMS[1]/2, -EGO_DIMS[0]/2],
                    [-EGO_DIMS[1]/2, EGO_DIMS[0]/2]])

    pts = np.round(
        (pts - bev_start_position[:2] + bev_resolution[:2] / 2.0) / bev_resolution[:2]
    ).astype(np.int32)
    vis_image = cv2.fillPoly(vis_image, [pts], (0, 0, 0))
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    
    return vis_image

@hydra.main(version_base="1.3", config_path="../configs", config_name="visualize.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    utils.modif_config_based_on_flags(cfg)
    visualise(cfg)



if __name__ == "__main__":
    main()

import cv2
import numpy as np
import hydra
import pyrootutils
import pytorch_lightning as L
import torch
import imageio.v3 as imageio
import matplotlib.pyplot as plt

from typing import Optional, Dict
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from tqdm.auto import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from bevpredformer import utils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.get_pylogger(__name__)

EGO_DIMS = (4.087,1.562,1.787)

def visualise(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    cfg.data.prefetch_factor = 0
    cfg.data.num_workers = 1
    cfg.data.batch_size = 1
    cfg.data.normalize_img = False
    # cfg.data.version = 'mini'

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    dataset = datamodule.val_dataloader().dataset

    torch.backends.cudnn.benchmark = True
    for scene_id in tqdm(cfg.scene_ids):
        x = dataset[scene_id]
        
        img_names = cfg.data.img_params.cams
        
        imgs = x['imgs'][-1]  # Present images
        bev_segments = x['binimg'][1:]  # BEV segment
        
        bev_segments_upsampled = []
        for bev_segment in bev_segments:
            # Use nearest neighbor interpolation to preserve class labels
            upsampled = torch.nn.functional.interpolate(
                bev_segment.unsqueeze(0),  # Add batch dimension
                scale_factor=2.0,
                mode='nearest'
            ).squeeze(0)  # Remove batch dimension
            bev_segments_upsampled.append(upsampled)
        bev_segments = torch.stack(bev_segments_upsampled)
        
        instance_pred_map = generate_gt_instance_pred(
            bev_bounds= cfg.data.grid,
            batch=x
        )
        # Upscale the instance prediction map by 2x
        instance_pred_map = cv2.resize(instance_pred_map, 
                                    (instance_pred_map.shape[1]*2, instance_pred_map.shape[0]*2), 
                                    interpolation=cv2.INTER_NEAREST)
        

        # Create a figure with subplots for images and BEV segmentation
        fig = plt.figure(figsize=(16, 8))
        
        # Create a list to store frames for the GIF
        frames = []
        
        # For each timestep, create a frame
        num_timesteps = len(bev_segments)
        for t in range(num_timesteps):
            # Create a figure for this timestep
            frame_fig = plt.figure(figsize=(16, 8))
            
            # Plot the six camera images (using the most recent images)
            for i, cam_name in enumerate(img_names):
                ax = frame_fig.add_subplot(3, 3, i+1)
                img = imgs[i].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.set_title(cam_name)
                ax.axis('off')
            
            # Plot the BEV segmentation for current timestep
            ax = frame_fig.add_subplot(3, 3, 9)
            bev = bev_segments[t].cpu().numpy()
            bev = bev.squeeze(0)
            ax.imshow(bev)
            ax.set_title(f'BEV Segmentation (t={t})')
            ax.axis('off')
            
            # Plot the gt final instance prediction
            ax = frame_fig.add_subplot(3, 3, 8)
            ax.imshow(instance_pred_map)
            ax.set_title('Instance Prediction')
            ax.axis('off')

            # Add timestep indicator to figure title
            frame_fig.suptitle(f"Scene: {scene_id} (Timestep {t+1}/{num_timesteps})")
            
            # Convert figure to image
            canvas = FigureCanvas(frame_fig)
            canvas.draw()
            image = np.array(canvas.renderer.buffer_rgba())
            
            # Add frame to list
            frames.append(image)
            
            # Close the figure to free memory
            plt.close(frame_fig)
        
        # Save as GIF
        gif_path = (
            cfg.paths.output_dir + f"/scene_{scene_id}_animation.gif"
        )
        imageio.imwrite(gif_path, frames, duration=2000)  # 2 seconds per frame (in ms)
        log.info(f"GIF animation saved to {gif_path}")
        
        plt.close(fig)


def generate_gt_instance_pred(
    bev_bounds: DictConfig,
    batch: Dict,
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

    # Generate gt instance prediction
    data = {
        'segmentation': batch['binimg_aug'][1:].unsqueeze(0),
        'instance_flow': batch['flow_map_aug'][1:].unsqueeze(0),
        'centerness': batch['centerness_aug'][1:].unsqueeze(0),
    }
    
    consistent_instance_seg, matched_centers = utils.generate_gt_instance_segmentation(
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
    
    pts = np.round((pts - bev_start_position[:2] + bev_resolution[:2] / 2.0) / bev_resolution[:2]).astype(np.int32)
    vis_image = cv2.fillPoly(vis_image, [pts], (0, 0, 0))
    vis_image =cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    return vis_image


@hydra.main(version_base="1.3", config_path="../configs", config_name="visualize.yaml")
def main(cfg: DictConfig) -> Optional[float]:

    utils.modif_config_based_on_flags(cfg)
    visualise(cfg)



if __name__ == "__main__":
    main()

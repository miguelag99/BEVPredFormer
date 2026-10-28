from typing import List

import torch
from einops import rearrange

from .common import CamProjector


class LSSProjector(CamProjector):
    """Lift-Splat-Shoot geometry (https://arxiv.org/abs/2008.05711).

    Instead of projecting BEV voxels to the cameras, lifts a frustum of depth bins at every
    image feature pixel to 3D and computes the BEV cell each frustum point falls in. The BEV
    grid follows CoordSelector: row 0 is XMAX, column 0 is YMAX, and cells are centered on the
    `spatial_bounds` linspace.
    """

    def __init__(
        self,
        spatial_bounds: List[float] = [-49.75, 49.75, -49.75, 49.75, -3.375, 5.375],
        bev_shape: List[int] = [200, 200, 8],
        dbound: List[float] = [2.0, 50.0, 1.0],
    ):
        super().__init__(spatial_bounds, voxel_ref="spatial", z_value_mode="zero")
        self.bev_shape = bev_shape
        self.dbound = dbound
        self.n_depth = len(torch.arange(*dbound))

        # Cell size and lower edge of the grid, per axis.
        res, edges = [], []
        for (vmin, vmax), size in zip(
            [spatial_bounds[0:2], spatial_bounds[2:4], spatial_bounds[4:6]], bev_shape
        ):
            res.append((vmax - vmin) / (size - 1))
            edges.append([vmin - res[-1] / 2, vmax + res[-1] / 2])
        self.register_buffer("bev_res", torch.tensor(res), persistent=False)
        self.register_buffer("bev_edges", torch.tensor(edges), persistent=False)

    def create_frustum(self, h, w, device):
        """(D, h, w, 3) frustum of (u, v, depth), with u, v at the centers of the feature pixels
        in the feature-scaled intrinsics."""
        depth = torch.arange(*self.dbound, dtype=torch.float, device=device)
        u = torch.arange(w, dtype=torch.float, device=device) + 0.5
        v = torch.arange(h, dtype=torch.float, device=device) + 0.5
        depth, v, u = torch.meshgrid(depth, v, u, indexing="ij")
        return torch.stack([u, v, depth], dim=-1)

    def get_geometry(self, rots, trans, intrins, bev_aug, egoTin_to_seq, h, w):
        """(bt, n, D, h, w, 3) frustum points in the spatial reference frame.

        Inverse of CamProjector.from_voxel_ref_to_cams: cameras -> ego with (rots, trans), then
        augmented ego -> spatial with egoTin_to_seq @ inv(bev_aug).
        """
        bt, n = rots.shape[:2]
        points = self.create_frustum(h, w, rots.device)
        points = torch.cat([points[..., :2] * points[..., 2:3], points[..., 2:3]], dim=-1)

        # Cameras to ego.
        cam_to_ego = rots @ torch.linalg.inv(intrins)
        points = torch.einsum("bnij,dhwj->bndhwi", cam_to_ego, points)
        points = points + trans.view(bt, n, 1, 1, 1, 3)

        # Augmented ego to spatial.
        aug_to_spatial = rearrange(egoTin_to_seq, "b t i j -> (b t) i j") @ torch.linalg.inv(
            rearrange(bev_aug, "b t i j -> (b t) i j")
        )
        points = torch.einsum("bij,bndhwj->bndhwi", aug_to_spatial[:, :3, :3], points)
        return points + aug_to_spatial[:, :3, 3].view(bt, 1, 1, 1, 1, 3)

    def forward(self, dict_mat, dict_shape, dict_vox={}):
        """Returns `lss_ranks`: (b, t, n * D * h * w) flat BEV cell index (row * Y + col) of each
        frustum point, or X * Y for points outside the grid."""
        b, t = dict_shape["b"], dict_shape["t"]
        h, w = dict_shape["Hfeats"], dict_shape["Wfeats"]
        X, Y, _ = self.bev_shape

        # Geometry in float32 under mixed precision: linalg.inv does not support bf16.
        with torch.autocast(device_type=dict_mat["rots"].device.type, enabled=False):
            points = self.get_geometry(
                *[dict_mat[k].float() for k in ["rots", "trans", "intrins", "bev_aug", "egoTin_to_seq"]],
                h,
                w,
            )

        points = points.flatten(1, 4)
        row = ((self.bev_edges[0, 1] - points[..., 0]) / self.bev_res[0]).floor().long()
        col = ((self.bev_edges[1, 1] - points[..., 1]) / self.bev_res[1]).floor().long()
        z = points[..., 2]
        valid = (
            (row >= 0) & (row < X)
            & (col >= 0) & (col < Y)
            & (z >= self.bev_edges[2, 0]) & (z < self.bev_edges[2, 1])
        )
        ranks = torch.where(valid, row * Y + col, X * Y)
        return {"lss_ranks": rearrange(ranks, "(b t) N -> b t N", b=b, t=t)}

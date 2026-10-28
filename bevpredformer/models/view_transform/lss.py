from typing import List

import torch
from einops import rearrange
from torch import nn

from bevpredformer.utils.debug import debug_hook


class LiftSplatVT(nn.Module):
    """Lift-Splat-Shoot view transform (https://arxiv.org/abs/2008.05711), adapted from FIERY.

    A 1x1 conv on the neck features predicts a depth distribution and context features, their
    outer product is splatted (sum pooled) into the BEV cells computed by LSSProjector. Works with
    any backbone, as it only sees the neck output.
    """

    def __init__(
        self,
        in_c: int = 128,
        out_c: int = 128,
        bev_shape: List[int] = [200, 200],
        dbound: List[float] = [2.0, 50.0, 1.0],
    ):
        super().__init__()
        self.register_forward_hook(debug_hook)

        self.C = out_c
        self.D = len(torch.arange(*dbound))
        self.bev_shape = tuple(bev_shape[:2])
        self.depth_layer = nn.Conv2d(in_c, self.D + self.C, kernel_size=1)

    def lift(self, img_feats):
        """(bt, n, c, h, w) -> (bt * n * D * h * w, C) depth weighted features."""
        x = self.depth_layer(rearrange(img_feats, "bt n c h w -> (bt n) c h w"))
        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D :].unsqueeze(2)
        return rearrange(x, "btn c d h w -> (btn d h w) c")

    def splat(self, x, ranks):
        """Sum the features of each BEV cell, with static shapes: points outside the grid go to
        an extra discard cell (index X * Y) dropped at the end.
        Adapted from tgrip.models.fiery.predictor.LiftSplatWarp.projection_to_birds_eye_view."""
        bt = ranks.shape[0]
        n_cells = self.bev_shape[0] * self.bev_shape[1]
        ranks = ranks + torch.arange(bt, device=ranks.device).unsqueeze(1) * (n_cells + 1)

        # Sum in float32, as the cumsum of the original VoxelsSumming.
        out = torch.zeros((bt * (n_cells + 1), self.C), dtype=torch.float, device=x.device)
        out.index_add_(0, ranks.flatten(), x.float())
        return out.view(bt, n_cells + 1, self.C)[:, :n_cells]

    def forward(self, query, query_pos, img_feats, dict_vox):
        """Same interface as DefAttnVT: `query` and `query_pos` are unused (expected None).

        Returns:
            - BEV features: (b, t, X * Y, C)
            - mask: None
        """
        ranks = dict_vox["lss_ranks"]
        b, t = ranks.shape[:2]

        x = self.lift(img_feats)
        x = self.splat(x, rearrange(ranks, "b t N -> (b t) N"))
        return rearrange(x, "(b t) N c -> b t N c", b=b, t=t), None

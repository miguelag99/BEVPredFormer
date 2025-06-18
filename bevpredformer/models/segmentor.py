import copy
from functools import partial
import pdb
from typing import Dict, List, Optional, OrderedDict, Tuple

import hydra
import spconv.pytorch as spconv
import torch
import numpy as np
from einops import rearrange
from torch import nn

from bevpredformer.models.common import Network
from bevpredformer.utils.debug import debug_hook



class BEVPredFormerSegmentor(Network):
    def __init__(
        self,
        # Modules
        backbone=None,
        neck=None,
        embedding=None,
        query_gen=None,
        projector=None,
        view_transform=None,
        autoencoder=None,
        temporal=None,
        use_future_ego=False,
        out_seq_len=1,
        in_seq_len=3,
        heads=None,
        in_c: Dict[str, int] = {},
        out_c: Dict[str, int] = {},
        in_shape: Dict[str, int] = {},
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projector=projector,
            view_transform=view_transform,
            autoencoder=autoencoder,
            temporal=temporal,
            heads=heads,
            in_c=in_c,
            in_shape=in_shape,
            out_c=out_c,
        )
        # Images
        self.embedding = embedding

        # View Transform
        self.query_gen = query_gen
        
        # Temporal        
        self.temporal = temporal
        self.use_future_ego = use_future_ego
        
        # Prediction
        self.out_seq_len = out_seq_len
        self.n_classes = len(heads.class_weights)
        
    # Decoder.
    def _prepare_decoder(self, query, hq_wq):
        # Alias
        b, nq, Nq, c = query.shape
        hq, wq = hq_wq
        query = rearrange(
            query, "b nq (hq wq) c -> (b nq) c hq wq", b=b, nq=nq, hq=hq, wq=wq, c=c
        )
        return query

    def forward_decoder(self, query, hq_wq):
        # Alias
        b, nq, Nq, c = query.shape
        query = self._prepare_decoder(query, hq_wq)
        query = self.decoder(query)
        if isinstance(query, spconv.SparseConvTensor):
            query = query.dense()
        return self._arrange_decoder(query, (b, nq))

    # Temporal.
    def forward_temporal(self, bev_query):
        if self.temporal is None:
            return bev_query
        else:
            bev_query = self.temporal(bev_query)
            return bev_query
            
    def forward(self, imgs, rots, trans, intrins, bev_aug, egoTin_to_seq, **kwargs):
        (
            dict_shape,
            dict_vox,
            dict_img,
            dict_mat,
        ) = self._common_init_backneck_prepare_vt(
            imgs, rots, trans, intrins, bev_aug, egoTin_to_seq
        )

        # Projector
        self._prepare_dict_vox(dict_vox, dict_shape)
        dict_vox.update(self.projector(dict_mat, dict_shape, dict_vox))
        # VT
        b_t = (dict_shape["b"], dict_shape["t"])
        query, hq_wq = self.query_gen(b_t)
        query_pos, hq_wq = self.query_gen(b_t)
        bev_query, *_ = self.view_transform(
            query, query_pos, dict_img["img_feats"], dict_vox
        )
        hq, wq = hq_wq
             
        # Optional: decoder
        if self.decoder is not None:
            bev_query = self.forward_decoder(bev_query, hq_wq)
        else:
            hq, wq = hq_wq
            bev_query = rearrange(
                bev_query, "b nq (hq wq) c -> b nq c hq wq", hq=hq, wq=wq
            )

        # Optional: calculate egomotion info
        if (
            self.use_future_ego
            and "future_egomotion" in kwargs
            and self.temporal is not None
        ):
            future_egomotion = kwargs["future_egomotion"]
            b, s, c = future_egomotion.shape
            _, tin, _, h, w = bev_query.shape
            future_egomotions_spatial = future_egomotion.view(b, s, c, 1, 1)
            future_egomotions_spatial = future_egomotions_spatial.expand(
                b, s, c, h, w
            )
            # At time 0, no egomotion so feed zero vector
            future_egomotions_spatial = torch.cat(
                [
                    torch.zeros_like(future_egomotions_spatial[:, :1]),
                    future_egomotions_spatial[:, : (tin - 1)],
                ],
                dim=1,
            )
            bev_query = torch.cat([bev_query, future_egomotions_spatial], dim=2)
            
        # Temporal
        bev_query = self.forward_temporal(bev_query)

        # Heads
        dict_out = self.heads(bev_query)
                
        for k, v in dict_out.items():
            if isinstance(v, torch.Tensor):
                dict_out[k] = rearrange(v, "(b t) c h w -> b t c h w", t=self.out_seq_len)
    
        return {"bev": dict_out}


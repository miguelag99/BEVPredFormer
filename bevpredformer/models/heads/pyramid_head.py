import torch

from torch import nn
from copy import deepcopy
from typing import List

from bevpredformer.utils.debug import debug_hook
from bevpredformer.models.layers import STconv, ConvNormAct


class MultiPyramidHead(nn.Module):
    def __init__(
        self,
        shared_out_c: int = 128,
        latent_dim: int = 64,
        in_seq_len: int = 3,
        out_seq_len: int = 6,
        # Outputs
        with_centr_offs: bool = False,
        with_hdmap: bool = False,
        hdmap_names: List[str] = [],
        with_binimg: bool = True,
        with_multiclass: bool = False,
        class_weights: List[float] = [],
        with_flow: bool = False,
    ):
        super().__init__()
        # assert with_flow and with_binimg, "Flow and binimg are required"
        self.register_forward_hook(debug_hook)
        dict_input = dict(
            with_binimg=with_binimg,
            with_centr_offs=with_centr_offs,
            with_hdmap=with_hdmap,
            with_flow=with_flow,
        )
        self.with_centr_offs = with_centr_offs
        self.with_flow = with_flow
        self.out_seq_len = out_seq_len
        self.in_seq_len = in_seq_len
        self.hdmap_names = hdmap_names
        self.with_multiclass = with_multiclass
        self.class_weights = class_weights
        self._get_layers(dict_input, shared_out_c, latent_dim)

    def _get_layers(self, dict_input, shared_out_c, latent_dim):
        # Unpack
        (with_binimg, with_centr_offs, with_hdmap, with_flow) = (
            dict_input["with_binimg"],
            dict_input["with_centr_offs"],
            dict_input["with_hdmap"],
            dict_input["with_flow"],
        )

        # Prepare out
        map_out = nn.ModuleDict()

        # Initialize layers.
        shared_in_c = shared_out_c * self.in_seq_len
        out_classes = len(self.class_weights) if self.with_multiclass else 1

        norm = nn.InstanceNorm2d
        
        # Segmentation head
        convnormact_conv_seg = nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[16, 24, 32, 48, 64],
                n_blocks=3,
            ),
            # Rearrange from (b, t, c, h, w) to (b*t, c, h, w)
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
            nn.Conv2d(latent_dim, out_channels=out_classes,
                        kernel_size=1, padding=0)
        )
        
        # Flow head
        convnormact_conv_flow = nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[16, 24, 32, 48, 64],
                n_blocks=3,
            ),
            # Rearrange from (b, t, c, h, w) to (b*t, c, h, w)
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
            nn.Conv2d(latent_dim, out_channels=2,
                        kernel_size=1, padding=0),
        )
        
        # Center and offset head
        convnormact_conv_offs = nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[16, 24, 32, 48, 64],
                n_blocks=3,
            ),
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
            nn.Conv2d(latent_dim, out_channels=2,
                        kernel_size=1, padding=0),
        )
        convnormact_conv_centerness = nn.Sequential(
            STconv(
                in_channels=shared_out_c,
                out_channels=latent_dim,
                in_seq_len=self.in_seq_len,
                out_seq_len=self.out_seq_len,
                middle_channels=[16, 24, 32, 48, 64],
                n_blocks=3,
            ),
            nn.Flatten(0, 1),
            ConvNormAct(latent_dim, latent_dim,
                        3, 1, False, norm, nn.ReLU),
            nn.Conv2d(latent_dim, out_channels=1,
                        kernel_size=1, padding=0)
        )
        
        # HDMap head
        if with_hdmap:
            convnormact_conv_chdmap = nn.Sequential(
                ConvNormAct(shared_out_c, latent_dim,
                            3, 1, False, norm, nn.ReLU),
                nn.Conv2d(
                    latent_dim, len(self.hdmap_names),
                    kernel_size=1, padding=0
                ),
            )

        # Initialize heads.
        if with_binimg:
            map_out.update({"binimg": deepcopy(convnormact_conv_seg)})
            
        if with_flow:
            map_out.update({"flow": deepcopy(convnormact_conv_flow)})

        if with_centr_offs:
            map_out.update(
                {
                    "offsets": deepcopy(convnormact_conv_offs),
                    "centerness": deepcopy(convnormact_conv_centerness),
                }
            )

        if with_hdmap:
            map_out.update({"hdmap": convnormact_conv_chdmap})
        self.map_out = map_out

    def forward_layers(self, x):
        out_dict = {}
        out_dict.update({k: (layer(x)) for k, layer in self.map_out.items()})
        return out_dict

    def _apply_final_activation(self, out_dict):
        """Since spconv can not apply sigmoid to sparse tensor, we do it here."""
        if self.with_centr_offs:
            feats = out_dict["centerness"]
            out_dict["centerness"] = torch.sigmoid(feats)
        return out_dict

    def forward(self, x):
        b, t, c, h, w = x.shape
        out_dict = self.forward_layers(x)
        out_dict = self._apply_final_activation(out_dict)
        return out_dict


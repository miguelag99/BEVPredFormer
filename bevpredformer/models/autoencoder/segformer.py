import torch
import torch.nn as nn

from transformers import SegformerModel, SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(
        self,
        name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        embed_dim: int = 128,
        out_channels: int = 128,
        head_up_kernel: int = 4,
        head_up_stride: int = 4,
    ):
        super().__init__()

        self.model =  SegformerModel.from_pretrained(
            name,
            return_dict=True,
            output_hidden_states=True
        )
        self.cfg = self.model.config

        # Adapt input dims to pretrained model
        self._init_stem(embed_dim)
        
        self.linear_c = SegformerForSemanticSegmentation(self.cfg).decode_head.linear_c
        self.linear_fuse = nn.Conv2d(
            self.cfg.decoder_hidden_size*self.cfg.num_encoder_blocks,
            out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1,
                                 affine=True, track_running_stats=True)
        self.upsampler = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=head_up_kernel,
            stride=head_up_stride,
        )
    
    def _init_stem(self, in_channels):
        old_stem = self.model.encoder.patch_embeddings[0].proj
        new_stem = nn.Conv2d(
            in_channels,
            old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=old_stem.bias is not None
        )
        self.model.encoder.patch_embeddings[0].proj = new_stem
        
    def forward(self, x):
        b, c, _, _ = x.shape
        x = self.model(x).hidden_states  # Tuple of torch tensors

        states = []
        h0, w0 = x[0].shape[-2:]
        for i, state in enumerate(x):
            h, w = state.shape[-2:]
            state = self.linear_c[i](state).permute(0, 2, 1)
            state = state.reshape(b, -1, h, w)
            state = nn.functional.interpolate(state, size=(h0, w0),
                                              mode='bilinear', align_corners=False)
            states.append(state)

        x = torch.cat(states, dim=1)
        x = self.linear_fuse(x)
        x = self.upsampler(x)
        x = self.bn(x)
        return x

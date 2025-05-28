# ------------------------------------------------------------------------
# PowerBEV
# Copyright (c) 2023 Peizheng Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from FIERY (https://github.com/wayveai/fiery)
# Copyright (c) 2021 Wayve Technologies Limited. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import List
import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

class Residual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        upsample=False,
        downsample=False,
    ):
        super().__init__()
        self._downsample = downsample
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, dilation=1, stride=2, output_padding=padding_size, padding=padding_size)
        elif downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, dilation=dilation, stride=2, padding=padding_size)
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False, dilation=dilation, padding=padding_size)
        
        self.layers = nn.Sequential(conv, nn.BatchNorm2d(out_channels), nn.LeakyReLU(inplace=True))

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': nn.Upsample(scale_factor=2, mode='bilinear')})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)
    
    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x


class Bottleneck(nn.Module):
    """expand + depthwise + pointwise"""
    def __init__(
        self,
        in_channels,
        out_channels=None,
        kernel_size=3,
        dilation=1,
        upsample=False,
        downsample=False,
        expand_ratio=2,
        dropout=0.0,
    ):
        super().__init__()
        self._downsample = downsample
        expand_channels = round(in_channels * expand_ratio)
        out_channels = out_channels or in_channels
        padding_size = ((kernel_size - 1) * dilation + 1) // 2

        if upsample:
            assert not downsample, 'downsample and upsample not possible simultaneously.'
            conv = nn.ConvTranspose2d(expand_channels, expand_channels, kernel_size=kernel_size, bias=False, dilation=1, stride=2, output_padding=padding_size, padding=padding_size, groups=expand_channels)
        elif downsample:
            conv = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, bias=False, dilation=dilation, stride=2, padding=padding_size, groups=expand_channels)
        else:
            conv = nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, bias=False, dilation=dilation, padding=padding_size, groups=expand_channels)
        
        self.layers = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, expand_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(inplace=True),
            # dw
            conv,
            nn.BatchNorm2d(expand_channels),
            nn.Hardswish(inplace=True),
            # pw-linear
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout),
            # SeModule(out_channels),
        )

        if out_channels == in_channels and not downsample and not upsample:
            self.projection = None
        else:
            projection = OrderedDict()
            if upsample:
                projection.update({'upsample_skip_proj': nn.Upsample(scale_factor=2, mode='bilinear')})
            elif downsample:
                projection.update({'upsample_skip_proj': nn.MaxPool2d(kernel_size=2, stride=2)})
            projection.update(
                {
                    'conv_skip_proj': nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': nn.BatchNorm2d(out_channels),
                }
            )
            self.projection = nn.Sequential(projection)

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        if self.projection is not None:
            if self._downsample:
                # pad h/w dimensions if they are odd to prevent shape mismatch with residual layer
                x = nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2), value=0)
            return x_residual + self.projection(x)
        return x_residual + x
    
    
conv_block = Residual # [Residual, Bottleneck]

# class MultiBranchSTconv(torch.nn.Module):
#     """"""
#     def __init__(
#         self,
#         cfg,
#         in_channels,
#     ):
#         super().__init__()
#         self.cfg = cfg
#         self.latent_dim = self.cfg.MODEL.STCONV.LATENT_DIM

#         self.segmentation_branch = STconv(cfg, in_channels, self.latent_dim)
#         self.segmentation_head = Head(self.latent_dim, len(self.cfg.SEMANTIC_SEG.WEIGHTS))

#         self.flow_branch = STconv(cfg, in_channels, self.latent_dim)
#         self.flow_head = Head(self.latent_dim, 2)

#         self.parameter_init()
        
#     def parameter_init(self):
#         if isinstance(self.segmentation_head.last_conv, nn.Conv2d):
#             self.segmentation_head.last_conv.bias = nn.parameter.Parameter(torch.tensor([4.6, 0.0], requires_grad=True))
#         if isinstance(self.flow_head.last_conv, nn.Conv2d):
#             self.flow_head.last_conv.bias = nn.parameter.Parameter(torch.tensor([0.0, 0.0], requires_grad=True))

#     def forward(self, x):
#         output = {}

#         segmentation_branch_output = self.branch_forward(x, self.segmentation_branch)
#         output['segmentation'] = self.head_forward(segmentation_branch_output, self.segmentation_head)
        
#         flow_branch_output = self.branch_forward(x, self.flow_branch)
#         output['instance_flow'] = self.head_forward(flow_branch_output, self.flow_head)
        
#         return output
    
#     @staticmethod
#     def branch_forward(x, branch):
#         return branch(x)
    
#     @staticmethod
#     def head_forward(x, head):
#         return head(x)
        
        
class STconv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 16,
        in_seq_len: int = 3,
        out_seq_len: int = 6,
        middle_channels: List[int] = [16, 24, 32, 48, 64],
        n_blocks: int = 3,
    ):
        super().__init__()
        # Data configs
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_past_curr_steps = in_seq_len
        self.num_waypoints = out_seq_len

        # Model configs
        self.num_features = middle_channels
        self.num_blocks = n_blocks

        # BEV Encoder
        self.BEV_encoder = STEncoder(
            num_features=[self.in_channels] + self.num_features,
            num_timesteps=self.num_past_curr_steps,
            num_blocks=self.num_blocks,
        )

        # BEV Predictor
        self.BEV_predictor = STPredictor(
            num_features=self.num_features,
            in_timesteps=self.num_past_curr_steps,
            out_timesteps=self.num_waypoints,
        )

        # BEV Decoder
        self.BEV_decoder = STDecoder(
            num_features=self.num_features[::-1] + [self.out_channels],
            num_timesteps=self.num_waypoints,
            num_blocks=self.num_blocks,
        )

    def forward(self, f_in):
        # BEV Encoder
        f_enc = self.BEV_encoder(f_in)  # list of N x T_in x C_i x H_i x W_i
        
        # BEV Predictor
        f_dec = self.BEV_predictor(f_enc)  # list of N x T_out x C_i x H_i x W_i

        # BEV Decoder
        f_out = self.BEV_decoder(f_dec)  # N x T_out x C x H x W

        # Output
        return f_out


class STEncoder(nn.Module):
    def __init__(
        self,
        num_features,
        num_timesteps,
        num_blocks=3,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_timesteps = num_timesteps

        self.conv = nn.ModuleList()
        for i in range(1, len(self.num_features)):
            stage = nn.Sequential()
            for j in range(1, num_blocks+1):
                stage.add_module(
                    f'downconv_{i}_{j}', conv_block(
                        in_channels=self.num_features[i - 1] * self.num_timesteps if j == 1 else self.num_features[i] * self.num_timesteps,
                        out_channels=self.num_features[i] * self.num_timesteps,
                        downsample=True if j == num_blocks else False
                    )
                )
            self.conv.append(stage)

    def forward(self, x):
        b, t, _, _, _ = x.shape
        x = x.reshape(b, -1, *x.shape[3:])

        feature_pyramid = []
        for _, stage in enumerate(self.conv):
            x = stage(x)
            feature_pyramid.append(x.reshape(b, t, -1, *x.shape[2:]))
            
        return feature_pyramid


class STPredictor(nn.Module):
    def __init__(
        self,
        num_features,
        in_timesteps,
        out_timesteps,
    ):
        super().__init__()
        self.predictor = nn.ModuleList()
        for nf in num_features:
            self.predictor.append(nn.Sequential(
                conv_block(nf * in_timesteps, nf * in_timesteps),
                conv_block(nf * in_timesteps, nf * in_timesteps),
                conv_block(nf * in_timesteps, nf * out_timesteps),
                conv_block(nf * out_timesteps, nf * out_timesteps),
                conv_block(nf * out_timesteps, nf * out_timesteps),
            ))

    def forward(self, x):
        assert len(x) == len(self.predictor), f'The number of input feature tensors ({len(x)}) must be the same as the number of STPredictor blocks {len(self.predictor)}.'
        
        y = []
        for i in range(len(x)):
            b, _, c, _, _ = x[i].shape
            x_temp = x[i].reshape(b, -1, *x[i].shape[3:])
            y.append(self.predictor[i](x_temp).reshape(b, -1, c, *x_temp.shape[2:]))
        
        return y


class STDecoder(nn.Module):
    def __init__(
        self,
        num_features,
        num_timesteps,
        num_blocks=3,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_timesteps = num_timesteps

        self.conv = nn.ModuleList()
        for i in range(1, len(self.num_features)):
            stage = nn.Sequential()
            for j in range(1, num_blocks+1):
                stage.add_module(
                    f'upconv_{i}_{j}', conv_block(
                        in_channels=self.num_features[i - 1] * 2 * self.num_timesteps if j == 1 else self.num_features[i] * self.num_timesteps,
                        out_channels=self.num_features[i] * self.num_timesteps,
                        upsample=True if j == 1 else False
                    )
                )
            self.conv.append(stage)

    def forward(self, x):
        assert isinstance(x, list)
        for i, stage in enumerate(self.conv):
            b, t, _, _, _ = x[-1-i].shape
            x_temp = x[-1-i]
            x_temp = x_temp.reshape(b, -1, *x_temp.shape[3:])

            if i == 0:
                y = x_temp.repeat(1, 2, 1, 1)
            else:
                if y.shape != x_temp.shape:
                    y = F.interpolate(y, size=x_temp.shape[2:], mode='bilinear', align_corners=True)
                y = torch.cat((y, x_temp), dim=1)
            y = stage(y)

        y = y.reshape((b, t, -1, *y.shape[2:]))
        return y


class STHead(nn.Module):
    def __init__(self, in_channels, out_channels, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid

        self.head = nn.Sequential(
            conv_block(in_channels, in_channels//2),
            conv_block(in_channels//2, in_channels//2),
            conv_block(in_channels//2, in_channels//4),
            conv_block(in_channels//4, in_channels//4),
        )

        self.last_conv = nn.Conv2d(in_channels//4, out_channels, kernel_size=3, padding=1, bias=True)
        if self.sigmoid:
            self.last_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = x.clone()
        b, t, _, _, _ = y.shape
        y = y.reshape(-1, *y.shape[2:])
        y = self.head(y)
        y = self.last_conv(y)
        if self.sigmoid:
            y = self.last_sigmoid(y)
        return y.reshape(b, t, *y.shape[1:])
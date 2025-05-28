from typing import Literal, Tuple
import torch
from torch import nn
from einops import rearrange

from bevpredformer.models.layers.gated_transformer import GatedTransformer

def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

class PredFormerTemporalProjector(nn.Module):
    # https://arxiv.org/abs/2410.04733
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        bev_size: Tuple[int, int] = (200, 200),
        embed_dim: int = 256,
        depth: int = 4,
        attn_depth: int = 1,
        num_heads: int = 8,
        patch_size: int = 4,
        num_past_frames: int = 3,
        num_future_frames: int = 6,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        up_type: Literal["upsample", "convtranspose"] = "convtranspose",
        attn_type: Literal[
            "BinTS", "BinSt", "TripletTST", "TripletSTS", "QuadTSST", "QuadSTTS"
        ] = "BinTS",
        query_type: Literal["learnable", "linear"] = "learnable",
        temp_fuser: nn.Module = None,
    ) -> None:
        """PredFormer Temporal Projector implementation from PredFormer: Modeling Motion in Spacetime
        https://arxiv.org/abs/2410.04733
        
        Args:
            in_channels: Input BEV feature channels
            out_channels: Output BEV feature channels
            bev_size: BEV image size
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            attn_depth: Number of attention layers
            num_heads: Number of attention heads
            patch_size: Patch size for spatial dimensions
            num_past_frames: Number of past frames
            num_future_frames: Number of future frames to predict
            dropout: Dropout rate
            mlp_ratio: MLP hidden dim ratio
            up_type: Upsampling method, either "upsample" or "convtranspose"
            attn_type: Attention type, either "BinTS" or "BinSt"
            query_type: Query generation type, either "learnable" or "linear"
            temp_fuser: Module to perform final temporal fusion projection
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.bev_size = bev_size

        # Patch embed transformation
        assert (bev_size[0] % patch_size == 0) and (bev_size[1] % patch_size == 0), (
            "BEV size must be divisible by patch size"
        )
        self.n_patches_per_bev = (bev_size[0] // patch_size) * (bev_size[1] // patch_size)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Attnt layers
        # Dynamically instantiate the appropriate attention layer based on attn_type
        assert attn_type in [
            "BinTS",
            "BinST",
            "TripletTST",
            "TripletSTS",
            "QuadTSST",
            "QuadSTTS",
        ], "Invalid attention type"
        
        layer_class = eval(f"PredFormerLayer{attn_type}")
        self.blocks = nn.ModuleList(
            [
                layer_class(
                    dim=embed_dim,
                    depth=attn_depth,
                    heads=num_heads,
                    dim_head=embed_dim // num_heads,
                    mlp_dim=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    attn_dropout=dropout,
                    drop_path=dropout,
                )
                for _ in range(depth)
            ]
        )
        
        # Future query
        self.query_type = query_type
        if query_type == "learnable":
            # Learnable query tokens for future frames.
            self.future_query = nn.Parameter(
                torch.zeros(1, num_past_frames, self.n_patches_per_bev, embed_dim)
            )
        elif query_type == "linear":
            self.query_linear = nn.Linear(
                num_past_frames*embed_dim,
                num_past_frames*embed_dim
            )
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            sinusoidal_embedding(self.n_patches_per_bev*num_past_frames, embed_dim),
            requires_grad=False
        ).view(1, num_past_frames, self.n_patches_per_bev, embed_dim)
        
        # Output projection to reconstruct BEV feature maps
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_channels),
        )
        assert up_type in ["upsample", "convtranspose"], "Invalid up_type"
        if up_type == "upsample":
            self.upsample = nn.Upsample(scale_factor=patch_size, mode="nearest")
        else:
            self.upsample = nn.ConvTranspose2d(
                out_channels*self.num_past_frames, out_channels*self.num_past_frames,
                kernel_size=patch_size, stride=patch_size
            )

        # Final temporal fusion
        if temp_fuser is None:
            # If not provided, use simple convolutional fusion
            self.temp_fuser = nn.Conv2d(
                3*out_channels, 6*out_channels, kernel_size=1, stride=1
            )
        else:
            self.temp_fuser = temp_fuser

    def forward(self, x):
        """
        x: [B, T, C, H, W] - T = num_past_frames
        """
        b, t, c, h, w = x.shape
        assert t == self.num_past_frames, f"Expected {self.num_past_frames} past frames, got {t}"
        
        # Extract patches from past frames
        patches = [self.patch_embed(x[:, i]) for i in range(t)]
        _, _, h_p, w_p = patches[0].shape
        
        # Reshape patches into tokens
        patches = [p.flatten(2).transpose(1, 2) for p in patches]  # [B, num_patches, embed_dim]
        tokens = torch.stack(patches, dim=1)  # [B, T, num_patches, embed_dim]
        
        # Future query tokens
        if self.query_type == "learnable":
            queries = self.future_query.expand(b, -1, -1, -1)  # [B, num_past_frames, num_patches, embed_dim]
        elif self.query_type == "linear":
            queries = self.query_linear(
                rearrange(tokens,'b tp np c -> b np (tp c)')
            )
            queries = rearrange(
                queries, 'b np (tp c) -> b tp np c', tp = self.num_past_frames
            )
        tokens = tokens + self.pos_embed.to(tokens.device)
        queries = queries + self.pos_embed.to(queries.device)
        
        for blk in self.blocks:
            queries = blk(queries, tokens)
        
        # Project to out channels
        queries = self.out_proj(queries)
        queries = rearrange(queries, 'b tp np c -> b tp c np')
        
        # Upsample to BEV size
        ph, pw = self.bev_size[0] // self.patch_size, self.bev_size[1] // self.patch_size
        queries = rearrange(queries, 'b tp c (ph pw) -> b (tp c) ph pw', ph=ph, pw=pw)
        queries = self.upsample(queries)
        
        # Final temporal fusion [b, tp*c, h, w] -> [b, tf*c, h, w]
        queries = self.temp_fuser(queries)
        queries = rearrange(
            queries, 'b (tf c) h w -> b tf c h w', tf=self.num_future_frames
        )
        
        return queries
        
## Binary TS and ST

class PredFormerLayerBinTS(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super(PredFormerLayerBinTS, self).__init__()

        self.ts_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.ts_space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )

    def forward(self, q, kv):
        """
        Args:
            q (torch.Tensor): query tensor of shape [b, t, n, dim]
            kv (torch.Tensor): key-value tensor of shape [b, t, n, dim]

        Returns:
            torch.Tensor: output tensor of shape [b, t, n, dim]
        """
        b, t, n, _ = q.shape
        x_ts = q
        kv_ts = kv

        # ts-t branch
        x_ts = rearrange(x_ts, "b t n d -> b n t d")
        x_ts = rearrange(x_ts, "b n t d -> (b n) t d")
        kv_ts = rearrange(kv_ts, "b t n d -> b n t d")
        kv_ts = rearrange(kv_ts, "b n t d -> (b n) t d")
        x_ts = self.ts_temporal_transformer(x_ts, kv_ts)

        # ts-s branch
        x_ts = rearrange(x_ts, "(b n) t d -> b n t d", b=b)
        kv_ts = rearrange(kv_ts, "(b n) t d -> b n t d", b=b)
        x_ts = rearrange(x_ts, "b n t d -> b t n d")
        kv_ts = rearrange(kv_ts, "b n t d -> b t n d")
        x_ts = rearrange(x_ts, "b t n d -> (b t) n d")
        kv_ts = rearrange(kv_ts, "b t n d -> (b t) n d")
        x_ts = self.ts_space_transformer(x_ts, kv_ts)

        # ts output branch
        x_ts = rearrange(x_ts, "(b t) n d -> b t n d", b=b)

        return x_ts

class PredFormerLayerBinST(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super(PredFormerLayerBinST, self).__init__()

        self.st_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.st_space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )

    def forward(self, q, kv):
        """
        Args:
            q (torch.Tensor): query tensor of shape [b, t, n, dim]
            kv (torch.Tensor): key-value tensor of shape [b, t, n, dim]

        Returns:
            torch.Tensor: output tensor of shape [b, t, n, dim]
        """
        b, t, n, _ = q.shape
        x_st = q
        kv_st = kv
        
        # st-s branch
        x_st = rearrange(x_st, 'b t n d -> (b t) n d')
        kv_st = rearrange(kv_st, 'b t n d -> (b t) n d')
        x_st = self.st_space_transformer(x_st, kv_st)
        
        # st-t branch
        x_st = rearrange(x_st, '(b t) ... -> b t ...', b=b)
        kv_st = rearrange(kv_st, '(b t) ... -> b t ...', b=b)
        x_st = x_st.permute(0, 2, 1, 3) # b n T d        
        kv_st = kv_st.permute(0, 2, 1, 3)
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')  
        kv_st = rearrange(kv_st, 'b n t d -> (b n) t d')
        x_st = self.st_temporal_transformer(x_st, kv_st)

        # st output branch     
        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d', b=b) 
        
        return x_st

## Triplet TST and STS

class PredFormerLayerTripletTST(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super(PredFormerLayerTripletTST, self).__init__()

        self.temporal_transformer_first = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.temporal_transformer_second = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )

    def forward(self, q, kv):
        """
        Args:
            q (torch.Tensor): query tensor of shape [b, t, n, dim]
            kv (torch.Tensor): key-value tensor of shape [b, t, n, dim]

        Returns:
            torch.Tensor: output tensor of shape [b, t, n, dim]
        """
        
        b, t, n, _ = q.shape
        x_t = q
        kv_tst = kv 
        
        # t branch (first temporal)
        x_t = rearrange(x_t, 'b t n d -> b n t d')
        x_t = rearrange(x_t, 'b n t d -> (b n) t d')
        kv_tst = rearrange(kv_tst, 'b t n d -> b n t d')
        kv_tst = rearrange(kv_tst, 'b n t d -> (b n) t d')
        x_t = self.temporal_transformer_first(x_t, kv_tst)
        
        # s branch (space)
        x_ts = rearrange(x_t, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d') 
        kv_tst = rearrange(kv_tst, '(b n) t d -> b n t d', b=b)
        kv_tst = rearrange(kv_tst, 'b n t d -> b t n d')
        kv_tst = rearrange(kv_tst, 'b t n d -> (b t) n d')
        x_ts = self.space_transformer(x_ts, kv_tst)
        
        # t branch (second temporal)
        x_tst = rearrange(x_ts, '(b t) n d -> b t n d', b=b)
        x_tst = rearrange(x_tst, 'b t n d -> b n t d')
        x_tst = rearrange(x_tst, 'b n t d -> (b n) t d')
        kv_tst = rearrange(kv_tst, '(b t) n d -> b t n d', b=b)
        kv_tst = rearrange(kv_tst, 'b t n d -> b n t d')
        kv_tst = rearrange(kv_tst, 'b n t d -> (b n) t d')
        x_tst = self.temporal_transformer_second(x_tst, kv_tst)

        # ts output branch     
        x_tst = rearrange(x_tst, '(b n) t d -> b n t d', b=b)
        x_tst = rearrange(x_tst, 'b n t d -> b t n d', b=b) 

        return x_tst

class PredFormerLayerTripletSTS(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super(PredFormerLayerTripletSTS, self).__init__()

        self.space_transformer_first = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.space_transformer_second = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )

    def forward(self, q, kv):
        """
        Args:
            q (torch.Tensor): query tensor of shape [b, t, n, dim]
            kv (torch.Tensor): key-value tensor of shape [b, t, n, dim]

        Returns:
            torch.Tensor: output tensor of shape [b, t, n, dim]
        """
        b, t, n, _ = q.shape        
        x_s = q
        kv_sts = kv
        
        # space branch (first)
        x_s = rearrange(x_s, 'b t n d -> (b t) n d')
        kv_sts = rearrange(kv_sts, 'b t n d -> (b t) n d')
        x_s = self.space_transformer_first(x_s, kv_sts)
        
        # temporal branch
        x_st = rearrange(x_s, '(b t) ... -> b t ...', b=b)  
        x_st = x_st.permute(0, 2, 1, 3)  # b n t d        
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')  
        kv_sts = rearrange(kv_sts, '(b t) ... -> b t ...', b=b)
        kv_sts = kv_sts.permute(0, 2, 1, 3)
        kv_sts = rearrange(kv_sts, 'b n t d -> (b n) t d')
        x_st = self.temporal_transformer(x_st, kv_sts)
        
        # space branch (second)     
        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        kv_sts = rearrange(kv_sts, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d') 
        kv_sts = rearrange(kv_sts, 'b n t d -> b t n d') 
        
        x_sts = rearrange(x_st, 'b t n d -> (b t) n d') 
        kv_sts = rearrange(kv_sts, 'b t n d -> (b t) n d')
        x_sts = self.space_transformer_second(x_sts, kv_sts)

        # ts output branch     
        x_sts = rearrange(x_sts, '(b t) n d -> b t n d', b=b)
        
        return x_sts

## Quadruple TSST and STTS

class PredFormerLayerQuadTSST(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super(PredFormerLayerQuadTSST, self).__init__()

        self.ts_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.ts_space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.st_space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.st_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )

    def forward(self, q, kv):
        b, t, n, _ = q.shape        
        x_ts = q    
        kv_ts = kv
        
        # ts-t branch
        x_ts = rearrange(x_ts, 'b t n d -> b n t d')
        x_ts = rearrange(x_ts, 'b n t d -> (b n) t d')
        kv_ts = rearrange(kv_ts, 'b t n d -> b n t d')
        kv_ts = rearrange(kv_ts, 'b n t d -> (b n) t d')
        x_ts = self.ts_temporal_transformer(x_ts, kv_ts)
        
        # ts-s branch
        x_ts = rearrange(x_ts, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d')
        kv_ts = rearrange(kv_ts, '(b n) t d -> b n t d', b=b)
        kv_ts = rearrange(kv_ts, 'b n t d -> b t n d')
        kv_ts = rearrange(kv_ts, 'b t n d -> (b t) n d')
        x_ts = self.ts_space_transformer(x_ts, kv_ts)


        # ts output branch     
        x_ts = rearrange(x_ts, '(b t) n d -> b t n d', b=b)
        kv_ts = rearrange(kv_ts, '(b t) n d -> b t n d', b=b)  
        
        x_st = x_ts
        kv_st = kv_ts
        
        # st-s branch
        x_st = rearrange(x_st, 'b t n d -> (b t) n d')
        kv_st = rearrange(kv_st, 'b t n d -> (b t) n d')
        x_st = self.st_space_transformer(x_st, kv_st)
        
        # st-t branch
        x_st = rearrange(x_st, '(b t) ... -> b t ...', b=b)  
        x_st = x_st.permute(0, 2, 1, 3) # b n T d        
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')
        kv_st = rearrange(kv_st, '(b t) ... -> b t ...', b=b)
        kv_st = kv_st.permute(0, 2, 1, 3)
        kv_st = rearrange(kv_st, 'b n t d -> (b n) t d')
        x_st = self.st_temporal_transformer(x_st, kv_st)

        # st output branch     
        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d', b=b) 
        
        return x_st

class PredFormerLayerQuadSTTS(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        depth: int = 1,
        heads: int = 8,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super(PredFormerLayerQuadSTTS, self).__init__()

        self.st_space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.st_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.ts_temporal_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )
        self.ts_space_transformer = GatedTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, attn_dropout, drop_path
        )

    def forward(self, q, kv):
        b, t, n, _ = q.shape        
        x_st = q    
        kv_st = kv
    
        # st-s branch
        x_st = rearrange(x_st, 'b t n d -> (b t) n d')
        kv_st = rearrange(kv_st, 'b t n d -> (b t) n d')
        x_st = self.st_space_transformer(x_st, kv_st)
        
        # st-t branch
        x_st = rearrange(x_st, '(b t) ... -> b t ...', b=b)  
        x_st = x_st.permute(0, 2, 1, 3) # b n T d        
        x_st = rearrange(x_st, 'b n t d -> (b n) t d')
        kv_st = rearrange(kv_st, '(b t) ... -> b t ...', b=b)
        kv_st = kv_st.permute(0, 2, 1, 3)
        kv_st = rearrange(kv_st, 'b n t d -> (b n) t d')
        x_st = self.st_temporal_transformer(x_st, kv_st)

        # st output branch     
        x_st = rearrange(x_st, '(b n) t d -> b n t d', b=b)
        x_st = rearrange(x_st, 'b n t d -> b t n d', b=b)
        kv_st = rearrange(kv_st, '(b n) t d -> b n t d', b=b)
        kv_st = rearrange(kv_st, 'b n t d -> b t n d', b=b)
        
        x_ts = x_st
        kv_ts = kv_st
        
        # ts-t branch
        x_ts = rearrange(x_ts, 'b t n d -> b n t d')
        x_ts = rearrange(x_ts, 'b n t d -> (b n) t d')
        kv_ts = rearrange(kv_ts, 'b t n d -> b n t d')
        kv_ts = rearrange(kv_ts, 'b n t d -> (b n) t d')
        x_ts = self.ts_temporal_transformer(x_ts, kv_ts)
        
        # ts-s branch
        x_ts = rearrange(x_ts, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d')
        kv_ts = rearrange(kv_ts, '(b n) t d -> b n t d', b=b)
        kv_ts = rearrange(kv_ts, 'b n t d -> b t n d')
        kv_ts = rearrange(kv_ts, 'b t n d -> (b t) n d')
        x_ts = self.ts_space_transformer(x_ts, kv_ts)

        # ts output branch     
        x_ts = rearrange(x_ts, '(b t) n d -> b t n d', b=b)
        
        return x_ts
 
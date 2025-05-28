from typing import Literal
from torch import nn, einsum
from einops import rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# https://github.com/yyyujintang/PredFormer/blob/main/openstl/models/

class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class GatedTransformer(nn.Module):
    # Modified Gated Transformer with kv as input
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNormQKV(dim, AttentionQKV(dim, dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)       
            
    def forward(self, q, kv):
        x = q
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(q, kv))
            x = x + drop_path2(ff(x))
        return self.norm(x)
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNormQKV(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, kv, **kwargs):
        return self.fn(self.norm(q), self.norm(kv), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class AttentionQKV(nn.Module):
    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        # Projection for keys and values
        self.to_kv = nn.Linear(key_value_dim, self.inner_dim * 2, bias=False)
        
        # Projection for query
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            # nn.Dropout(dropout)
        )

        self.dropout = dropout

    def forward(self, query, key_value_features):
        # Batch size and input dimensions
        batch, query_len, _ = query.shape
        batch, kv_len, _ = key_value_features.shape
        h = self.heads

        # Project query
        q = self.to_q(query)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        # Project keys and values
        kv = self.to_kv(key_value_features)
        k, v = kv.chunk(2, dim=-1)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
       
        # # Compute attention scores
        # dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        # # Apply softmax to get attention weights
        # attn = dots.softmax(dim=-1)
        # # Apply attention weights to values
        # out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # # Rearrange and project output
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # out = self.to_out(out)
        
        ## New implementation
        # https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb
        
        # Apply dropout if in training mode
        use_dropout = 0.0 if not self.training else self.dropout

        # Perform scaled dot product attention
        context_vec = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=False,    # is_causal is usually False for cross-attention
        )  
        # Combine the attention heads by transposing and reshaping
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch, query_len, self.inner_dim)
        )
 
        out = self.to_out(context_vec)

        return out
    

# https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb

class MHAPyTorchScaledDotProduct(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape

        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(x)

        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)

        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv

        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_out)

        context_vec = self.proj(context_vec)

        return context_vec

class CrossAttentionPyTorchScaledDotProduct(nn.Module):
    def __init__(
        self,
        d_in_q,
        d_in_kv,
        d_out,
        num_heads,
        # context_length_q,
        # context_length_kv,
        dropout=0.0,
        qkv_bias=False,
    ):
        super().__init__()

        assert d_out % num_heads == 0, "embed_dim is indivisible by num_heads"

        self.num_heads = num_heads
        # self.context_length_q = context_length_q
        # self.context_length_kv = context_length_kv
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        # Linear layer to project the query input
        self.q_proj = nn.Linear(d_in_q, d_out, bias=qkv_bias)
        # Linear layer to project the key-value input (outputs 2 * d_out for keys and values)
        self.kv_proj = nn.Linear(d_in_kv, 2 * d_out, bias=qkv_bias)
        # Linear layer for the final projection of the combined context vector
        self.proj = nn.Linear(d_out, d_out)
        self.dropout = dropout

    def forward(self, q, kv):
        # Get the batch size, number of tokens, and embedding dimension for the query input
        batch_size_q, num_tokens_q, embed_dim_q = q.shape
        # Get the batch size, number of tokens, and embedding dimension for the key-value input
        batch_size_kv, num_tokens_kv, embed_dim_kv = kv.shape

        # Project the query input using the query projection layer
        queries = self.q_proj(q)
        # Reshape the queries to (batch_size_q, num_tokens_q, num_heads, head_dim) and then permute to (batch_size_q, num_heads, num_tokens_q, head_dim)
        queries = queries.view(
            batch_size_q, num_tokens_q, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        # (b, num_heads, num_tokens_q, head_dim)

        # Project the key-value input using the key-value projection layer
        kv_out = self.kv_proj(kv)
        # Reshape the key-value output to (batch_size_kv, num_tokens_kv, 2, num_heads, head_dim) and then permute to (2, batch_size_kv, num_heads, num_tokens_kv, head_dim)
        kv_out = kv_out.view(
            batch_size_kv, num_tokens_kv, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        # (2, b, num_heads, num_tokens_kv, head_dim)
        # Split the key-value output into keys and values
        keys, values = (
            kv_out[0],
            kv_out[1],
        )  # (b, num_heads, num_tokens_kv, head_dim)

        # Apply dropout if in training mode
        use_dropout = 0.0 if not self.training else self.dropout

        # Perform scaled dot product attention
        context_vec = nn.functional.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=use_dropout,
            is_causal=False,
        )  # is_causal is usually False for cross-attention

        # Combine the attention heads by transposing and reshaping
        context_vec = (
            context_vec.transpose(1, 2)
            .contiguous()
            .view(batch_size_q, num_tokens_q, self.d_out)
        )

        # Apply the final projection layer
        context_vec = self.proj(context_vec)

        return context_vec
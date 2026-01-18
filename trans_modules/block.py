import torch
from torch import nn, Tensor
from torch.nn import functional as F
from timm.models.vision_transformer import Attention, DropPath, LayerScale
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from typing import Optional
import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))

from model.trans_modules.embeddings import RotaryEmbedding

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    from timm.models.vision_transformer.Mlp
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpMixerBlock(nn.Module):
    def __init__(self, num_frames, d_model, 
                 mlp_ratio=4., norm=nn.LayerNorm, 
                 dropout=0.1, droppath=0.) -> None:
        super().__init__()
        mlp_hidden_dim1 = mlp_ratio * num_frames
        mlp_hidden_dim2 = mlp_ratio * d_model
        self.token_mixer = Mlp(num_frames, mlp_hidden_dim1, drop=dropout)
        self.channel_mixer = Mlp(d_model, mlp_hidden_dim2, drop=dropout)
        
        if isinstance(norm, nn.BatchNorm1d):
            self.norm1 = nn.Sequential(
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(d_model),
                Rearrange('b c n -> b n c')
            )
            self.norm2 = nn.Sequential(
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(d_model),
                Rearrange('b c n -> b n c')
            )
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        
        self.norm1 = norm(d_model)
        self.norm2 = norm(d_model)
        
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()

    def forward(self, x):
        B, T, C = x.shape
        
        # Token-mixer
        xt = self.norm1(x)
        xt = rearrange(xt, 'b t c -> b c t')
        xt = self.token_mixer(xt)
        xt = rearrange(xt, 'b c t -> b t c')
        x = x + self.drop_path(xt)

        # Channel-mixer
        xt = self.channel_mixer(self.norm2(x))
        x = x + self.drop_path(xt)
        return x


class SelfAttention(nn.Module):
    """
    seq_len:    Not None if using casual mask.
    """
    def __init__(self, d_model, seq_len=None, nhead=4, 
                 attn_dropout=0.1, proj_dropout=0.1,
                 use_rope=False) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5
        self.seq_len = seq_len
        self.use_rope = use_rope
        if self.use_rope:
            assert d_model // nhead % 2 == 0, 'head_dim must be even.'
            self.rotary_emb = RotaryEmbedding(d_model // nhead)

        self.proj_qkv = nn.Linear(d_model, 3 * d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        # regularization
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        # casual mask
        if seq_len is not None:
            self.casual_mask = nn.Parameter(torch.tril(torch.ones(seq_len, seq_len))[None, None, :, :], requires_grad=False)
    
    def forward(self, x,
                attn_mask=None,
                key_padding_mask=None, 
                is_causal=False,
                need_weights=False):
        """
        attn_mask:              (B, T, T), True means masked.
        key_padding_mask:       (B, T), True means padded.
        """
        B, T, C = x.size()
        qkv = self.proj_qkv(x).reshape(B, T, 3, self.nhead, C // self.nhead).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if self.use_rope:
            q, k = self.rotary_emb(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale       # (B, H, T, T)
        if is_causal:
            assert self.seq_len is not None, 'causal mask is not define!'
            # causal mask
            attn = attn.masked_fill(self.casual_mask[:,:,:T,:T]==0, float('-inf'))
        if key_padding_mask is not None:
            # padding mask
            attn = attn.masked_fill(key_padding_mask[:, None, None, :]>0, float('-inf'))
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask[:, None]>0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn_scores = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj_out(x)
        x = self.proj_drop(x)
        
        if need_weights:
            return x, attn_scores
        return x, None


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, mlp_ratio=4, 
                 dropout=0.1, droppath=0.,
                 init_values=None,
                 activation=nn.GELU, norm=nn.LayerNorm,
                 use_rope=False) -> None:
        super().__init__()

        if isinstance(norm, nn.BatchNorm1d):
            self.norm1 = nn.Sequential(
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(d_model),
                Rearrange('b c n -> b n c')
            )
            self.norm2 = nn.Sequential(
                Rearrange('b n c -> b c n'),
                nn.BatchNorm1d(d_model),
                Rearrange('b c n -> b n c')
            )
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, 
                                  nhead=nhead, 
                                  attn_dropout=dropout, 
                                  proj_dropout=dropout,
                                  use_rope=use_rope)
        self.ls1 = LayerScale(d_model, init_values=init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = Mlp(in_features=d_model,
                       hidden_features=mlp_hidden_dim,
                       act_layer=activation,
                       drop=dropout)
        self.ls2 = LayerScale(d_model, init_values=init_values) if init_values else nn.Identity()
        
    def forward(self, x:Tensor,
                attn_mask:Optional[Tensor]=None, 
                key_padding_mask:Optional[Tensor] = None, 
                need_weights=False,
                is_causal=False):
        """
        key_padding_mask:   For a binary mask, a `True` value indicates that the corresponding `key` value will be ignored for
            the purpose of attention.
        """
        # attn
        xt = self.norm1(x)
        xt, weights = self.attn(xt,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            is_causal=is_causal,
                            need_weights=need_weights)
        x = x + self.drop_path(self.ls1(xt))

        # ffn
        xt = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.ls2(xt))
        
        return x, weights

class CrossAttention(nn.Module):
    def __init__(self, q_dim, k_dim=None, v_dim=None, embed_dim=None, nhead=4,
                 attn_dropout=0.1, proj_dropout=0.1) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim or q_dim
        self.v_dim = v_dim or k_dim or q_dim
        self.embed_dim = embed_dim or q_dim
        assert self.embed_dim % nhead == 0
        self.nhead = nhead
        self.scale = (self.embed_dim // nhead) ** -0.5

        self.proj_q = nn.Linear(self.q_dim, self.embed_dim)
        self.proj_k = nn.Linear(self.k_dim, self.embed_dim)
        self.proj_v = nn.Linear(self.v_dim, self.embed_dim)
        self.proj_out = nn.Linear(self.embed_dim, self.q_dim)
        # regularization
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, q, k, v, mask=None):
        """
        parameter:
            q:          (B, Tq, Cq)
            k, v:       (B, Tk, Ck), k, v is the same.
        returns:
            x:          (B, Tq, Cq)
        """
        B, Tq, Cq = q.size()
        B, Tk, Ck = k.size()
        B, Tv, Cv = v.size()
        assert Tk == Tv, "requir that K and V is in same sequence length."
        
        q = self.proj_q(q).reshape(B, Tq, self.nhead, self.embed_dim//self.nhead).permute(0, 2, 1, 3)
        k = self.proj_k(k).reshape(B, Tk, self.nhead, self.embed_dim//self.nhead).permute(0, 2, 1, 3)
        v = self.proj_v(v).reshape(B, Tv, self.nhead, self.embed_dim//self.nhead).permute(0, 2, 1, 3)
        # q: (B, nhead, Tq, C)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn_scores = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Tq, self.embed_dim)
        x = self.proj_out(x)
        x = self.proj_drop(x)

        return x, attn_scores
    
class CrossAttnBlock(nn.Module):
    def __init__(self, q_dim, k_dim,
                 nhead=4, mlp_ratio=4,
                 dropout=0.1, droppath=0.) -> None:
        super().__init__()
        self.q_norm1 = nn.LayerNorm(q_dim)
        self.k_norm1 = nn.LayerNorm(k_dim)
        self.cross_attn = CrossAttention(q_dim, k_dim, k_dim, q_dim*4, nhead)

        self.drop_path = DropPath(droppath) if droppath > 0. else nn.Identity()

        self.q_norm2 = nn.LayerNorm(q_dim)
        hidden_mlp_dim = int(q_dim * mlp_ratio)
        self.mlp = Mlp(in_features=q_dim,
                        hidden_features=hidden_mlp_dim,
                        drop=dropout)
        
    def forward(self, x_q:Tensor, x_k:Tensor,
                need_weights=False):
        ## norm first
        # attn
        q = self.q_norm1(x_q)
        k = self.k_norm1(x_k)
        xt, weights = self.cross_attn(q, k, k)
        x_q = x_q + self.drop_path(xt)
        # ffn
        xt = self.mlp(self.q_norm2(x_q))
        x_q = x_q + self.drop_path(xt)

        if need_weights:
            return x_q, weights
        return x_q

if __name__ == '__main__':
    # 1. test causal attention
    x = torch.randn(2, 4, 12)
    padding_mask = torch.Tensor([[False, False, False, True],
                                  [False, False, True, True]]).to(dtype=torch.bool)
    cattn = SelfAttention(d_model=12, seq_len=4, nhead=4)
    cattn(x, is_causal=False, key_padding_mask=padding_mask)

    # 2. test cross attention
    q = torch.randn(2, 4, 3)
    v = torch.randn(2, 5, 6)
    cross_attn = CrossAttention(3, 6, 6, 12, nhead=3)
    y = cross_attn(q, v, v)

import torch
from torch import nn, Tensor
import numpy

from model.trans_modules.block import TransformerBlock
from model.trans_modules.embeddings import get_pos_encoder

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, mlp_ratio=4,
                 dropout=0.1, droppath=0., init_values=None,
                 activation=nn.GELU, norm=nn.LayerNorm, 
                 pos_type='rope',
                 depth=2,) -> None:
        super().__init__()

        if pos_type == 'fixed' or pos_type == 'learnable':
            self.pos_embed = get_pos_encoder(pos_type)(d_model=d_model, dropout=0)
        else:
            self.pos_embed = get_pos_encoder('none')()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             mlp_ratio=mlp_ratio,
                             dropout=dropout,
                             droppath=droppath,
                             init_values=init_values,
                             activation=activation,
                             norm=norm,
                             use_rope=(pos_type == 'rope'))
        for _ in range(depth)])

        self.norm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    @property
    def no_weight_decay(self):
        return {'cls_token', 'pos_embed'}

    def forward(self, x:Tensor):
        B, T, C = x.shape

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.pos_embed(x)
        for blk in self.blocks:
            x, attn_weights = blk(x, need_weights=True)
        self.attn_weights = attn_weights
        x = self.norm(x)
        return x        # (B, T+1, C)
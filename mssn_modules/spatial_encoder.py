import torch
from torch import nn, Tensor
import numpy

from einops.layers.torch import Rearrange
from einops import rearrange
from model.trans_modules.block import TransformerBlock
from model.bnt_modules.dec import DEC

class Mlp(nn.Module):
    def __init__(self, num_roi, hidden_features=256,
                 depth=3) -> None:
        super().__init__()
        assert depth >= 2
        num_conn = num_roi * (num_roi - 1) // 2
        layers = [
            nn.Linear(num_conn, hidden_features * 4, bias=False),
            nn.LayerNorm(hidden_features * 4),
            nn.LeakyReLU(),
        ]
        # the minimum dimension is 4
        in_dim = hidden_features*4
        for _ in range(depth-2):
            out_dim = max(in_dim // 2, hidden_features)
            layers.extend([
                nn.Linear(in_dim, out_dim, bias=False),
                nn.LayerNorm(out_dim),
                nn.LeakyReLU()
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, hidden_features))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        B, T, N, _ = x.shape

        # 1. flatten
        upper_indices = torch.triu_indices(N, N, offset=1)
        x = x[:, :, upper_indices[0], upper_indices[1]]    # (B, T, C)

        # 2. fc
        x = self.fc(x)                                     # (B, T, D)

        return x
    
class BNT(nn.Module):
    def __init__(self, num_roi, output_dim=512,
                 nhead=4, mlp_ratio=4,
                 dropout=.1, depth=2) -> None:
        super().__init__()
        self.transformer_block = nn.ModuleList([
            TransformerBlock(d_model=num_roi,
                             nhead=nhead,
                             mlp_ratio=mlp_ratio,
                             dropout=dropout,
                             norm=nn.LayerNorm)
        for _ in range(depth)])

        ## readout
        embed_dim = num_roi
        encoder_hidden_size = 32
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim*num_roi, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_hidden_size, embed_dim*num_roi),
        )
        self.pooling = DEC(cluster_number=num_roi//2,
                           hidden_dimension=embed_dim,
                           encoder=self.encoder,
                           orthogonal=True,
                           freeze_center=True,
                           project_assignment=True)
        self.dim_reduction = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.LeakyReLU(),
            Rearrange('b n d -> b (n d)'),
        )
        readout_dim = num_roi//2 * 8
        self.fc = nn.Sequential(
            nn.Linear(readout_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        

    def forward(self, x:Tensor, *args, **kwargs):
        B, T, N, _ = x.shape

        x = rearrange(x, 'b t n m -> (b t) n m')

        # encoder
        for blk in self.transformer_block:
            x, attn_weights = blk(x)

        # readout
        x, self.assignment = self.pooling(x)
        x = self.dim_reduction(x)
        x = rearrange(x, '(b t) d -> b t d', b=B)

        # post fc
        x = self.fc(x)
        return x

if __name__ == '__main__':
    B = 2
    T = 2
    N = 3
    D = 4
    depth = 2
    x = torch.randn(B, T, N, N)
    model = Mlp(N, D, depth=depth)

    y = model(x)

    print(y.shape)
import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import reduce, rearrange

class ClassifyHead(nn.Module):
    def __init__(self, embed_dim, num_class=2,
                 act=nn.GELU, dropout=0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.act = act()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_class)

    def forward(self, x:Tensor):
        x = self.dropout(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head(x)
        return x

class ClassifyHeadPlus(nn.Module):
    def __init__(self, embed_dim, num_class=2,
                 dropout=0.) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_class)
        )
    def forward(self, x:Tensor):
        x = self.head(x)
        return x


class ConcatReadoutHead(nn.Module):
    """
    input_shape == (num_sample, num_node, embed_dim)
    output_shape == (num_sample, num_class)
    First, it takes dimension reduction.
    Second, it flatten [num_node] axis to shape (num_sample, num_node * c)
    Third, following two linear layers.
    """
    def __init__(self, embed_dim, num_node, num_class=2,
                 dropout=0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 8),
            nn.GELU(),
            Rearrange('b r c -> b (r c)'),
            nn.Linear(8 * num_node, 256),
            nn.GELU(),
            nn.Linear(256, 32),
            nn.GELU(),
            nn.Linear(32, num_class)
        )

    def forward(self, x:Tensor):
        # x.shape == (num_sample, num_roi, embed_dim)
        # note: It's without [CLS] token.
        x = self.dropout(x)
        x = self.norm(x)
        x = self.head(x)
        return x

class MeanReadoutHead(nn.Module):
    """
    input_shape == (num_sample, num_node, embed_dim)
    output_shape == (num_sample, num_class)
    It will mean along with [num_node] axis
    """
    def __init__(self, embed_dim, num_class=2,
                 dropout=0.) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_class)
        )

    def forward(self, x:Tensor):
        # x.shape == (num_sample, num_node, embed_dim)
        # note: It's without [CLS] token.
        x = reduce(x, 'b t n -> b n', 'mean')
        x = self.head(x)
        return x

class SumReadoutHead(nn.Module):
    """
    input_shape == (num_sample, num_node, embed_dim)
    output_shape == (num_sample, num_class)
    It will sum along with [num_node] axis
    """
    def __init__(self, embed_dim, num_class=2,
                 dropout=0.) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_class)
        )
    def forward(self, x:Tensor):
        # x.shape == (num_sample, num_node, embed_dim)
        # note: It's without [CLS] token.
        x = reduce(x, 'b t n -> b n', 'sum')
        x = self.head(x)
        return x
    
class ReconstructHead(nn.Module):
    def __init__(self, embed_dim, target_dim, 
                 act=nn.GELU, dropout=0.) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.act = act()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, target_dim)

    def forward(self, x:Tensor):
        x = self.dropout(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head(x)
        return x
    



    
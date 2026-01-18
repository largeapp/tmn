import torch
from torch import nn, Tensor
from einops import einsum

"""
reference: https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
"""
class ABP(nn.Module):
    """attention-based pooling
    
    input: (B, K, M)
    output: (B, M*attn_branches), attn_branches is usually equal to 1.
    """
    def __init__(self, in_channels, 
                 hidden_channels=128,
                 attn_branches=1) -> None:
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, attn_branches)
        )
    
    def forward(self, x:Tensor):
        B, K, M = x.shape

        A = self.attn(x)    # (B, K, attn_branches)
        A = torch.transpose(A, 1, 2)    # (B, attn_branches, K)
        A = A.softmax(-1)

        Z = einsum(A, x, 'b h k, b k m -> b h m')  # (B, attn_branches, M)

        return Z.view(B, -1)
    

if __name__ == '__main__':
    x = torch.randn((2, 3, 4))
    abp = ABP(in_channels=4)

    y = abp(x)

    print(y.shape)
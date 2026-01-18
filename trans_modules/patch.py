import torch
from torch import nn, Tensor
from einops import rearrange

class PatchEmbed1D(nn.Module):
    """1D Sequence to Patch Embedding
    """
    def __init__(self, num_frames, num_vars, embed_dim=None, 
                 patch_len=1, patch_stride=1,
                 padding=0, norm_layer=None) -> None:
        super().__init__()
        self.embed_dim = embed_dim or num_vars * patch_len
        self.num_patches = (num_frames - patch_len + 2*padding) // patch_stride + 1

        self.proj = nn.Conv1d(num_vars, self.embed_dim, 
                              kernel_size=patch_len, stride=patch_stride,
                              padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, T, N = x.shape
        x = rearrange(x, 'b t n -> b n t')
        x = self.proj(x)        # (B, N, T) -> (B, embed_dim, num_patch)
        x = rearrange(x, 'b c p -> b p c')
        x = self.norm(x)
        return x

class PatchEmbed2D(nn.Module):
    """ 2D Sequence to Patch Embedding
    reference from timm.models.vision_transformer.PatchEmbed
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py#L25
    """
    def __init__(self, num_frames, num_vars, embed_dim=None,
                 patch_len=1, patch_stride=1,
                 padding=0, norm_layer=None) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.embed_dim = embed_dim or num_vars * patch_len
        self.num_patches = (num_frames - patch_len + 2*padding) // patch_stride + 1

        # kernel_size (height, width), the same order as input
        self.proj = nn.Conv2d(1, self.embed_dim, kernel_size=(patch_len, num_vars),
                              stride=patch_stride,
                              padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x):
        B, T, N = x.shape
        assert N == self.num_vars, f"Input sequence variables doesn't match model ({self.num_vars})."
        x = rearrange(x, 'b t n -> b () t n')
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)        # BCTN -> BPC
        x = self.norm(x)
        return x

class Patch(nn.Module):
    def __init__(self, patch_len, stride):
        """
        perform patching on the batch input data
        Args:
            patch_len:        patch length
            stride:           stride
        """
        self.patch_len = patch_len
        self.stride = stride
       
    def forward(self, x:Tensor):
        """
        convert to patch: [bs x seq_len x n_vars] -> [bs x num_patch x n_vars x patch_len]
        """
        xb_patch, num_patch = create_patch(x, self.patch_len, self.stride)    # x: [bs x seq_len x n_vars]
        return xb_patch

def create_patch(xb:Tensor, patch_len, stride):
    """
    in:         [bs x seq_len x n_vars]
    out:        [bs x num_patch x n_vars x patch_len]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len)-patch_len) // stride + 1
    tgt_len = patch_len  + stride*(num_patch-1)
    s_begin = seq_len - tgt_len
        
    xb = xb[:, s_begin:, :]                                                    # xb: [bs x tgt_len x nvars]
    xb = xb.unfold(dimension=1, size=patch_len, step=stride)                 # xb: [bs x num_patch x n_vars x patch_len]
    return xb

def patch_with_pad(x:Tensor, real_seq_len:Tensor, patch_len, patch_stride):
    """
    parameters:
        x:                  (B, T, N)
        real_seq_len:       (B, )
    return:
        x:                  (B, num_patch, N, patch_len)
        pad_mask:           (B, num_patch), True means paded.
    """
    B, T, N = x.shape
    assert T >= patch_len
    device = x.device
    num_patch = (T - patch_len) // patch_stride + 1
    # discard no use len
    tgt_len = patch_len  + patch_stride*(num_patch-1)
    s_begin = T - tgt_len

    if real_seq_len is not None:
        num_real_patch = (real_seq_len - patch_len) // patch_stride + 1
        pad_mask = torch.arange(num_patch, device=device)[None, :] >= num_real_patch[:, None]
    else:
        pad_mask = None
    
    x = x[:, s_begin:, :]
    x = x.unfold(dimension=1, size=patch_len, step=patch_stride)
    return x, pad_mask



if __name__ == '__main__':
    B, T, N = 2, 4, 5
    x = torch.randn(B, T, N)
    patch_embed = PatchEmbed2D(num_frames=T, num_vars=N, embed_dim=7, patch_len=1, patch_stride=1)

    y = patch_embed(x)
    print(y.shape)
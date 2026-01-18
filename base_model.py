from torch import nn, Tensor
import torch
from model.trans_modules.patch import patch_with_pad
from model.loss_modules.focal_loss import FocalLoss
import numpy as np

class TransBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}
    
    def padding_mask(self, lengths:Tensor, max_len=None):
        """
        Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
        where 1 means keep element at this position (time step)
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()  # trick works because of overloading of 'or' operator for non-boolean types
        return (torch.arange(0, max_len, device=lengths.device)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))
    
    def patchify(self, x, real_seq_len, patch_len=8, patch_stride=4):
        """
        x: (B, T, N) -> (B, num_patch, N, patch_len)
        pad_mask: (B, num_patch), True means paded.
        """
        x, pad_mask = patch_with_pad(x, real_seq_len, patch_len, patch_stride)
        return x, pad_mask

class ClassifyBaseModel(nn.Module):
    def __init__(self, num_class=2, loss_fn_type='ce') -> None:
        super().__init__()
        self.num_class = num_class
        self.loss_fn_type = loss_fn_type
        if loss_fn_type == 'ce':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn_type == 'focal':
            self.loss_fn = FocalLoss(gamma=2, alpha=0.3)
        
        # print(f"num_class: {num_class}, loss type: {loss_fn_type}")
        
    def get_loss(self, logit, label):
        loss = self.loss_fn(logit, label)
        return loss
    
    def fmri2dfc(self, fmri:Tensor, window_size, window_step):
        """
        parameters:
            fmri:       (B, T, N)
        returns:
            dfc:        (B, K, N, N), K is the intervals(number of windows)
        """
        window_size = hasattr(self, 'window_size') and self.window_size or window_size
        window_step = hasattr(self, 'window_step') and self.window_step or window_step

        B, T, N = fmri.shape
        fmri_unfold = fmri.unfold(dimension=1, size=window_size, step=window_step)    # (B, num_intervals, N, window_size)
        fmri_mean = torch.mean(fmri_unfold, dim=-1, keepdim=True)
        fmri_std = torch.std(fmri_unfold, dim=-1, unbiased=True, keepdim=True)         # do not use bessel's Correction
        fmri_norm = (fmri_unfold - fmri_mean) / fmri_std

        fc = torch.einsum('bknt, bkmt -> bknm', fmri_norm, fmri_norm)
        fc /= window_size-1
        fc = torch.clip(fc, -1, 1)

        # norm fc
        ## 1. nan
        fc[torch.isnan(fc)] = 0.
        ## 2. zeros diagnoal
        diag_indices = np.diag_indices(fc.shape[-1])
        fc[:, :, diag_indices[0], diag_indices[1]] = 0.
        ## 3. fisherz
        # eps = 1e-16
        # fc[fc>=1] = 1 - eps
        # fc[fc<=-1] = -1 + eps
        # fc = 0.5 * torch.log((1+fc) / (1-fc))
        return fc
    
    def fmri2spd(self, fmri:Tensor, window_size, window_step):
        """
        parameters:
            fmri:       (B, T, N)
        returns:
            cov:        (B, K, N, N), K is the intervals(number of windows)
        """
        window_size = hasattr(self, 'window_size') and self.window_size or window_size
        window_step = hasattr(self, 'window_step') and self.window_step or window_step

        B, T, N = fmri.shape
        fmri_unfold = fmri.unfold(dimension=1, size=window_size, step=window_step)    # (B, num_intervals, N, window_size)
        fmri_mean = torch.mean(fmri_unfold, dim=-1, keepdim=True)
        fmri_norm = fmri_unfold - fmri_mean

        cov = torch.einsum('bknt, bkmt -> bknm', fmri_norm, fmri_norm)
        cov /= window_size-1        # (B, K, N, N)

        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(B, -1, 1, 1)
        cov /= tra
        identity = torch.eye(N, device=fmri.device)
        cov = cov+(1e-5*identity)

        return cov

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())

        return n_params
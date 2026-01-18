import random
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy
from einops import repeat, rearrange
from einops.layers.torch import Reduce

import os
import sys
import pathlib
path = pathlib.Path(os.path.pardir)
sys.path.append(path.resolve().as_posix())

from model.base_model import ClassifyBaseModel
from model.trans_modules.heads import ClassifyHeadPlus

from model.mssn_modules.spatial_encoder import Mlp, BNT
from model.mssn_modules.temporal_encoder import TransformerEncoder
from model.mil.abp import ABP

from utils.randomized_quantization import RandomizedQuantizationAugModule

class MultiScaleSegmentNetwork(ClassifyBaseModel):
    def __init__(self, num_roi, region_num=8,
                 num_window_list = [1, 2, 4],
                 window_step = None,
                 spatial_dim=256, spatial_depth=3, 
                 temporal_dim=256, temporal_depth=2, pos_type='rope',
                 fusion_type='ms_concat_trans', fusion_depth=1,
                 alpha=1, beta=1, data_type='dfc',
                 num_class=2, loss_fn_type='ce') -> None:
        super().__init__(num_class, loss_fn_type)
        self.num_window_list = sorted(num_window_list)
        self.window_step = window_step
        self.num_roi = num_roi
        self.spatial_dim = spatial_dim
        self.spatial_depth = spatial_depth
        self.alpha = alpha
        self.beta = beta
        self.fusion_type = fusion_type
        self.data_type = data_type
        if region_num is not None:
            self.rand_quant_layer = RandomizedQuantizationAugModule(region_num=region_num)
        else:
            self.rand_quant_layer = nn.Identity()
        
        self.spatial_encoder = Mlp(num_roi=num_roi,
                                   hidden_features=spatial_dim,
                                   depth=spatial_depth)
        self.temporal_encoder = TransformerEncoder(
            d_model=temporal_dim,
            nhead=4, mlp_ratio=4, dropout=0.1, droppath=0.,
            init_values=None, activation=nn.GELU, norm=nn.LayerNorm,
            pos_type=pos_type, depth=temporal_depth
        )
        self.head1 = ClassifyHeadPlus(temporal_dim, num_class, dropout=0.1)

        if 'ms' in fusion_type:
            if 'concat' in fusion_type:
                fusion_dim = temporal_dim*len(self.num_window_list)
            elif 'add' in fusion_type:
                fusion_dim = temporal_dim
            else:
                raise NotImplementedError('unknown fusion type: {}'.format(fusion_type))
            if 'trans' in fusion_type:
                self.fusion_encoder = TransformerEncoder(
                    d_model=fusion_dim,
                    nhead=4, mlp_ratio=4, dropout=0.1, droppath=0.,
                    init_values=None, activation=nn.GELU, norm=nn.LayerNorm,
                    pos_type='none', depth=fusion_depth
                )
            elif 'abp' in fusion_type:
                self.fusion_encoder = ABP(in_channels=fusion_dim)
            elif 'mean' in fusion_type:
                self.fusion_encoder = nn.Sequential(
                    Reduce('b k c -> b c', 'mean'),
                )
            elif 'max' in fusion_type:
                self.fusion_encoder = nn.Sequential(
                    Reduce('b k c -> b c', 'max')
                )
        elif 'mean_concat' in fusion_type:
            fusion_dim = temporal_dim*len(self.num_window_list)
            self.fusion_encoder = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
            )
        elif fusion_type == 'mean_cls' or fusion_type == 'tier_1_only':
            assert alpha == 0, 'alpha == 0, that satisfies the tier_1_only'
            assert beta == 1
            fusion_dim = temporal_dim
            self.fusion_encoder = nn.Sequential(
                Reduce('b p c -> b c', 'mean'),
            )

        self.head2 = ClassifyHeadPlus(fusion_dim, num_class, dropout=0.1)

    def process_input(self, x: Tensor):
        B, T, N = x.shape
        pathway_x = []
        pathway_cls = []
        # remove some tails
        max_num_window = max(self.num_window_list)
        min_window_size = T // max_num_window
        T_clipped = min_window_size * max_num_window

        for num_window in self.num_window_list:
            assert max_num_window % num_window == 0
            window_size = T_clipped // num_window
            window_step = window_size if self.window_step is None else self.window_step
            if self.data_type == 'spd':
                p_x = self.fmri2spd(x[:, :window_size * num_window],
                                    window_size=window_size,
                                    window_step=window_step)  # (B, K, N, N)
            else:
                p_x = self.fmri2dfc(x[:, :window_size * num_window],
                                    window_size=window_size,
                                    window_step=window_step)  # (B, K, N, N)

            pathway_x.append(p_x)

        return tuple(pathway_x)  # 将 list 转换为 tuple

    def forward_mw(self, x1, x2, x3):
        pathway_x = []
        pathway_cls = []
        x = [x1, x2, x3]
        max_num_window = max(self.num_window_list)
        index = 0
        for num_window in self.num_window_list:
            p_x = x[index]
            index += 1
            # data augmentation
            # if 'aug' in kwargs and kwargs['aug'] == True:
            #     if isinstance(self.spatial_encoder, Mlp):
            #         p_x = rearrange(p_x, 'b k n m -> b (n m) k ()')
            #         p_x = self.rand_quant_layer(p_x)
            #         p_x = rearrange(p_x, 'b (n m) k () -> b k n m', n=N)
            #     else:
            #         p_x = rearrange(p_x, 'b k n c -> b c n k')
            #         p_x = self.rand_quant_layer(p_x)
            #         p_x = rearrange(p_x, 'b c n k -> b k n c')

            p_x = self.spatial_encoder(p_x)  # (B, K, C)

            if num_window != 1:
                p_x = self.temporal_encoder(p_x)  # (B, K+1, C) or (B, 1, C)
                cls_token, p_x = p_x[:, 0], p_x[:, 1:]  # (B, C)
            else:

                cls_token = p_x[:, 0]  # sFC: cls_token and x are the same

            if 'ms' in self.fusion_type:
                p_x = repeat(p_x, 'b k c -> b (k p) c', p=max_num_window // num_window)  # (B, max_K, C)

            pathway_x.append(p_x)
            pathway_cls.append(cls_token)
            # 处理pathway_x
        if 'ms_concat' in self.fusion_type:
            pathway_x = torch.cat(pathway_x, dim=-1)  # (B, max_K, P*C)
        elif 'ms_add' in self.fusion_type:
            pathway_x = torch.stack(pathway_x, dim=-1)  # (B, max_K, C, P)
            pathway_x = torch.sum(pathway_x, dim=-1)  # (B, max_K, C)
        elif 'mean_concat' in self.fusion_type:
            pathway_x = [torch.mean(p_x, dim=1) for p_x in pathway_x]  # (B, C)
            pathway_x = torch.cat(pathway_x, dim=-1)  # (B, P*C)
        pathway_cls = torch.stack(pathway_cls, dim=1)  # (B, P, C)

        ### layer1 - logit
        self.pathway_logit = self.head1(pathway_cls) \
            .view(-1, self.num_class)  # (B*P, num_class)

        ### layer2 - logit
        if len(self.num_window_list) != 1:
            if 'ms' in self.fusion_type:
                if 'trans' in self.fusion_type:
                    fusion_x = self.fusion_encoder(pathway_x)  # (B, max_K+1, C)
                    fusion_logit = self.head2(fusion_x[:, 0])  # (B, num_class)
                elif 'abp' in self.fusion_type or \
                        'mean' in self.fusion_type or \
                        'max' in self.fusion_type:
                    fusion_x = self.fusion_encoder(pathway_x)  # (B, C)
                    fusion_logit = self.head2(fusion_x)  # (B, num_class)
                return fusion_logit
            elif 'mean_concat' in self.fusion_type:
                fusion_x = self.fusion_encoder(pathway_x)  # (B, C)
                fusion_logit = self.head2(fusion_x)  # (B, num_class)
                return fusion_logit
            elif self.fusion_type == 'mean_cls' or self.fusion_type == 'tier_1_only':
                fusion_x = self.fusion_encoder(pathway_cls)  # (B, C)
                fusion_logit = self.head2(fusion_x)  # (B, num_class)
                return fusion_logit
        else:
            # only one pathway, so there is no layer2
            return self.pathway_logit.detach()

    def forward(self, x):
        # num_window alias as K
        # len(num_window_list) alias as P
        # max_num_window alias as max_K
        if isinstance(x, torch.Tensor):
            B, T, N = x.shape
            pathway_x = []
            pathway_cls= []
            # remove some tails
            max_num_window = max(self.num_window_list)
            min_window_size = T // max_num_window
            T_clipped = min_window_size * max_num_window

            for num_window in self.num_window_list:
                assert max_num_window % num_window == 0
                window_size = T_clipped // num_window
                window_step = window_size if self.window_step is None else self.window_step
                if self.data_type == 'spd':
                    p_x = self.fmri2spd(x[:, :window_size*num_window],
                                            window_size=window_size,
                                            window_step=window_step)  # (B, K, N, N)
                else:
                    p_x = self.fmri2dfc(x[:, :window_size*num_window],
                                            window_size=window_size,
                                            window_step=window_step)  # (B, K, N, N)
                # data augmentation
                # if 'aug' in kwargs and kwargs['aug'] == True:
                #     if isinstance(self.spatial_encoder, Mlp):
                #         p_x = rearrange(p_x, 'b k n m -> b (n m) k ()')
                #         p_x = self.rand_quant_layer(p_x)
                #         p_x = rearrange(p_x, 'b (n m) k () -> b k n m', n=N)
                #     else:
                #         p_x = rearrange(p_x, 'b k n c -> b c n k')
                #         p_x = self.rand_quant_layer(p_x)
                #         p_x = rearrange(p_x, 'b c n k -> b k n c')

                p_x = self.spatial_encoder(p_x)                   # (B, K, C)

                if num_window != 1:
                    p_x = self.temporal_encoder(p_x)              # (B, K+1, C) or (B, 1, C)
                    cls_token, p_x = p_x[:, 0], p_x[:, 1:]        # (B, C)
                else:

                    cls_token = p_x[:, 0]                         # sFC: cls_token and x are the same

                if 'ms' in self.fusion_type:
                    p_x = repeat(p_x, 'b k c -> b (k p) c', p=max_num_window//num_window)  # (B, max_K, C)

                pathway_x.append(p_x)
                pathway_cls.append(cls_token)
        elif isinstance(x, tuple):  # 如果 x 是元组
            pathway_x = []
            pathway_cls = []
            max_num_window = max(self.num_window_list)
            index = 0
            for num_window in self.num_window_list:
                p_x = x[index]
                index += 1
                # data augmentation
                # if 'aug' in kwargs and kwargs['aug'] == True:
                #     if isinstance(self.spatial_encoder, Mlp):
                #         p_x = rearrange(p_x, 'b k n m -> b (n m) k ()')
                #         p_x = self.rand_quant_layer(p_x)
                #         p_x = rearrange(p_x, 'b (n m) k () -> b k n m', n=N)
                #     else:
                #         p_x = rearrange(p_x, 'b k n c -> b c n k')
                #         p_x = self.rand_quant_layer(p_x)
                #         p_x = rearrange(p_x, 'b c n k -> b k n c')

                p_x = self.spatial_encoder(p_x)                   # (B, K, C)

                if num_window != 1:
                    p_x = self.temporal_encoder(p_x)              # (B, K+1, C) or (B, 1, C)
                    cls_token, p_x = p_x[:, 0], p_x[:, 1:]        # (B, C)
                else:

                    cls_token = p_x[:, 0]                         # sFC: cls_token and x are the same

                if 'ms' in self.fusion_type:
                    p_x = repeat(p_x, 'b k c -> b (k p) c', p=max_num_window//num_window)  # (B, max_K, C)

                pathway_x.append(p_x)
                pathway_cls.append(cls_token)

        # 处理pathway_x
        if 'ms_concat' in self.fusion_type:
            pathway_x = torch.cat(pathway_x, dim=-1)      # (B, max_K, P*C)
        elif 'ms_add' in self.fusion_type:
            pathway_x = torch.stack(pathway_x, dim=-1)     # (B, max_K, C, P)
            pathway_x = torch.sum(pathway_x, dim=-1)       # (B, max_K, C)
        elif 'mean_concat' in self.fusion_type:
            pathway_x = [torch.mean(p_x, dim=1) for p_x in pathway_x]   # (B, C)
            pathway_x = torch.cat(pathway_x, dim=-1)      # (B, P*C)
        pathway_cls = torch.stack(pathway_cls, dim=1)   # (B, P, C)
        
        ### layer1 - logit
        self.pathway_logit = self.head1(pathway_cls)\
            .view(-1, self.num_class)                   # (B*P, num_class)
        
        ### layer2 - logit
        if len(self.num_window_list) != 1:
            if 'ms' in self.fusion_type:
                if 'trans' in self.fusion_type:
                    fusion_x = self.fusion_encoder(pathway_x)     # (B, max_K+1, C)
                    fusion_logit = self.head2(fusion_x[:, 0])      # (B, num_class)
                elif 'abp' in self.fusion_type or \
                    'mean' in self.fusion_type or \
                    'max' in self.fusion_type:
                    fusion_x = self.fusion_encoder(pathway_x)     # (B, C)
                    fusion_logit = self.head2(fusion_x)      # (B, num_class)
                return fusion_logit
            elif 'mean_concat' in self.fusion_type:
                fusion_x = self.fusion_encoder(pathway_x)     # (B, C)
                fusion_logit = self.head2(fusion_x)      # (B, num_class)
                return fusion_logit
            elif self.fusion_type == 'mean_cls' or self.fusion_type == 'tier_1_only':
                fusion_x = self.fusion_encoder(pathway_cls)     # (B, C)
                fusion_logit = self.head2(fusion_x)      # (B, num_class)
                return fusion_logit
        else:
            # only one pathway, so there is no layer2
            return self.pathway_logit.detach()

    # def get_loss(self, logit, label):
    #     pathway_label = repeat(label, 'b -> (b p)', p=len(self.num_window_list))      # assign the same label to each pathway
        
    #     pathway_loss = super().get_loss(self.pathway_logit, pathway_label)
    #     fusion_loss = super().get_loss(logit, label)
    #     total_loss = fusion_loss + self.alpha * pathway_loss

    #     self.pathway_loss = pathway_loss
    #     self.fusion_loss = fusion_loss
    #     return total_loss
    def get_loss(self, logit, label):
        pathway_label = repeat(label, 'b -> (b p)', p=len(self.num_window_list))      # assign the same label to each pathway
        
        pathway_loss = super().get_loss(self.pathway_logit, pathway_label)
        fusion_loss = super().get_loss(logit, label)

        self.pathway_loss_ori = F.cross_entropy(self.pathway_logit, pathway_label, reduction='none')\
            .view(-1, len(self.num_window_list)).mean(0)

        self.pathway_loss = pathway_loss
        self.fusion_loss = fusion_loss

        total_loss = self.beta * fusion_loss + self.alpha * pathway_loss
        
        return total_loss


class MSSN_Trans(MultiScaleSegmentNetwork):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spatial_encoder = BNT(num_roi=self.num_roi,
                                   output_dim=self.spatial_dim,
                                   depth=self.spatial_depth)

    def forward(self, *inputs):
        if len(inputs) == 1:
            # 如果输入的参数个数为 1，则调用 forward2
            return super().forward(inputs[0])
        elif len(inputs) == 3:
            # 如果输入的参数个数为 3，则调用 forward1
            return super().forward_mw(inputs[0], inputs[1], inputs[2])
        else:
            raise ValueError("Expected 1 or 3 input tensors, but got {}".format(len(inputs)))


# if __name__ == '__main__':
#     B, T, N = 2, 230, 116
#     x = torch.randn(B, T, N)
#     label = torch.ones(B, dtype=torch.long)
#     model = MSSN_Trans(num_roi=N,
#                        num_window_list=[1, 2, 4],
#                        spatial_dim=64, temporal_dim=64,
#                        pos_type='rope',
#                        fusion_depth=1,
#                        alpha=1)
#
#     input = model.process_input(x)
#     model.eval()
#     # y = model(x)
#     y = model(input)
#     print(y.shape)
#     loss = model.get_loss(y, label)
#
#     print(loss)
#     print(model.pathway_loss)
#     print(model.fusion_loss)


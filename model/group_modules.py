"""
Group-specific modules
They handle features that also depends on the mask. 
Features are typically of shape
    batch_size * num_objects * num_channels * H * W

All of them are permutation equivariant w.r.t. to the num_objects dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_groups(g, ratio, mode, align_corners):
    batch_size, num_objects = g.shape[:2]
    g = F.interpolate(g.flatten(start_dim=0, end_dim=1), 
                scale_factor=ratio, mode=mode, align_corners=align_corners)
    g = g.view(batch_size, num_objects, *g.shape[1:])
    return g

def upsample_groups(g, ratio=2, mode='bilinear', align_corners=False):
    return interpolate_groups(g, ratio, mode, align_corners)

def downsample_groups(g, ratio=1/2, mode='area', align_corners=None):
    return interpolate_groups(g, ratio, mode, align_corners)


class GConv2D(nn.Conv2d):
    def forward(self, g):
        batch_size, num_objects = g.shape[:2]
        g = super().forward(g.flatten(start_dim=0, end_dim=1))
        return g.view(batch_size, num_objects, *g.shape[1:])


class GroupResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = GConv2D(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = GConv2D(out_dim, out_dim, kernel_size=3, padding=1)
 
    def forward(self, g):
        out_g = self.conv1(F.relu(g))
        out_g = self.conv2(F.relu(out_g))
        
        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g


class MainToGroupDistributor(nn.Module):
    def __init__(self, x_transform=None, method='cat', reverse_order=False):
        super().__init__()

        self.x_transform = x_transform
        self.method = method
        self.reverse_order = reverse_order

    def forward(self, x, g):
        num_objects = g.shape[1]

        if self.x_transform is not None:
            x = self.x_transform(x)
        # print(f"x: {x.shape} ") 
        # print(f"g: {g.shape} ") 
        if self.method == 'cat':
            if self.reverse_order:
                g = torch.cat([g, x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1)], 2)
            else:
                g = torch.cat([x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1), g], 2)
        elif self.method == 'add':
            g = x.unsqueeze(1).expand(-1,num_objects,-1,-1,-1) + g
        else:
            raise NotImplementedError

        return g

#     def forward(self, x, g):
#         # 1. 统一 g 维度为 5D: [B, num_objects, C, H, W]
#         #    方便后续扩展和拼接操作
#         if g.dim() == 3:
#             # g 形状 [C, H, W] -> 扩展 batch 和对象维度
#             g = g.unsqueeze(0).unsqueeze(0)  # 变成 [1, 1, C, H, W]
#         elif g.dim() == 4:
#             # g 形状 [B, C, H, W] -> 扩展对象维度
#             g = g.unsqueeze(1)  # 变成 [B, 1, C, H, W]
#         elif g.dim() != 5:
#             raise ValueError(f"[ERROR] Unsupported g dim: {g.dim()}")

#         # 2. 记录 num_objects 数量，用于后续扩展维度
#         num_objects = g.shape[1]

#         # 3. 如果有 x 的变换操作，则先执行变换（通常是卷积/线性层）
#         if self.x_transform is not None:
#             x = self.x_transform(x)
#             # 变换后 x 形状仍然是 [B, C, H, W]

#         # 4. 扩展 x 的对象维度，与 g 匹配形状以便拼接
#         #    x 形状: [B, C, H, W] -> unsqueeze 后 [B, 1, C, H, W]
#         #    expand 成 [B, num_objects, C, H, W]
#         x_expanded = x.unsqueeze(1).expand(-1, num_objects, -1, -1, -1)

#         # 5. 按照不同的方法拼接或相加 x_expanded 和 g
#         if self.method == 'cat':
#             # 拼接时通道维度 dim=2
#             if self.reverse_order:
#                 # g 在前，x 在后拼接
#                 g = torch.cat([g, x_expanded], dim=2)
#             else:
#                 # x 在前，g 在后拼接
#                 g = torch.cat([x_expanded, g], dim=2)
#         elif self.method == 'add':
#             # 通道数相同时，直接元素相加
#             g = x_expanded + g
#         else:
#             raise NotImplementedError

#         # 6. 返回拼接或相加后的张量，形状 [B, num_objects, C_new, H, W]
#         return g


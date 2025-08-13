# # import os
# # import glob
# # import torch
# # from torch.utils.data import Dataset
# # import numpy as np
# # import SimpleITK as sitk

# # class MRIMHADataset(Dataset):
# #     def __init__(self, root_dir, num_frames=5, transform=None):
# #         """
# #         root_dir: 包含多个样本子目录的根目录，每个子目录里有一个 .mha 文件作为序列
# #         num_frames: 每次训练用几个帧
# #         """
# #         self.root_dir = root_dir
# #         self.transform = transform
# #         self.num_frames = num_frames

# #         self.samples = sorted(glob.glob(os.path.join(root_dir, "*.mha")))
# #         print(f"[MRIMHADataset] Found {len(self.samples)} .mha sequences in {root_dir}")

# #     def __len__(self):
# #         return len(self.samples)

# #     def __getitem__(self, idx):
# #         mha_path = self.samples[idx]
# #         image = sitk.ReadImage(mha_path)
# #         array = sitk.GetArrayFromImage(image)  # shape: [T, H, W]
# #         print(f"[MRIMHADataset] Loaded {mha_path} with shape: {array.shape}")

# #         # 归一化为0-1
# #         array = array.astype(np.float32)
# #         array = (array - array.min()) / (array.max() - array.min() + 1e-6)

# #         # [T, H, W] -> [1, T, H, W] for single-channel
# #         array = torch.from_numpy(array).unsqueeze(0)

# #         # 只截取前 num_frames 帧（如果不够则循环填充）
# #         T = array.shape[1]
# #         if T >= self.num_frames:
# #             frames = array[:, :self.num_frames]
# #         else:
# #             repeat = self.num_frames // T + 1
# #             frames = array.repeat(1, repeat, 1, 1)[:, :self.num_frames]

# #         print(f"[MRIMHADataset] Final shape: {frames.shape}")  # [1, num_frames, H, W]

# #         # 为了兼容 XMem 的接口格式，转换为 list of dict
# #         sample = {
# #             'rgb': [frames[:, i] for i in range(self.num_frames)],  # 每帧 [1, H, W]
# #             'gt': [frames[:, 0]] + [torch.zeros_like(frames[:, 0])]*(self.num_frames-1),
# #             'info': {'name': os.path.basename(mha_path)},
            
# #         }

# #         return sample

# import os
# import glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import SimpleITK as sitk


# import os
# import glob
# import torch
# from torch.utils.data import Dataset
# import SimpleITK as sitk
# import numpy as np

# class MRIMHADataset(Dataset):
#     def __init__(self, root_dir, num_frames=5, transform=None):
#         """
#         root_dir: 包含多个样本子目录的根目录，每个子目录里有一个 .mha 文件作为序列
#         num_frames: 每次训练用几个帧
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.num_frames = num_frames

#         self.samples = sorted(glob.glob(os.path.join(root_dir, "*.mha")))
#         print(f"[MRIMHADataset] 找到 {len(self.samples)} 个.mha序列，路径: {root_dir}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         mha_path = self.samples[idx]
#         image = sitk.ReadImage(mha_path)
#         array = sitk.GetArrayFromImage(image)  # shape: [H, W, T] 根据之前打印是 (125,125,10)
#         print(f"[MRIMHADataset] 加载 {mha_path}，原始numpy数组shape: {array.shape} (H, W, T)")

#         # 转换为 float32 并归一化到 [0,1]
#         array = array.astype(np.float32)
#         array = (array - array.min()) / (array.max() - array.min() + 1e-6)
#         print(f"[MRIMHADataset] 归一化后，min: {array.min()}, max: {array.max()}")

#         # 交换轴，把时间维放到第0维，变成 [T, H, W]
#         array = np.transpose(array, (2, 0, 1))
#         print(f"[MRIMHADataset] 转置后数组shape (T, H, W): {array.shape}")

#         # 转为tensor，并补充通道维度C=1，变成 [T, C, H, W]
#         array = torch.from_numpy(array).unsqueeze(1)  # [T, 1, H, W]
#         print(f"[MRIMHADataset] 转为tensor并加通道维度后shape: {array.shape}")

#         T = array.shape[0]
#         # 如果帧数不足num_frames，则循环复制填充
#         if T >= self.num_frames:
#             frames = array[:self.num_frames]
#         else:
#             repeat = self.num_frames // T + 1
#             frames = array.repeat(repeat, 1, 1, 1)[:self.num_frames]
#         print(f"[MRIMHADataset] 最终选取帧数为 {self.num_frames}，frames shape: {frames.shape} (T, C, H, W)")

#         # 加入batch维度B=1，变成 [B, T, C, H, W]
#         frames = frames.unsqueeze(0)
#         print(f"[MRIMHADataset] 加batch维度后shape: {frames.shape} (B, T, C, H, W)")

#         sample = {
#             'rgb': frames,                     # [B, T, C, H, W]
#             'first_frame_gt': frames[0, 0, 0], # 取第一帧第一个通道的二维图像 [H, W]
#             'info': {
#                 'name': os.path.basename(mha_path),
#                 'num_objects': torch.tensor([1])
#             },
#             'selector': torch.tensor([0])
#         }

#         print(f"[MRIMHADataset] sample构造完成:")
#         print(f"  rgb shape: {sample['rgb'].shape} (B, T, C, H, W)")
#         print(f"  first_frame_gt shape: {sample['first_frame_gt'].shape} (H, W)")
#         print(f"  info: {sample['info']}")
#         print(f"  selector: {sample['selector']}")

#         return sample

    
# #     def __getitem__(self, idx):
# #         mha_path = self.samples[idx]
# #         image = sitk.ReadImage(mha_path)
# #         array = sitk.GetArrayFromImage(image)  # shape: [T, H, W]
# #         print(f"[MRIMHADataset] Loaded {mha_path} with shape: {array.shape}")

# #         # 归一化为0-1
# #         array = array.astype(np.float32)
# #         array = (array - array.min()) / (array.max() - array.min() + 1e-6)
# #         print(f"[DEBUG] After normalization, min: {array.min()}, max: {array.max()}")

# #         # [T, H, W] -> [1, T, H, W] for single-channel
# #         array = torch.from_numpy(array).unsqueeze(0)
# #         print(f"[DEBUG] After to tensor and unsqueeze(0), shape: {array.shape}, dtype: {array.dtype}")

# #         # 只截取前 num_frames 帧（如果不够则循环填充）
# #         T = array.shape[1]
# #         if T >= self.num_frames:
# #             frames = array[:, :self.num_frames]
# #             print(f"[DEBUG] Selected first {self.num_frames} frames, shape: {frames.shape}")
# #         else:
# #             repeat = self.num_frames // T + 1
# #             frames = array.repeat(1, repeat, 1, 1)[:, :self.num_frames]
# #             print(f"[DEBUG] Repeated frames to fill {self.num_frames}, shape: {frames.shape}")

# #         print(f"[MRIMHADataset] Final frames shape before stacking: {frames.shape}")

# #         # stack形成形状 [1, num_frames, H, W]
# #         rgb_stack = torch.stack([frames[:, i] for i in range(self.num_frames)], dim=1)
# #         rgb_stack = rgb_stack.squeeze(0)
# #         # rgb_stack = torch.stack([frames[:, i] for i in range(self.num_frames)], dim=1)  # [1, T, H, W]
# #         rgb_stack = rgb_stack.unsqueeze(1).expand(-1, 3, -1, -1)  # [1, T, 1, H, W]
# #         rgb_stack = rgb_stack.permute(0, 1, 3, 2)
# #         print(f"[DEBUG] After stacking frames, rgb_stack shape: {rgb_stack.shape}")

# #         sample = {
# #             'rgb': rgb_stack,                   # [1, T, H, W]
# #             'first_frame_gt': frames[:, 0].squeeze(0),  # [T, H, W] -> squeeze to [H, W]
# #             'info': {
# #                 'name': os.path.basename(mha_path),
# #                 'num_objects': torch.tensor([1])
# #             },
# #             'selector': torch.tensor([0])
# #         }

# #         print(f"[DEBUG] Sample keys: {list(sample.keys())}")
# #         print(f"[DEBUG] sample['rgb'] type: {type(sample['rgb'])}, shape: {sample['rgb'].shape}")
# #         print(f"[DEBUG] sample['first_frame_gt'] shape: {sample['first_frame_gt'].shape}")
# #         print(f"[DEBUG] sample['info']: {sample['info']}")
# #         print(f"[DEBUG] sample['selector']: {sample['selector']}")

# #         return sample









# import os
# import glob
# import torch
# from torch.utils.data import Dataset
# import SimpleITK as sitk
# import numpy as np
# import torch.nn.functional as F


# class MRIMHADataset(Dataset):
#     def __init__(self, root_dir, num_frames=5, transform=None):
#         """
#         root_dir: 包含多个样本子目录的根目录，每个子目录里有一个 .mha 文件作为序列
#         num_frames: 每次训练用几个帧
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.num_frames = num_frames

#         self.samples = sorted(glob.glob(os.path.join(root_dir, "*.mha")))
#         print(f"[MRIMHADataset] 找到 {len(self.samples)} 个.mha序列，路径: {root_dir}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         mha_path = self.samples[idx]
#         image = sitk.ReadImage(mha_path)
#         array = sitk.GetArrayFromImage(image)  # shape: [H, W, T] 根据之前打印是 (125,125,10)
#         print(f"[MRIMHADataset] 加载 {mha_path}，原始numpy数组shape: {array.shape} (H, W, T)")

#         # 转换为 float32 并归一化到 [0,1]
#         array = array.astype(np.float32)
#         array = (array - array.min()) / (array.max() - array.min() + 1e-6)
#         print(f"[MRIMHADataset] 归一化后，min: {array.min()}, max: {array.max()}")

#         # 交换轴，把时间维放到第0维，变成 [T, H, W]
#         array = np.transpose(array, (2, 0, 1))
#         print(f"[MRIMHADataset] 转置后数组shape (T, H, W): {array.shape}")

#         # 转为tensor，并补充通道维度C=1，变成 [T, C, H, W]
#         array = torch.from_numpy(array).unsqueeze(1)  # [T, 1, H, W]
#         print(f"[MRIMHADataset] 转为tensor并加通道维度后shape: {array.shape}")

#         T = array.shape[0]
#         # 如果帧数不足num_frames，则循环复制填充
#         if T >= self.num_frames:
#             frames = array[:self.num_frames]
#         else:
#             repeat = self.num_frames // T + 1
#             frames = array.repeat(repeat, 1, 1, 1)[:self.num_frames]
#         print(f"[MRIMHADataset] 最终选取帧数为 {self.num_frames}，frames shape: {frames.shape} (T, C, H, W)")

#         # Resize frames到 (128, 128)
#         frames = F.interpolate(frames, size=(128, 128), mode='bilinear', align_corners=False)
#         print(f"[MRIMHADataset] Resize后frames shape: {frames.shape}")
#         # 单通道复制成3通道,变成 [T, 3, H, W]
#         frames = frames.repeat(1, 3, 1, 1)  # (T, 3, H, W)
#         print(f"[MRIMHADataset] 单通道复制成3通道: {frames.shape} (T, C, H, W)")

#         sample = {
#             'rgb': frames,                     # [T, C, H, W]
#             'first_frame_gt': frames[0, 0].unsqueeze(0).unsqueeze(0), # [1, 1, H, W]
#             'info': {
#                 'name': os.path.basename(mha_path),
#                 'num_objects': torch.tensor([1])
#             },
#             'selector': torch.tensor([0]),
#             'cls_gt': frames[0, 0].unsqueeze(0).unsqueeze(0)
#         }

#         print(f"[MRIMHADataset] sample构造完成:")
#         print(f"  rgb shape: {sample['rgb'].shape} (B, T, C, H, W)")
#         print(f"  first_frame_gt shape: {sample['first_frame_gt'].shape} (H, W)")
#         print(f"  info: {sample['info']}")
#         print(f"  selector: {sample['selector']}")

#         return sample



    
    
    
# import os
# import torch
# from torch.utils.data import Dataset
# import SimpleITK as sitk
# import numpy as np
# import torch.nn.functional as F

# class MRIMHADataset(Dataset):
#     def __init__(self, root_dir, num_frames=3, max_jump=5, resize=(384,384)):
#         self.root_dir = root_dir
#         self.num_frames = num_frames
#         self.max_jump = max_jump
#         self.resize = resize

#         # 读取所有样本文件夹名，例如 Z_001, Z_002 ...
#         self.samples = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

#         print(f"[MRIMHAVOSDataset] 找到 {len(self.samples)} 个样本文件夹")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample_name = self.samples[idx]
#         images_path = os.path.join(self.root_dir, sample_name, "images", f"{sample_name}_frames.mha")
#         targets_path = os.path.join(self.root_dir, sample_name, "targets", f"{sample_name}_labels.mha")

#         # 读mha
#         img_itk = sitk.ReadImage(images_path)
#         tgt_itk = sitk.ReadImage(targets_path)

#         img_array = sitk.GetArrayFromImage(img_itk).astype(np.float32)
#         tgt_array = sitk.GetArrayFromImage(tgt_itk).astype(np.int64)

#         # 归一化图像到[0,1]
#         img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-6)

#         # 转置成 [T, H, W]
#         img_array = np.transpose(img_array, (2, 0, 1))
#         tgt_array = np.transpose(tgt_array, (2, 0, 1))

#         T, H, W = img_array.shape

#         # 随机采样num_frames帧，控制最大跳跃max_jump
#         trials = 0
#         while trials < 5:
#             frames_idx = [np.random.randint(T)]
#             acceptable_set = set(range(max(0, frames_idx[-1]-self.max_jump), min(T, frames_idx[-1]+self.max_jump+1))).difference(set(frames_idx))
#             while len(frames_idx) < self.num_frames:
#                 idx_f = np.random.choice(list(acceptable_set))
#                 frames_idx.append(idx_f)
#                 new_set = set(range(max(0, frames_idx[-1]-self.max_jump), min(T, frames_idx[-1]+self.max_jump+1)))
#                 acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))
#             frames_idx = sorted(frames_idx)
#             if np.random.rand() < 0.5:
#                 frames_idx = frames_idx[::-1]

#             # 检查第一帧标签是否包含目标
#             first_labels = np.unique(tgt_array[frames_idx[0]])
#             first_labels = first_labels[first_labels != 0]
#             if len(first_labels) == 0:
#                 trials += 1
#             else:
#                 break
#         else:
#             # 失败时直接用前num_frames帧
#             frames_idx = list(range(min(self.num_frames, T)))

#         imgs = img_array[frames_idx]
#         tgts = tgt_array[frames_idx]

#         imgs = torch.from_numpy(imgs).unsqueeze(1)  # [T, 1, H, W]
#         tgts = torch.from_numpy(tgts)

#         imgs = F.interpolate(imgs.float(), size=self.resize, mode='bilinear', align_corners=False)
#         tgts = tgts.unsqueeze(1).float()
#         tgts = F.interpolate(tgts, size=self.resize, mode='nearest')
#         tgts = tgts.squeeze(1).long()

#         labels = torch.unique(tgts[0])
#         labels = labels[labels != 0]
#         max_objs = 3
#         if len(labels) > max_objs:
#             labels = labels[:max_objs]
#         num_objects = len(labels)

#         cls_gt = torch.zeros((self.num_frames, self.resize[0], self.resize[1]), dtype=torch.int64)
#         # cls_gt = torch.zeros((frames.shape[0], frames.shape[2], frames.shape[3]), dtype=torch.long)
#         first_frame_gt = torch.zeros((1, num_objects, self.resize[0], self.resize[1]), dtype=torch.int64)

#         for i, l in enumerate(labels):
#             mask = (tgts == l)
#             cls_gt[mask] = i + 1
#             first_frame_gt[0, i] = mask[0]

#         selector = torch.zeros(max_objs, dtype=torch.float32)
#         selector[:num_objects] = 1.0

#         info = {
#             'name': sample_name,
#             'num_objects': num_objects,
#             'frames_idx': frames_idx,
#         }

#         return {
#             'rgb': imgs,
#             'first_frame_gt': first_frame_gt,
#             'cls_gt': cls_gt,
#             'selector': selector,
#             'info': info,
#         }



# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import torch.nn.functional as F
# import SimpleITK as sitk

# class MRIMHADataset(Dataset):
#     def __init__(self, root_dir, num_frames=3, max_jump=5, resize=(384,384)):
#         self.root_dir = root_dir
#         self.num_frames = num_frames
#         self.max_jump = max_jump
#         self.resize = resize

#         # 读取所有样本文件夹名，例如 Z_001, Z_002 ...
#         self.samples = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

#         print(f"[MRIMHAVOSDataset] 找到 {len(self.samples)} 个样本文件夹")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample_name = self.samples[idx]
#         images_path = os.path.join(self.root_dir, sample_name, "images", f"{sample_name}_frames.mha")
#         targets_path = os.path.join(self.root_dir, sample_name, "targets", f"{sample_name}_labels.mha")

#         # 读mha图像和标签
#         img_itk = sitk.ReadImage(images_path)
#         tgt_itk = sitk.ReadImage(targets_path)

#         img_array = sitk.GetArrayFromImage(img_itk).astype(np.float32)
#         tgt_array = sitk.GetArrayFromImage(tgt_itk).astype(np.int64)

#         # 归一化图像到[0,1]
#         img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-6)

#         # 转置成 [T, H, W] 格式
#         img_array = np.transpose(img_array, (2, 0, 1))
#         tgt_array = np.transpose(tgt_array, (2, 0, 1))

#         T, H, W = img_array.shape
#         print(f"[DEBUG] Loaded sample '{sample_name}' with shape img_array: {img_array.shape}, tgt_array: {tgt_array.shape}")

#         # 随机采样num_frames帧，控制最大跳跃max_jump
#         trials = 0
#         while trials < 5:
#             frames_idx = [np.random.randint(T)]
#             acceptable_set = set(range(max(0, frames_idx[-1]-self.max_jump), min(T, frames_idx[-1]+self.max_jump+1))).difference(set(frames_idx))
#             while len(frames_idx) < self.num_frames:
#                 idx_f = np.random.choice(list(acceptable_set))
#                 frames_idx.append(idx_f)
#                 new_set = set(range(max(0, frames_idx[-1]-self.max_jump), min(T, frames_idx[-1]+self.max_jump+1)))
#                 acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))
#             frames_idx = sorted(frames_idx)
#             if np.random.rand() < 0.5:
#                 frames_idx = frames_idx[::-1]

#             # 检查第一帧标签是否包含目标
#             first_labels = np.unique(tgt_array[frames_idx[0]])
#             first_labels = first_labels[first_labels != 0]
#             if len(first_labels) == 0:
#                 trials += 1
#             else:
#                 break
#         else:
#             # 失败时直接用前num_frames帧
#             frames_idx = list(range(min(self.num_frames, T)))

#         # 取对应帧
#         imgs = img_array[frames_idx]  # [T, H, W]
#         tgts = tgt_array[frames_idx]  # [T, H, W]

#         # 转为Tensor并增加通道维，变成 [T, 1, H, W]
#         imgs = torch.from_numpy(imgs).unsqueeze(1)
#         tgts = torch.from_numpy(tgts)

#         print(f"[DEBUG] Before resize imgs shape: {imgs.shape}, tgts shape: {tgts.shape}")

#         # 双线性插值调整尺寸（imgs）
#         imgs = F.interpolate(imgs.float(), size=self.resize, mode='bilinear', align_corners=False)

#         # 最近邻插值调整标签尺寸（tgts）
#         tgts = tgts.unsqueeze(1).float()
#         tgts = F.interpolate(tgts, size=self.resize, mode='nearest')
#         tgts = tgts.squeeze(1).long()

#         print(f"[DEBUG] After resize imgs shape: {imgs.shape}, tgts shape: {tgts.shape}")

#         # 复制通道，将单通道imgs扩展成3通道
#         imgs = imgs.repeat(1, 3, 1, 1)  # [T, 3, H, W]
#         print(f"[DEBUG] After channel repeat imgs shape: {imgs.shape}")

#         # 处理标签，提取第一帧的物体类别标签
#         labels = torch.unique(tgts[0])
#         labels = labels[labels != 0]
#         max_objs = 3
#         if len(labels) > max_objs:
#             labels = labels[:max_objs]
#         num_objects = len(labels)
#         print(f"[DEBUG] Number of objects in first frame: {num_objects}")

#         # 初始化cls_gt和first_frame_gt
#         cls_gt = torch.zeros((self.num_frames, self.resize[0], self.resize[1]), dtype=torch.int64)
#         first_frame_gt = torch.zeros((1, num_objects, self.resize[0], self.resize[1]), dtype=torch.int64)

#         # 生成目标掩码标签
#         for i, l in enumerate(labels):
#             mask = (tgts == l)
#             cls_gt[mask] = i + 1
#             first_frame_gt[0, i] = mask[0]

#         # 生成selector向量，激活存在目标的索引
#         selector = torch.zeros(max_objs, dtype=torch.float32)
#         selector[:num_objects] = 1.0

#         info = {
#             'name': sample_name,
#             'num_objects': num_objects,
#             'frames_idx': frames_idx,
#         }

#         return {
#             'rgb': imgs,                   # [T, 3, H, W]，3通道输入，满足模型需求
#             'first_frame_gt': first_frame_gt,
#             'cls_gt': cls_gt,
#             'selector': selector,
#             'info': info,
#         }



# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# import SimpleITK as sitk
# from torch.utils.data import Dataset

# class MRIMHADataset(Dataset):
#     def __init__(self, root_dir, num_frames=3, max_jump=5, resize=(384,384)):
#         self.root_dir = root_dir
#         self.num_frames = num_frames
#         self.max_jump = max_jump
#         self.resize = resize

#         # 读取所有样本文件夹名，例如 Z_001, Z_002 ...
#         self.samples = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

#         print(f"[MRIMHADataset] 找到 {len(self.samples)} 个样本文件夹")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample_name = self.samples[idx]
#         images_path = os.path.join(self.root_dir, sample_name, "images", f"{sample_name}_frames.mha")
#         targets_path = os.path.join(self.root_dir, sample_name, "targets", f"{sample_name}_labels.mha")

#         # 读取mha文件
#         img_itk = sitk.ReadImage(images_path)
#         tgt_itk = sitk.ReadImage(targets_path)

#         img_array = sitk.GetArrayFromImage(img_itk).astype(np.float32)
#         tgt_array = sitk.GetArrayFromImage(tgt_itk).astype(np.int64)

#         # 归一化图像到[0,1]
#         img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-6)

#         # 转置成 [T, H, W]，mha默认是 [Z, Y, X]
#         img_array = np.transpose(img_array, (2, 0, 1))
#         tgt_array = np.transpose(tgt_array, (2, 0, 1))

#         T, H, W = img_array.shape

#         print(f"[DEBUG] 载入样本 '{sample_name}': 原始帧数={T}, 高={H}, 宽={W}")

#         # 随机采样num_frames帧，控制最大跳跃max_jump
#         trials = 0
#         while trials < 5:
#             frames_idx = [np.random.randint(T)]
#             acceptable_set = set(range(max(0, frames_idx[-1]-self.max_jump), min(T, frames_idx[-1]+self.max_jump+1))).difference(set(frames_idx))
#             while len(frames_idx) < self.num_frames:
#                 idx_f = np.random.choice(list(acceptable_set))
#                 frames_idx.append(idx_f)
#                 new_set = set(range(max(0, frames_idx[-1]-self.max_jump), min(T, frames_idx[-1]+self.max_jump+1)))
#                 acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))
#             frames_idx = sorted(frames_idx)
#             if np.random.rand() < 0.5:
#                 frames_idx = frames_idx[::-1]

#             # 检查第一帧标签是否包含目标
#             first_labels = np.unique(tgt_array[frames_idx[0]])
#             first_labels = first_labels[first_labels != 0]
#             if len(first_labels) == 0:
#                 trials += 1
#             else:
#                 break
#         else:
#             # 失败时直接用前num_frames帧
#             frames_idx = list(range(min(self.num_frames, T)))

#         imgs = img_array[frames_idx]  # [T, H, W]
#         tgts = tgt_array[frames_idx]  # [T, H, W]

#         print(f"[DEBUG] 采样帧索引: {frames_idx}")

#         # 转为tensor，补充channel维，单通道扩展到3通道，resize
#         imgs = torch.from_numpy(imgs).unsqueeze(1)  # [T, 1, H, W]
#         tgts = torch.from_numpy(tgts)  # [T, H, W]

#         print(f"[DEBUG] resize前 imgs shape: {imgs.shape}, tgts shape: {tgts.shape}")

#         imgs = F.interpolate(imgs.float(), size=self.resize, mode='bilinear', align_corners=False)  # [T, 1, newH, newW]
#         tgts = tgts.unsqueeze(1).float()  # [T, 1, H, W]
#         tgts = F.interpolate(tgts, size=self.resize, mode='nearest')  # [T, 1, newH, newW]
#         tgts = tgts.squeeze(1).long()  # [T, newH, newW]

#         # 单通道扩展成3通道（模型期望3通道RGB）
#         imgs = imgs.repeat(1, 3, 1, 1)  # [T, 3, H, W]

#         print(f"[DEBUG] resize后 imgs shape: {imgs.shape}, tgts shape: {tgts.shape}")

#         labels = torch.unique(tgts[0])
#         labels = labels[labels != 0]
#         max_objs = 3
#         if len(labels) > max_objs:
#             labels = labels[:max_objs]
#         num_objects = len(labels)

#         print(f"[DEBUG] 第1帧目标数量: {num_objects}")

#         cls_gt = torch.zeros((self.num_frames, 1, self.resize[0], self.resize[1]), dtype=torch.int64)
#         first_frame_gt = torch.zeros((1, num_objects, self.resize[0], self.resize[1]), dtype=torch.int64)

#         for i, l in enumerate(labels):
#             mask = (tgts == l)
#             cls_gt[mask] = i + 1
#             first_frame_gt[0, i] = mask[0]

#         selector = torch.zeros(max_objs, dtype=torch.float32)
#         selector[:num_objects] = 1.0

#         info = {
#             'name': sample_name,
#             'num_objects': num_objects,
#             'frames_idx': frames_idx,
#         }
        
#         print(f"[MRIMHADataset] sample构造完成:")
#         print(f"  rgb shape: {imgs.shape}")
#         print(f"  cls_gt: {cls_gt.shape}")
#         print(f"  first_frame_gt shape: {first_frame_gt.shape}")
#         print(f"  info: {info}")
#         print(f"  selector: {selector}")
        
#         # 最终返回格式：
#         # imgs: [T, 3, H, W]  注意batch维由DataLoader自动添加
#         return {
#             'rgb': imgs,
#             'first_frame_gt': first_frame_gt,
#             'cls_gt': cls_gt,
#             'selector': selector,
#             'info': info,
#         }

import os 
import logging 
from typing import List, Dict, Tuple, Optional, Any 
import torch 
import torch.nn.functional as F 
import numpy as np 
import SimpleITK as sitk 
from torch.utils.data import Dataset 

# Configure logging 
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__) 

class MRIMHADataset(Dataset): 
 """ 
 MRI MHA格式数据集加载类，用于处理医学影像数据
 实现了从MHA文件读取、帧采样、数据预处理和格式转换功能
 """
 def __init__(self, root_dir: str, num_frames: int = 3, max_jump: int = 5, resize: Tuple[int, int] = (384, 384), max_objs: int = 3, debug: bool = False): 
     self.root_dir = root_dir  # 数据集根目录
     self.num_frames = num_frames  # 每个样本采样的帧数
     self.max_jump = max_jump  # 帧间最大跳跃间隔
     self.resize = resize  # 图像Resize目标尺寸 (高度, 宽度)
     self.max_objs = max_objs  # 最大目标数量
     self.debug = debug  # 是否启用调试模式

     # 验证输入参数
     self._validate_parameters()

     # 读取所有样本文件夹名，例如 Z_001, Z_002 ... 
     self.samples = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]) 

     logger.info(f"找到 {len(self.samples)} 个样本文件夹") 

 def __len__(self): 
     return len(self.samples) 

 def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
     """获取单个样本数据
     Args:
         idx: 样本索引
     Returns:
         包含图像数据、掩码和元信息的字典
     """ 
     sample_name = self.samples[idx] 
     images_path = os.path.join(self.root_dir, sample_name, "images", f"{sample_name}_frames.mha") 
     targets_path = os.path.join(self.root_dir, sample_name, "targets", f"{sample_name}_labels.mha") 

     # 验证文件存在性
     self._validate_file_exists(images_path, "图像")
     self._validate_file_exists(targets_path, "标签")

     # 读取mha文件 
     img_itk = sitk.ReadImage(images_path) 
     tgt_itk = sitk.ReadImage(targets_path) 

     # 将ITK图像转换为numpy数组 
     img_array = sitk.GetArrayFromImage(img_itk).astype(np.float32)  # 形状: [Z, Y, X] 
     tgt_array = sitk.GetArrayFromImage(tgt_itk).astype(np.int64)  # 形状: [Z, Y, X] 
    
#      print("++++++++")

#      print(img_array.shape)

#      print(tgt_array.shape)


     # 归一化图像到[0,1]范围 
     img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-6) 

     # 转置成 [T, H, W]，mha默认是 [Z, Y, X] 
     img_array = np.transpose(img_array, (2, 0, 1))  # 形状: [T, H, W] 
     tgt_array = np.transpose(tgt_array, (2, 0, 1))  # 形状: [T, H, W] 

     T, H, W = img_array.shape 

     if self.debug:
         logger.debug(f"载入样本 '{sample_name}': 原始帧数={T}, 高={H}, 宽={W}") 

     # 随机采样num_frames帧
     frames_idx = self._sample_frames(T, tgt_array) 

     imgs = img_array[frames_idx]  # [T, H, W] 
     tgts = tgt_array[frames_idx]  # [T, H, W] 

     if self.debug:
         logger.debug(f"采样帧索引: {frames_idx}") 

     # 已移至采样后输出 

     # 转为tensor，补充channel维，单通道扩展到3通道，resize 
     imgs = torch.from_numpy(imgs).unsqueeze(1)  # [T, 1, H, W] 
     tgts = torch.from_numpy(tgts)  # [T, H, W] 

     if self.debug:
         logger.debug(f"resize前 imgs shape: {imgs.shape}, tgts shape: {tgts.shape}") 

     # 图像双线性插值 resize
     imgs = F.interpolate(imgs.float(), size=self.resize, mode='bilinear', align_corners=False)  # [T, 1, newH, newW] 
     # 标签最近邻插值 resize
     tgts = tgts.unsqueeze(1).float()  # [T, 1, H, W] 
     tgts = F.interpolate(tgts, size=self.resize, mode='nearest')  # [T, 1, newH, newW] 
     tgts = tgts.squeeze(1).long()  # [T, newH, newW] 

     # 单通道扩展成3通道（模型期望3通道RGB输入） 
     imgs = imgs.repeat(1, 3, 1, 1)  # [T, 3, H, W] 

     if self.debug:
         logger.debug(f"resize后 imgs shape: {imgs.shape}, tgts shape: {tgts.shape}") 

     # 处理目标标签
     labels = torch.unique(tgts[0]) 
     labels = labels[labels != 0] 
     max_objs = 3  # 最大目标数量
     if len(labels) > max_objs: 
         labels = labels[:self.max_objs] 
     num_objects = len(labels) 

     if self.debug:
         logger.debug(f"第1帧目标数量: {num_objects}") 

     # 初始化输出标签数组
     cls_gt = torch.zeros((self.num_frames, 1, self.resize[0], self.resize[1]), dtype=torch.int64) 
     first_frame_gt = torch.zeros((1, num_objects, self.resize[0], self.resize[1]), dtype=torch.int64) 

     # 填充标签数据
     for i, l in enumerate(labels): 
         mask = (tgts == l).unsqueeze(1)  # 增加通道维度以匹配cls_gt形状 [T, 1, H, W]
         cls_gt[mask] = i + 1 
         first_frame_gt[0, i] = mask[0, 0]  # 提取单通道掩码 [H, W] 

     # 创建目标选择器
     selector = torch.zeros(max_objs, dtype=torch.float32) 
     selector[:num_objects] = 1.0 

     # 构建信息字典
     info = { 
         'name': sample_name, 
         'num_objects': num_objects, 
         'frames_idx': frames_idx, 
     } 


     # logger.info("sample构造完成:") 
     # logger.info(f"  rgb shape: {imgs.shape} [帧数, 通道数, 高度, 宽度]") 
     # logger.info(f"  cls_gt: {cls_gt.shape} [帧数, 1, 高度, 宽度]") 
     # logger.info(f"  first_frame_gt shape: {first_frame_gt.shape} [1, 目标数, 高度, 宽度]") 
     # logger.info(f"  info: {info}") 
     # logger.info(f"  selector: {selector}") 

#      print(imgs)
     # 最终返回格式： 
     # imgs: [T, 3, H, W]  注意batch维由DataLoader自动添加 
     return { 
         'rgb': imgs,  # 图像数据 [T, 3, H, W] 
         'first_frame_gt': first_frame_gt,  # 第一帧目标掩码 [1, 目标数, H, W] 
         'cls_gt': cls_gt,  # 所有帧目标分类 [T, 1, H, W] 
         'selector': selector,  # 目标选择器 [max_objs,] 
         'info': info,  # 样本信息字典 
     } 

 def _validate_parameters(self) -> None:
     """验证初始化参数的有效性"""
     if self.num_frames < 1:
         raise ValueError(f"无效的帧数: {self.num_frames}, 必须为正整数")
     if self.max_jump < 1:
         raise ValueError(f"无效的最大跳跃间隔: {self.max_jump}, 必须为正整数")
     if len(self.resize) != 2 or any(dim <= 0 for dim in self.resize):
         raise ValueError(f"无效的resize尺寸: {self.resize}, 必须为正整数元组")
     if not os.path.isdir(self.root_dir):
         raise NotADirectoryError(f"数据集根目录不存在: {self.root_dir}")

 def _validate_file_exists(self, file_path: str, file_type: str) -> None:
     """验证文件是否存在"""
     if not os.path.exists(file_path):
         raise FileNotFoundError(f"{file_type}文件不存在: {file_path}")
     if not os.path.isfile(file_path):
         raise IsADirectoryError(f"{file_type}路径不是文件: {file_path}")

 def _sample_frames(self, total_frames: int, target_array: np.ndarray) -> List[int]:
     """采样帧索引，确保第一帧包含目标
     Args:
         total_frames: 总帧数
         target_array: 目标标签数组
     Returns:
         采样的帧索引列表
     """
     # 最多尝试5次，确保第一帧包含目标
     trials = 0
     while trials < 5:
         frames_idx = [np.random.randint(total_frames)]  # 随机选择起始帧
         # 构建可接受帧集合，确保帧间跳跃不超过max_jump
         acceptable_set = set(range(max(0, frames_idx[-1]-self.max_jump),
                                   min(total_frames, frames_idx[-1]+self.max_jump+1)))
         acceptable_set.difference_update(set(frames_idx))

         while len(frames_idx) < self.num_frames and acceptable_set:
             idx_f = np.random.choice(list(acceptable_set))
             frames_idx.append(idx_f)
             new_set = set(range(max(0, frames_idx[-1]-self.max_jump),
                                min(total_frames, frames_idx[-1]+self.max_jump+1)))
             acceptable_set.update(new_set)
             acceptable_set.difference_update(set(frames_idx))

         # 如果无法收集足够帧数，补充剩余帧
         while len(frames_idx) < self.num_frames:
             frames_idx.append(frames_idx[-1])  # 使用最后一帧填充

         frames_idx = sorted(frames_idx)
         # 50%概率反转帧顺序
         if np.random.rand() < 0.5:
             frames_idx = frames_idx[::-1]

         # 检查第一帧标签是否包含目标
         first_labels = np.unique(target_array[frames_idx[0]])
         first_labels = first_labels[first_labels != 0]
         if len(first_labels) > 0:
             return frames_idx
         trials += 1

     # 所有尝试失败，使用前num_frames帧
     logger.warning(f"样本帧采样失败，使用前{self.num_frames}帧")
     return list(range(min(self.num_frames, total_frames)))
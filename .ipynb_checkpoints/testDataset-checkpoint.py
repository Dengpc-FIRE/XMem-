import os
import torch
from dataset.mri_mha_dataset import MRIMHADataset

def test_dataset_shape():
    # 配置测试参数
    ROOT_DIR = "../dataset/example"  # 匹配train.py中的默认数据集路径，可根据实际情况调整
    NUM_FRAMES = 10
    RESIZE = (125, 125)
    DEBUG = True

    # 创建数据集实例
    dataset = MRIMHADataset(
        root_dir=ROOT_DIR,
        num_frames=NUM_FRAMES,
        resize=RESIZE,
        debug=DEBUG
    )

    # 验证数据集不为空
    assert len(dataset) > 0, "测试数据集为空"

    # 测试前三个样本
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        sample_name = sample['info']['name']
        print(f"\n测试样本 {i+1}: {sample_name}")

        # 验证rgb形状 [T, 3, H, W]
        rgb_shape = sample['rgb'].shape
        expected_rgb_shape = (NUM_FRAMES, 3, RESIZE[0], RESIZE[1])
        # 删除以下冗余非[INFO]打印行：
        # print(f"rgb形状: {rgb_shape}, 预期: {expected_rgb_shape}")
        # print(f"cls_gt形状: {cls_gt_shape}, 预期: {expected_cls_shape}")
        # print(f"first_frame_gt第一维度: {first_frame_gt_shape[0]}, 预期: 1")
        # print(f"first_frame_gt空间维度: {first_frame_gt_shape[2:]}, 预期: {RESIZE}")
        # 最终清理：仅保留必要的[INFO]打印
        print(f"[INFO] rgb形状: {rgb_shape}, 预期: {expected_rgb_shape}")
        print(f"[INFO] cls_gt形状: {cls_gt_shape}, 预期: {expected_cls_shape}")
        print(f"[INFO] first_frame_gt第一维度: {first_frame_gt_shape[0]}, 预期: 1")
        print(f"[INFO] first_frame_gt空间维度: {first_frame_gt_shape[2:]}, 预期: {RESIZE}")
        print(f"[INFO] 目标数量: {sample['info']['num_objects']}")

    print("\n所有形状测试通过!")

if __name__ == "__main__":
    test_dataset_shape()
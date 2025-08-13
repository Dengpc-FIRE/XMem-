"""
TrackRAD2025 基线算法的入口脚本

该脚本负责：
1. 加载输入数据（MRI图像、帧率、磁场强度、扫描区域等）
2. 运行算法（在model.py中实现）
3. 保存输出数据（分割结果）

实际的算法实现在 model.py 文件中。

通常不需要修改此文件。
"""

# 导入必要的库
from pathlib import Path  # 用于处理文件路径
import json  # 用于读取JSON配置文件
from glob import glob  # 用于文件模式匹配
import SimpleITK  # 医学图像处理库
import numpy as np  # 数值计算库
import time  # 用于计时
import argparse  # 用于解析命令行参数
import sys  # 系统相关功能

def parse_args():
    """
    解析命令行参数
    
    返回:
        argparse.Namespace: 包含输入和输出路径的参数对象
    """
    parser = argparse.ArgumentParser(description='运行 TrackRAD2025 基线算法')
    parser.add_argument('--input', type=str, default='../dataset/example/Z_001', 
                       help='输入目录路径 (默认: /input)')
    parser.add_argument('--output', type=str, default='output_pred', 
                       help='输出目录路径 (默认: /output)')
    return parser.parse_args()

def run():
    """
    主运行函数
    
    执行流程：
    1. 解析命令行参数
    2. 加载输入数据
    3. 运行算法
    4. 保存输出结果
    
    返回:
        int: 0表示成功，1表示失败
    """
    # 解析命令行参数
    args = parse_args()
    
    # 使用命令行参数或默认值设置路径
    INPUT_PATH = Path(args.input)  # 输入数据目录
    OUTPUT_PATH = Path(args.output)  # 输出结果目录
    
    # # 检查输入目录是否存在
    # if not INPUT_PATH.exists():
    #     print(f"错误: 输入目录 '{INPUT_PATH}' 不存在!")
    #     print(f"当前路径下的可用目录:")
    #     for p in Path('.').iterdir():
    #         if p.is_dir():
    #             print(f"  - {p}")
    #     return 1
    
    # 开始计时 - 数据加载阶段
    loading_start_time = time.perf_counter()

    # 读取输入数据 - JSON配置文件
    # 1. 读取帧率信息
    input_frame_rate = load_json_file(
         location=INPUT_PATH / "frame-rate.json",
    )
    # 2. 读取磁场强度信息
    input_magnetic_field_strength = load_json_file(
         location=INPUT_PATH / "b-field-strength.json",
    )
    # 3. 读取扫描区域信息
    input_scanned_region = load_json_file(
         location=INPUT_PATH / "scanned-region.json",
    )

# 
    # 加载输入帧序列 - 尝试不同的可能位置
    input_mri_linac_series = None
    possible_frame_locations = [
        INPUT_PATH / "images/mri-linacs",  # 标准Docker路径
        INPUT_PATH / "images",             # 简化路径
        INPUT_PATH / "images/Z_001_frames.mha"  # 具体文件名
    ]
    
    # 遍历可能的帧序列位置，找到第一个存在的
    for loc in possible_frame_locations:
        if loc.exists():
            if loc.is_file():
                # 直接文件路径 - 加载单个文件
                input_mri_linac_series = load_single_image_file(loc)
                print(f"从文件加载帧序列: {loc}")
                break
            else:
                # 目录路径 - 从目录中加载第一个图像文件
                input_mri_linac_series = load_image_file_as_array(location=loc)
                print(f"从目录加载帧序列: {loc}")
                break
    
    # 检查是否成功加载帧序列
    if input_mri_linac_series is None:
        print("错误: 找不到输入帧序列!")
        print("期望的位置:")
        for loc in possible_frame_locations:
            print(f"  - {loc}")
        return 1
    

    
    # 加载目标标签 - 尝试不同的可能位置
    input_mri_linac_target = None
    possible_target_locations = [
        INPUT_PATH / "images/mri-linac-target",  # 标准Docker路径
        INPUT_PATH / "targets",                  # 简化路径
        # INPUT_PATH / "targets/Z_001_labels.mha", # 完整标签文件
        INPUT_PATH / "targets/Z_001_first_label.mha"  # 第一帧标签文件
    ]
    
    # 遍历可能的目标位置，找到第一个存在的
    for loc in possible_target_locations:
        if loc.exists():
            if loc.is_file():
                # 直接文件路径 - 加载单个文件
                input_mri_linac_target = load_single_image_file(loc)
                print(f"从文件加载目标: {loc}")
                break
            else:
                # 目录路径 - 从目录中加载第一个图像文件
                input_mri_linac_target = load_image_file_as_array(location=loc)
                print(f"从目录加载目标: {loc}")
                break
    
    # 检查是否成功加载目标
    if input_mri_linac_target is None:
        print("错误: 找不到目标标签!")
        print("期望的位置:")
        for loc in possible_target_locations:
            print(f"  - {loc}")
        return 1
    
    # ✅ 打印加载数据的维度和类型
    print(f"输入帧序列维度: {input_mri_linac_series.shape}, 类型: {input_mri_linac_series.dtype}")
    print(f"输入目标标签维度: {input_mri_linac_target.shape}, 类型: {input_mri_linac_target.dtype}")
    
    
    # 打印数据加载耗时
    print(f"数据加载耗时:   {time.perf_counter() - loading_start_time:.5f} 秒")

    # 导入算法模块
    from XMemModel import run_algorithm

    # 开始计时 - 算法运行阶段
    algo_start_time = time.perf_counter()
    
    print(f"[Debug]输入帧序列:{input_mri_linac_series.shape}")
    print(f"[Debug]目标标签:{input_mri_linac_target.shape}")

    # 运行算法，传入所有必要的参数
    output_mri_linac_series_targets = run_algorithm(
        frames=input_mri_linac_series,           # 输入帧序列
        target=input_mri_linac_target,           # 目标标签
        frame_rate=input_frame_rate,             # 帧率
        magnetic_field_strength=input_magnetic_field_strength,  # 磁场强度
        scanned_region=input_scanned_region      # 扫描区域
    )
    
    # 打印输出维度和类型
    print(f"输出预测标签维度: {output_mri_linac_series_targets.shape}, 类型: {output_mri_linac_series_targets.dtype}")

    # 强制转换为uint8数据类型（确保输出格式正确）
    output_mri_linac_series_targets = output_mri_linac_series_targets.astype(np.uint8)
    
    # 打印算法运行耗时
    print(f"算法运行耗时: {time.perf_counter() - algo_start_time:.5f} 秒")

    # 开始计时 - 结果保存阶段
    writing_start_time = time.perf_counter()

    # 保存输出结果
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/mri-linac-series-targets",  # 输出路径
        array=output_mri_linac_series_targets,                     # 输出数组
    )
    # 打印结果保存耗时
    print(f"结果保存耗时:   {time.perf_counter() - writing_start_time:.5f} 秒")
    
    return 0  # 返回成功状态


def load_json_file(*, location):
    """
    读取JSON文件
    
    参数:
        location: JSON文件路径
    
    返回:
        dict: JSON文件内容
    """
    with open(location, 'r') as f:
        return json.loads(f.read())


def load_single_image_file(location):
    """
    加载单个图像文件
    
    参数:
        location: 图像文件路径
    
    返回:
        numpy.ndarray: 图像数组
    """
    # 使用SimpleITK读取单个文件
    result = SimpleITK.ReadImage(str(location))
    # 转换为NumPy数组
    return SimpleITK.GetArrayFromImage(result)

def load_image_file_as_array(*, location):
    """
    从目录中加载图像文件
    
    参数:
        location: 包含图像文件的目录路径
    
    返回:
        numpy.ndarray: 图像数组
    """
    # 使用glob查找.tiff和.mha文件
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
    # 读取第一个找到的文件
    result = SimpleITK.ReadImage(input_files[0])
    # 转换为NumPy数组
    return SimpleITK.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    """
    将数组保存为图像文件
    
    参数:
        location: 输出目录路径
        array: 要保存的数组
    """
    # 创建输出目录（如果不存在）
    location.mkdir(parents=True, exist_ok=True)

    # 设置文件后缀为.mha
    suffix = ".mha"

    # 将NumPy数组转换为SimpleITK图像
    image = SimpleITK.GetImageFromArray(array)
    output_path = location / f"output{suffix}"
    
    try:
        # 首先尝试使用压缩
        SimpleITK.WriteImage(image, str(output_path), useCompression=True)
    except TypeError:
        # 如果不支持useCompression参数，则不使用压缩
        SimpleITK.WriteImage(image, str(output_path))
    
    print(f"输出已保存到: {output_path}")


def _show_torch_cuda_info():
    """
    显示PyTorch CUDA信息（调试用）
    """
    import torch

    print("=+=" * 10)
    print("收集PyTorch CUDA信息")
    print(f"PyTorch CUDA可用: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\t设备数量: {torch.cuda.device_count()}")
        print(f"\t当前设备: { (current_device := torch.cuda.current_device())}")
        print(f"\t设备属性: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


# 主程序入口
if __name__ == "__main__":
    raise SystemExit(run())  # 运行主函数并退出程序

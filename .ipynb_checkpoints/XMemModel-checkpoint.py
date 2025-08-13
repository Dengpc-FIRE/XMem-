"""
该文件实现了基于XMem模型的MRI序列分割推理算法

包含核心函数`run_algorithm`，接收MRI序列数据和初始目标掩码，
通过XMem模型为序列中的每一帧生成分割结果
"""
import torch  # 导入PyTorch深度学习框架
import numpy as np  # 导入NumPy数值计算库
import SimpleITK as sitk  # 导入医学图像处理库
from os import path  # 导入路径处理模块
from model.network import XMem  # 导入XMem网络模型
from inference.inference_core import InferenceCore  # 导入推理核心组件
from inference.data.mask_mapper import MaskMapper  # 导入掩码映射工具
from pathlib import Path  # 导入路径处理类

RESOURCE_PATH = Path("resources")  # 资源文件路径，用于加载权重和其他资源


def run_algorithm(frames: np.ndarray, target: np.ndarray, frame_rate: float, magnetic_field_strength: float, scanned_region: str) -> np.ndarray:
    """
    Run XMem model for MRI sequence segmentation using pre-trained weights

    Args:
    - frames (numpy.ndarray): Shape (W, H, T) containing the MRI linac series
    - target (numpy.ndarray): Shape (W, H) containing the initial target mask for first frame
    - frame_rate (float): The frame rate of the MRI linac series
    - magnetic_field_strength (float): The magnetic field strength of the MRI linac series
    - scanned_region (str): The scanned region of the MRI linac series

    Returns:
    - numpy.ndarray: Shape (W, H, T) with segmentation masks for each frame
    """
    # Configuration parameters from eval.py
    config = {
        'enable_long_term': True,               # 启用长期记忆机制
        'enable_long_term_count_usage': True,    # 启用长期记忆使用计数
        'max_mid_term_frames': 10,              # 中期记忆的最大帧数
        'min_mid_term_frames': 5,               # 中期记忆的最小帧数
        'max_long_term_elements': 10000,        # 长期记忆的最大元素数量
        'num_prototypes': 128,                  # 原型数量
        'top_k': 30,                            # Top-K相似性选择
        'mem_every': 5,                         # 每隔多少帧存储记忆
        'deep_update_every': -1,                # 深度更新频率(-1表示不更新)
        'model': 'save/Weight_150000.pth',            # 模型权重文件路径
        'benchmark': False                      # 是否启用基准测试模式
    }

    # 初始化XMem模型
    network = XMem(config, config['model']).eval()  # 创建XMem模型实例并设置为评估模式

    # 加载预训练权重
    if path.exists(config['model']):
        # 加载模型权重，自动选择设备(cuda或cpu)
        model_weights = torch.load(config['model'], map_location='cuda:0' if torch.cuda.is_available() else 'cpu', weights_only=True)
        # 加载权重到网络，如遇缺失层则初始化为零
        network.load_weights(model_weights, init_as_zero_if_needed=True)
    else:
        # 如果权重文件不存在则抛出错误
        raise FileNotFoundError(f"模型权重文件未找到: {config['model']}")

    # 将模型移动到适当的设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 检测并选择可用设备
    network.to(device)  # 将模型移动到选定设备

    # 初始化推理组件(来自eval.py)
    processor = InferenceCore(network, config=config)  # 创建推理核心处理器
    mapper = MaskMapper()  # 创建掩码映射器，用于处理标签映射

    # 获取序列维度信息
    W, H, T = frames.shape[0], frames.shape[1], frames.shape[2]  # 宽度、高度、时间帧数
    results = []  # 存储每帧的分割结果

    for ti in range(T):
        # 准备帧数据(与MRIMHADataset中相同的归一化处理)
        frame = frames[:, :, ti].astype(np.float32)  # 提取当前帧并转换为float32类型
        # print(f"[Scale Debug] Frame {ti} original shape: {frame.shape}")  # 打印原始帧尺度
        # 归一化到[0,1]范围，避免除零错误添加小epsilon
        frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
        # print(f"[Scale Debug] Frame {ti} after normalization: {frame.shape}")  # 打印归一化后尺度

        # 转换为XMem要求的输入张量形状[1, 1, H, W]
        # 从numpy数组转换为tensor，并添加批次和通道维度，然后移动到设备
        # 将单通道MRI图像转换为3通道以匹配模型输入要求
        frame_3ch = np.stack([frame, frame, frame], axis=0)  # 形状变为[3, H, W]
        # print(f"[Scale Debug] Frame {ti} after 3-channel conversion: {frame_3ch.shape}")  # 打印3通道转换后尺度
        rgb = torch.from_numpy(frame_3ch).unsqueeze(0).to(device)  # 形状变为[1, 3, H, W]
        # print(f"[Scale Debug1] Frame {ti} input tensor shape: {rgb.shape} (device: {rgb.device})")  # 打印输入张量尺度
        
        
    
        # 处理第一帧的初始掩码
        if ti == 0:
            if target is not None:
                # print(f"[Debug10001]:target:{target.shape}")
                # 将维度顺序从(0,1,2,3)调整为(0,3,1,2)
                target = target.transpose(2,0,1)
                #target 确保数组内存连续（可选，视后续操作需求）
                target = np.ascontiguousarray(target)
                # print(f"[Debug10002]:target:{target.shape}")

            # 处理初始目标掩码(转换为二值掩码)
            mask = (target > 0).astype(np.uint8)  # 确保掩码为二值(0或1)
            # print(f"[Debug1000]:mask:{mask.shape}")
            mask, labels = mapper.convert_mask(mask)  # 转换掩码格式
            # print(f"[Debug1000]:mask:{mask.shape}")

            msk = torch.Tensor(mask).to(device)  # 转换为tensor并移动到设备
            # 设置所有标签到处理器
            processor.set_all_labels(list(mapper.remappings.values()))
            # print(f"[Debug1000]:mask:{msk.shape}")

        else:
            msk = None  # 非第一帧不需要提供掩码
            labels = None  # 非第一帧不需要提供标签
            

            
        # print(f"[Debug1000]:mask:{msk.shape}")
        # 通过XMem核心运行推理步骤
        # 使用混合精度推理(如果未启用基准测试)
        # 使用新的PyTorch AMP语法以消除警告
        with torch.amp.autocast(device_type='cuda', enabled=not config['benchmark']):
            # 执行推理步骤，end参数指示是否为最后一帧
            # print(f"[Debug]:rgb1:{rgb.shape}")
            # print(f"[Debug]:mask1:{msk.shape}")
            rgb = rgb.squeeze(0)
            if msk is not None:
                msk = msk.squeeze(0)
            prob = processor.step(rgb, msk, labels, end=(ti == T - 1))
            # print(f"[Scale Debug2] Frame {ti} model output shape: {prob.shape}")  # 打印模型输出尺度

        # 将概率图转换为索引掩码
        # 取概率最大的类别作为输出，并转换为numpy数组
        out_mask = torch.max(prob, dim=0).indices.detach().cpu().numpy().astype(np.uint8)
        # print(f"[Scale Debug3] Frame {ti} output mask shape: {out_mask.shape}")  # 打印输出掩码尺度
        out_mask = mapper.remap_index_mask(out_mask)  # 重新映射索引掩码
        results.append(out_mask)  # 将当前帧结果添加到列表

    # 将结果堆叠为(W, H, T)形状的最终输出
    return np.stack(results, axis=-1)  # 沿着最后一个维度堆叠所有帧结果

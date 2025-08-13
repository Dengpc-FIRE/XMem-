import os
import glob

root_dir = "../dataset/example/Z_001/images"
print("Checking files in:", root_dir)
files = glob.glob(os.path.join(root_dir, "*.mha"))
print("Found .mha files:", files)


import SimpleITK as sitk

mha_path = '../dataset/example/Z_001/images/Z_001_frames.mha'
image = sitk.ReadImage(mha_path)

# 获取尺寸（Size）
size = image.GetSize()  # (width, height, depth) 对应 (X, Y, Z)

# 获取维度数
dimension = image.GetDimension()

print(f"Image size: {size}")
print(f"Image dimension: {dimension}")

# 如果你想查看图像数组的shape，转成numpy查看
array = sitk.GetArrayFromImage(image)
print(f"Numpy array shape: {array.shape}")

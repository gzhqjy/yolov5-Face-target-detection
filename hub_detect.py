import torch
import os
import numpy as np

# 确保模型路径存在
model_path = r'runs/train/exp10/weights/best.pt'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件未找到: {model_path}")

# 加载模型
model = torch.hub.load('.', 'custom', model_path, source='local')

# 指定图像路径
img_path = r'data/images/bus.jpg'
if not os.path.exists(img_path):
    raise FileNotFoundError(f"图像文件未找到: {img_path}")

# 进行推理
results = model(img_path, augment=False)

# 确保结果图像是可写的
for i in range(len(results.imgs)):
    results.imgs[i] = np.array(results.imgs[i]).copy()  # 创建可写副本

# 打印和显示结果
results.print()
results.show()

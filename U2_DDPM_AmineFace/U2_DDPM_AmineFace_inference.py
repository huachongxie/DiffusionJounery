import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler
import os

# 定义训练设备
device = "cuda:8"
pth = 80

# 加载训练好的模型
model = UNet2DModel(
    sample_size=64,  # 目标图像分辨率
    in_channels=3,   # 输入通道数，RGB 图像为 3
    out_channels=3,  # 输出通道数，RGB 图像为 3
    layers_per_block=6,  # 每个 UNet 块的 ResNet 层数
    block_out_channels=(128, 256, 512),  # 基本 UNet 的通道数
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
).to(device)

state_dict = torch.load(f"U2_DDPM_AminePTH/DDPM_Amine_epoch_{pth}.pth", weights_only=True, map_location=device)
model.load_state_dict(state_dict)
model.eval()  # 设置为评估模式

# DDPM调度器
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 定义生成函数
def generate_image(model, scheduler, device, num_steps=1000, sample_size=64):
    # 从纯噪声开始生成
    noise = torch.randn(1, 3, sample_size, sample_size).to(device)
    timesteps = torch.arange(num_steps-1, -1, -1, device=device)  # 反向时间步，从最后的噪声开始

    # 逐步去噪
    with torch.no_grad():
        for t in timesteps:
            # 在当前时间步进行预测噪声
            predicted_noise = model(noise, timestep=t).sample

            # 使用 DDPM 调度器将噪声去除
            noise = scheduler.step(predicted_noise, t, noise).prev_sample

    # 归一化并转换为图片
    image = (noise / 2 + 0.5).clamp(0, 1)  # 反归一化到 [0, 1]
    return image.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 转换为 numpy 数组，方便绘图

# 生成一张图片
generated_image = generate_image(model, scheduler, device)

# 可选择保存图片
output_path = f"generated_image_{pth}.png"
Image.fromarray((generated_image * 255).astype('uint8')).save(output_path)
print(f"Generated image saved at {output_path}")
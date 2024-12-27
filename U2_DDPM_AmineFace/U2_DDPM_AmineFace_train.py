import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDPMPipeline, UNet2DModel, DDPMScheduler
from torch import nn, optim
from tqdm.auto import tqdm
import os
from PIL import Image

# 定义训练设备
device = "cuda:6"
stpth = 80
epochs = 201    # 设置训练周期数

# 加载数据集（假设数据集为AmineFace）
class ImageDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.img_names = [f"{i}.png" for i in range(1, 21552)]  # 图片文件命名规则：1.png, 2.png, ..., 21551.png

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        img = Image.open(img_path).convert("RGB")  # 打开图片，并转换为RGB模式

        if self.transform:
            img = self.transform(img)

        return img, img_name  # 返回图像数据和文件名作为标签（如果需要标签，可以根据需求进行调整）

# 数据预处理：转换为Tensor和归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化
])

dataset = ImageDataset(img_folder="AmineFace", transform=transform)
train_dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)

# 定义模型和调度器
model = UNet2DModel(
    sample_size=64,           # the target image resolution
    in_channels=3,            # the number of input channels, 3 for RGB images
    out_channels=3,           # the number of output channels
    layers_per_block=6,       # how many ResNet layers to use per UNet block
    block_out_channels=(128, 256, 512),  # Roughly matching our basic unet example
    down_block_types=( 
        "DownBlock2D",        # a regular ResNet downsampling block
        "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ), 
    up_block_types=(
        "AttnUpBlock2D", 
        "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",          # a regular ResNet upsampling block
      ),
)

state_dict = torch.load(f"U2_DDPM_AminePTH/DDPM_Amine_epoch_{stpth}.pth", weights_only=True)
model.load_state_dict(state_dict)
model.train()  # 设置为评估模式
model = model.to(device)

print(f"模型参数量：{sum([p.numel() for p in model.parameters()]):2e}")

# DDPM调度器
scheduler = DDPMScheduler(num_train_timesteps=1000)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练过程
for epoch in range(stpth + 1, epochs):
    model.train()
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs}")
    
    for step, (images, img_names) in enumerate(loop): 
        images = images.to(device)

        # 添加噪声
        noise = torch.randn_like(images).to(device)
        
        # 获取当前的时间步（可以使用一个线性序列，或者固定的时间步）
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (images.size(0),), device=device)
        
        # 添加噪声
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        # 清除梯度
        optimizer.zero_grad()

        # 预测噪声
        predicted_noise = model(noisy_images, timestep=timesteps).sample

        # 计算损失
        loss = nn.MSELoss()(predicted_noise, noise)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f"U2_DDPM_AminePTH/DDPM_Amine_epoch_{epoch}.pth")

# 完成训练
print("Training completed!")

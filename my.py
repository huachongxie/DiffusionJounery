import torch
import time

# 选择CUDA设备
cuda_device = 8  # 可以根据需要修改为对应的CUDA编号

device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')

memory_gb = 30  # 目标显存大小为40GB
tensor_size = int(memory_gb * 1024 * 1024 * 1024 / 4)  # 4字节表示float32类型数据的大小

tensor = torch.ones(tensor_size, dtype=torch.float32, device=device)

# 进行高强度计算任务：矩阵乘法、卷积等
def run_heavy_computation():
    while True:
        # 随机生成一个大矩阵
        A = torch.randn(10000, 10000, dtype=torch.float32, device=device)
        B = torch.randn(10000, 10000, dtype=torch.float32, device=device)
        
        # 执行矩阵乘法
        C = torch.matmul(A, B)
        
        # 进行一些额外的运算，确保GPU持续繁忙
        C = C * 1e-5 + torch.randn_like(C)

        # 你可以根据需求在此处增加更多计算任务，如卷积等
        time.sleep(0.1)  # 确保计算不被过度压制，避免过高CPU负载

# 启动计算任务
try:
    run_heavy_computation()
except KeyboardInterrupt:
    print("手动停止")
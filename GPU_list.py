import torch
import os

def get_device_with_max_free_memory():
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")
    
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_devices}")
    
    max_free_memory = 0
    best_device = None
    
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        free_memory, total_memory = torch.cuda.mem_get_info(i)
        free_memory_mb = free_memory / (1024 ** 3)  # 转换为 GB
        
        print(f"Device {i}:[{100*free_memory/total_memory:.2f}%] {props.name}, Free memory: {free_memory_mb:.2f} GB")
        
        if free_memory > max_free_memory:
            max_free_memory = free_memory
            best_device = i
    
    print(f"Selected Device: CUDA:{best_device} with {max_free_memory / (1024 ** 3):.2f} GB free memory")
    return torch.device(f"cuda:{best_device}")
    
if __name__ == "__main__":
    device = get_device_with_max_free_memory()
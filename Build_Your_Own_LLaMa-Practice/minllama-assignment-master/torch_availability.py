"""
检查 PyTorch 是否可用的工具模块
"""


def is_torch_available():
    """
    检查 PyTorch 是否可用
    
    Returns:
        bool: 如果 PyTorch 可用则返回 True，否则返回 False
    """
    try:
        import torch
        return True
    except ImportError:
        return False


def is_cuda_available():
    """
    检查 CUDA 是否可用
    
    Returns:
        bool: 如果 CUDA 可用则返回 True，否则返回 False
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_count():
    """
    获取可用 GPU 的数量
    
    Returns:
        int: 可用 GPU 的数量
    """
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def get_gpu_load(gpu_id=0):
    """
    获取指定 GPU 的负荷
    
    Args:
        gpu_id: GPU 设备 ID
    
    Returns:
        float: GPU 负荷（0.0-1.0），如果获取失败则返回 -1
    """
    try:
        import torch
        if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
            return -1
        
        # 使用 torch.cuda 提供的方法获取 GPU 内存使用情况
        # 注意：PyTorch 本身不直接提供 GPU 利用率的 API，这里返回内存使用比例作为参考
        device = torch.device(f"cuda:{gpu_id}")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        return allocated_memory / total_memory
    except ImportError:
        return -1
    except Exception:
        return -1


def get_gpu_temperature(gpu_id=0):
    """
    获取指定 GPU 的温度
    
    Args:
        gpu_id: GPU 设备 ID
    
    Returns:
        float: GPU 温度（摄氏度），如果获取失败则返回 -1
    """
    try:
        import torch
        if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
            return -1
        
        # 尝试使用 pynvml 库获取温度
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            return temperature
        except ImportError:
            # 如果 pynvml 不可用，返回 -1
            return -1
        except Exception:
            return -1
    except ImportError:
        return -1


if __name__ == "__main__":
    # 测试函数
    print(f"PyTorch 可用: {is_torch_available()}")
    
    if is_torch_available():
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {is_cuda_available()}")
        print(f"可用 GPU 数量: {get_gpu_count()}")
        
        gpu_count = get_gpu_count()
        if gpu_count > 0:
            for i in range(gpu_count):
                print(f"GPU {i} 负荷: {get_gpu_load(i):.2f}")
                temp = get_gpu_temperature(i)
                if temp >= 0:
                    print(f"GPU {i} 温度: {temp}°C")
                else:
                    print(f"GPU {i} 温度: 无法获取（可能需要安装 pynvml）")

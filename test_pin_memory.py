import torch
import psutil
import os

space = 12
num_layers = 28
num_kv_heads = 4
head_dim = 128
block_size = 128
tensors = [None] * num_layers
num_blocks = int(space * 1024 * 1024 * 1024 / (num_layers * block_size * num_kv_heads * head_dim * 2))

def get_current_memory_usage():
    """
    获取当前 Python 进程的内存使用量。
    返回单位为 MB。
    """
    process = psutil.Process(os.getpid())
    # memory_info() 返回一个命名 tuple，其中包含了多个内存相关指标
    # rss (Resident Set Size): 进程实际使用的物理内存（不包括 swap）
    # vms (Virtual Memory Size): 进程占用的虚拟内存大小
    # 请注意，rss 是更常用于衡量实际内存消耗的指标
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024) # 转换为 MB

if __name__ == "__main__":


    for i in range(num_layers):
        try:
            tensors[i] = torch.ones((num_blocks, block_size, num_kv_heads, head_dim), dtype=torch.bfloat16, device="cpu")
            print(f"Layer {i} allocation succeed, {tensors[i].shape}, "
            f"RSS: {get_current_memory_usage()} MB, ")
        except RuntimeError as e:
            print(f"Layer {i} allocation failed,: {e}")
            a = input("wait to exit")
            exit(-1)
            
    # 先分配后锁页，内存占用会小很多
    for i in range(num_layers):
        try:
            tensors[i].pin_memory()
            print(f"Layer {i} pin succeed, {tensors[i].shape}, "
            f"RSS: {get_current_memory_usage()} MB, ")
        except RuntimeError as e:
            print(f"Layer {i} pin failed: {e}")
            a = input("wait to exit")
            exit(-1)      

    a = input("wait to exit")
    
    
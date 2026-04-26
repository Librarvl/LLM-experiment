# test_nccl.py
import torch
import torch.distributed as dist
import os

def test_nccl():
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",       # GPU 分布式用 nccl
        init_method="env://",
    )
    
    rank       = dist.get_rank()           # 当前进程编号
    world_size = dist.get_world_size()     # 总进程数
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"进程 rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    # 创建一个 tensor 测试通信
    tensor = torch.ones(3).to(device) * rank
    print(f"rank {rank} 通信前: {tensor}")
    
    # AllReduce：所有 GPU 的 tensor 求和
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"rank {rank} AllReduce 后: {tensor}")
    # 期望结果：[0+1+2+...+n, ...]
    
    # Broadcast：rank0 的数据广播给所有 GPU
    tensor2 = torch.zeros(3).to(device)
    if rank == 0:
        tensor2 = torch.tensor([1.0, 2.0, 3.0]).to(device)
    dist.broadcast(tensor2, src=0)
    print(f"rank {rank} Broadcast 后: {tensor2}")
    # 期望结果：所有 GPU 都是 [1, 2, 3]
    
    dist.destroy_process_group()
    print(f"rank {rank} 通信测试通过 ✅")

if __name__ == "__main__":
    test_nccl()

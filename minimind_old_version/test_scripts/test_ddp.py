# test_ddp.py
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import os

def main():
    # 1. 初始化
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    print("1111", rank)
    local_rank = int(os.environ["LOCAL_RANK"])
    print("2222", rank)
    world_size = dist.get_world_size()
    print("3333", rank)
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 2. 创建简单模型
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)
    
    # 3. 包装成 DDP
    model = DDP(model, device_ids=[local_rank])
    
    # 4. 创建数据集
    dataset = TensorDataset(
        torch.randn(1000, 128),
        torch.randint(0, 10, (1000,)),
    )
    
    # DistributedSampler 保证每张 GPU 拿到不同的数据
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
    )
    
    # 5. 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.CrossEntropyLoss()
    
    for epoch in range(3):
        sampler.set_epoch(epoch)   # 每个 epoch 重新 shuffle
        total_loss = 0.0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            loss = loss_fn(pred, y)
            
            optimizer.zero_grad()
            loss.backward()       # DDP 自动同步梯度
            optimizer.step()
            
            total_loss += loss.item()
        
        # 只在 rank0 打印
        if rank == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    if rank == 0:
        print("✅ 分布式训练测试通过！")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

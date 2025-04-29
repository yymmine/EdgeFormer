# -*- coding: utf-8 -*-
import os
import torch
import torch.distributed as dist
import datetime

def recv_tensor(src, dtype=torch.float32):
    shape_tensor = torch.empty(3, dtype=torch.int64)
    print("tensor: ", shape_tensor)
    print("shape_tensor: ", shape_tensor.shape)
    dist.recv(tensor=shape_tensor, src=0)
    shape = tuple(shape_tensor.tolist())
    tensor = torch.empty(shape, dtype=dtype)
    dist.recv(tensor=tensor, src=src)
    return tensor

def receive_qk():
    os.environ['MASTER_ADDR'] = '192.168.1.101'  # 主节点的IP
    os.environ['MASTER_PORT'] = '29500'  # 端口号
    os.environ['RANK'] = '1'
    os.environ['WORLD_SIZE'] = '2'
    master_addr = "192.168.1.101"  # 主机 IP 地址
    master_port = "29500"  # 端口号
    print("正在接收")
    # 初始化进程组
    print(f"Initializing process group with master addr {master_addr} and port {master_port}")
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://192.168.1.101:29500",
        world_size=2,
        rank=1  # 对应接收端
    )
    # init_method=f'tcp://{master_addr}:{master_port}',

    while True:  # 确保程序一直运行，等待数据
        print("正在接收 q...")
        q = recv_tensor(src=0)

        print("正在接收 k...")
        k = recv_tensor(src=0)

        print("q.shape: ", q.shape)
        print("k.shape: ", k.shape)

        # 计算 q @ k^T
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [B, S, D] x [B, D, S] -> [B, S, S]

        print("计算完成，attn_scores.shape:", attn_scores.shape)
        shape_attn_scores = torch.tensor(attn_scores.shape, dtype=torch.int64)
        dist.send(tensor=shape_attn_scores, dst=0)
        dist.send(tensor=attn_scores, dst=0)


if __name__ == "__main__":
    receive_qk()
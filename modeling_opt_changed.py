# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OPT model."""
import datetime
from typing import List, Optional, Tuple, Union
import time
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import numpy as np
import json
import requests
import asyncio
import aiohttp
import orjson
import io
import os
import torch.distributed as dist
import speedtest
# from sklearn.metrics.pairwise import cosine_similarity
tag = False
if tag == True:
    master_addr="192.168.1.101"
    master_port="29500"
    world_size=2
    rank=0

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    dist.init_process_group(
        backend='gloo',
        init_method=f'tcp://{master_addr}:{master_port}',  # ← 主节点IP和端口
        rank=rank,                                   # ← 主节点是 rank 0
        world_size=world_size,                             # ← 总共两个进程（主+从）
        timeout=datetime.timedelta(seconds=1800)
    )
    print("初始化完成")


from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_opt import OPTConfig



if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

st = speedtest.Speedtest()
logger = logging.get_logger(__name__)
map_id = 0
dist_group = True
def load_table(cluster_file_path, updated_mapping_table_path):
    with open(cluster_file_path, "r") as f:
        data_list = json.load(f)

    cluster_centers = np.array(data_list)
        
    with open(updated_mapping_table_path, "r") as f:
        mapping_table = json.load(f)
    for key, value in mapping_table.items():
        value["hidden_state"] = np.array(value["hidden_state"])  # 转换为 numpy 数组
        value["cluster_center"] = np.array(value["cluster_center"])  # 转换为 numpy 数组
    return mapping_table, cluster_centers

table_file_path = '/media/yym/work/code/OPT-1.3B/mapping_table.json'
# table_file_path = '/media/yym/work/code/OPT-1.3B/gpt2_mapping_table.json'
cluster_file_path = "/media/yym/work/code/OPT-1.3B/cluster_centers.json"
updated_mapping_table_path = "/media/yym/work/code/OPT-1.3B/updated_mapping_table.json"
mapping_table, cluster_centers = load_table(cluster_file_path, updated_mapping_table_path)
_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/opt-350m-dummy-sc"
_SEQ_CLASS_EXPECTED_LOSS = 1.71
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_0'"

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def generate_spend_time(tra_rate):
    if os.environ["RANK"] == 1:
        local_time = 4 * 2 * 1 * 2048 * 2048 / (4 * 10e9)
    elif os.environ["RANK"] == 2:
        local_time = 4 * 2 * 1 * 2048 * 2048 / (2 * 10e9)
    elif os.environ["RANK"] == 3:
        local_time = 4 * 2 * 1 * 2048 * 2048 / (1 * 10e9)

    remote_time = 4 * 2 * 2 * 2048 * 2048 / tra_rate + local_time
    spend_time = np.min(local_time, remote_time)
    
    return spend_time

def sigmoid(x):
    return 1 /(1 + np.exp(-x))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def pca(X_tensor, num_components):
    # 1. 数据标准化（零均值）
    X = X_tensor.detach().numpy()
    X_meaned = X - np.mean(X, axis=0)
    
    # 2. 计算协方差矩阵
    cov_mat = np.cov(X_meaned, rowvar=False)
    
    # 3. 计算特征值和特征向量
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    
    # 4. 对特征值进行排序（从大到小）
    sorted_indices = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    
    # 5. 选择前 num_components 个特征向量
    principal_components = eigen_vectors[:, :num_components]
    
    # 6. 变换数据
    X_pca = np.dot(X_meaned, principal_components)
    
    return X_pca, principal_components


from flask import Flask, request, jsonify
import threading
import requests
import time

# # 定义一个 Flask 应用
# app = Flask(__name__)

# @app.route('/compute', methods=['POST'])
# def compute():
#     # 获取任务数据
#     data = request.json
#     q = data['q']
#     k = data['k']
#     v = data['v']

#     # 模拟计算（这里可以替换为实际的计算逻辑）
#     result = [[sum(q_i) for q_i in q], [sum(k_i) for k_i in k], [sum(v_i) for v_i in v]]

#     # 返回结果
#     return jsonify({"result": result})

# def run_flask_app():
#     # 启动 Flask 服务
#     app.run(port=5000)

def send_request_to_flask(ip_address, data):
    # 发送请求到 Flask 服务
    response = requests.post("http://127.0.0.1:5000/compute", json=data)
    return response.json()

# 定义填充函数
def pad_matrix(matrix, target_dim):
    """如果数据较短，则填充 0；如果较长，则截断"""
    current_dim = len(matrix)
    if current_dim <= target_dim:
        # print("matrix: ", matrix)
        padding = np.zeros((target_dim - current_dim))
        # print("padding: ", len(padding))
        # print("matrix_type: ", type(matrix))
        # print("padding_type: ", type(padding))
        return np.hstack((matrix, padding))
    else:
        return matrix[:target_dim]

def process_new_hidden_state(hidden_state, mapping_table, cluster_centers, num_head):
    """
    处理新的 hidden_state，根据映射表执行以下步骤：
    1. 找到离它最近的簇。
    2. 找到簇中与它最相似的 pca_hs。
    3. 保留贡献度最大的 k 个头的索引。
    4. 生成一个新的全为 0 的列表，长度为 num_head，将保留的索引位置置为 1。
    
    参数:
        hidden_state (np.ndarray): 新的 hidden_state，形状为 (hidden_dim,)。
        mapping_table (dict): 更新后的映射表。
        cluster_centers (np.ndarray): 所有簇的中心，形状为 (n_clusters, hidden_dim)。
        num_head (int): 头的总数。
        k (int): 保留的头的数量。
    
    返回:
        mask (list): 长度为 num_head 的列表，保留的索引位置为 1，其余为 0。
    """
    # 1. 找到离 hidden_state 最近的簇
    flattened_hidden_states = np.array(hidden_state).flatten()
    # print("flattened_hidden_states: ",len(flattened_hidden_states[0]))
    # TODO GPT2用399,OPT用373
    pca_hidden_state = pad_matrix(flattened_hidden_states, 373)
    distances_to_centers = np.linalg.norm(cluster_centers - pca_hidden_state, axis=1)  # 计算到每个簇中心的距离
    nearest_cluster_idx = np.argmin(distances_to_centers)  # 找到最近的簇索引
    # print(f"最近的簇索引: {nearest_cluster_idx}")

    # 2. 找到簇中与 hidden_state 最相似的 pca_hs
    cluster_pca_hs = [value["hidden_state"] for key, value in mapping_table.items() if value['cluster_label'] == nearest_cluster_idx]  # 获取簇中的所有 pca_hs
    cluster_pca_hs_contribution = [value["contribution"]for key, value in mapping_table.items() if value['cluster_label'] == nearest_cluster_idx]
    cluster_pca_hs_arrays = np.array(cluster_pca_hs)
    # cluster_pca_hs_arrays = [np.array(list(map(float, key.split(',')))) for key in cluster_pca_hs]  # 将 pca_hs 转换为 NumPy 数组
    # print("cluster_pca_hs_arrays: ", len(cluster_pca_hs_arrays))
    # 计算 hidden_state 与簇中所有 pca_hs 的余弦相似度
    similarities = [cosine_similarity(pca_hidden_state, pca_hs) for pca_hs in cluster_pca_hs_arrays]
    # print("similarities: ", similarities)
    most_similar_idx = np.argmax(similarities)  # 找到最相似的 pca_hs 索引
    # most_similar_pca_hs = cluster_pca_hs[most_similar_idx]  # 最相似的 pca_hs
    # print(f"最相似的 pca_hs索引: {most_similar_idx}")

    # 3. 保留贡献度最大的 k 个头的索引
    contribution = cluster_pca_hs_contribution[most_similar_idx]  # 获取贡献度
    tra_rate = st.upload()
    spend_time = generate_spend_time(tra_rate)
    S_contribution = 0.5 * contribution + 0.5 * 1 / sigmoid(spend_time)
    k = 0
    tolerant_time = 0.008
    for i in range(spend_time):
        if i > tolerant_time:
            k = k + 1
    # top_k_indices = np.argsort(contribution)[-k:]  
    top_k_indices = np.argsort(S_contribution)[-k:] # 找到贡献度最大的 k 个头的索引
    # print(f"贡献度最大的 {k} 个头的索引: {top_k_indices}")

    # 4. 生成一个新的全为 0 的列表，长度为 num_head，将保留的索引位置置为 1
    mask = [0] * num_head
    print("num_head: ", num_head)
    # print("mask: ", mask)
    # print("top_k_indices: ", top_k_indices)
    for idx in top_k_indices:
        # print("idx: ", idx)
        mask[idx] = 1
    # print(f"生成的 mask: {mask}")
    return mask

async def send_request(session, server, q, k):
    """异步发送请求，计算 qk = q @ k.T"""
    # data = {'q': q.tolist(), 'k': k.tolist(), 'send_time': time.time()} 
    buffer = io.BytesIO()
    torch.save({"q": q, "k": k}, buffer)
    buffer.seek(0)
    # json_data = orjson.dumps(data)  # 转为字节
    # content_length = len(json_data)  # 手动计算长度
    # print("content_length: ", content_length)
    # print("q: ", q)
    
    headers = {"Content-Type": "application/json"}
    async with session.post(
        server,
        data=buffer.read(),
        headers={
            "Content-Type": "application/octet-stream",
            }    ) as response:
        
        # print("Sent headers:", response.request.headers)
        response_bytes = await response.read()
        result_tensor = torch.load(io.BytesIO(response_bytes), map_location="cpu")
        return result_tensor

async def async_distributed_matmul(q1, k1, q2, k2):
    """并行发送计算请求，并等待结果返回"""
    server_1 = "http://192.168.1.107:8000/compute"
    server_2 = "http://192.168.1.108:8000/compute"

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10,keepalive_timeout=30)) as session:
        # 创建异步任务，保证同时发送请求
        task1 = asyncio.create_task(send_request(session, server_1, q1, k1))
        task2 = asyncio.create_task(send_request(session, server_2, q2, k2))

        # 等待所有任务完成
        res1, res2 = await asyncio.gather(task1, task2)

    return torch.cat([res1, res2], dim=0)

def distributed_matmul(query_states, key_states):
    """同步函数中运行 asyncio"""
    # print("query_states: ", query_states.shape)
    # 切分数据
    q1, k1 = query_states[:16, :, :], key_states[:16, :, :]
    q2, k2 = query_states[16:, :, :], key_states[16:, :, :]

    # 获取当前事件循环
    loop = asyncio.get_event_loop()

    if loop.is_running():
        # 如果事件循环已在运行（例如在 Jupyter Notebook 或 Web 服务器中）
        future = asyncio.ensure_future(async_distributed_matmul(q1, k1, q2, k2))
        final_result = loop.run_until_complete(future)
    else:
        # 在普通 Python 环境下运行
        final_result = loop.run_until_complete(async_distributed_matmul(q1, k1, q2, k2))
    print("Final Result Shape:", final_result.shape)
    return final_result
    # # 解析返回结果并合并
    # if 'result' in res1 and 'result' in res2:
    #     print("正在合并")
    #     print("res1: ", np.array(res1['result']).shape)
    #     print("res2: ", np.array(res2['result']).shape)
    #     final_result = torch.cat([torch.tensor(res1['result']), torch.tensor(res2['result'])], dim=0)
    #     print("Final Result Shape:", final_result.shape)
    #     return final_result
    # else:
    #     print("Error in computation!")
    #     return None
def send_tensor(tensor, dst):
    # 先发送形状信息
    shape_tensor = torch.tensor(tensor.shape, dtype=torch.int64)
    dist.send(tensor=shape_tensor, dst=dst)

    # 再发送实际数据
    dist.send(tensor=tensor, dst=dst)

def recv_tensor(src, dtype=torch.float32):
    shape_attn_scores = torch.empty(3, dtype=torch.int64)
    dist.recv(tensor=shape_attn_scores, src=src)
    shape = tuple(shape_attn_scores.tolist())

    tensor = torch.empty(shape, dtype=dtype)
    dist.recv(tensor=tensor, src=src)
    return tensor



def run_distributed_inference(query_states, key_states):
    # master_addr="192.168.1.101"
    # master_port="29500"
    # world_size=2
    # rank=0

    # os.environ["MASTER_ADDR"] = master_addr
    # os.environ["MASTER_PORT"] = master_port
    # os.environ["RANK"] = str(rank)
    # os.environ["WORLD_SIZE"] = str(world_size)

    # # dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # dist.init_process_group(
    #     backend='gloo',
    #     init_method=f'tcp://{master_addr}:{master_port}',  # ← 主节点IP和端口
    #     rank=rank,                                   # ← 主节点是 rank 0
    #     world_size=world_size,                             # ← 总共两个进程（主+从）
    #     timeout=datetime.timedelta(seconds=1800)
    # )
    # print("初始化完成")
    # print("q_shape: ", query_states.shape)
    num_heads, seq_len, head_dim = query_states.shape
    q1, k1 = query_states[:10, :, :].contiguous().cpu(), key_states[:10, :, :].contiguous().cpu()
    q2, k2 = query_states[10:, :, :].contiguous().cpu(), key_states[10:, :, :].contiguous().cpu()
    print("q1_shape: ", q1.shape)
    send_tensor(q1, dst=1)
    send_tensor(k1, dst=1)
    print("数据已发送，等待接收计算结果...")
    # 主机自己处理前 16 个 head
    output1 = torch.matmul(q2, k2.transpose(-1, -2))  # [16, seq_len, seq_len]
    # 等待 Jetson 返回后 16 个 head 的结果
    output2 = recv_tensor(src=1)
    # output2 = torch.empty(16, seq_len, seq_len, dtype=torch.float16)
    # dist.recv(output2, src=1)
    # 合并最终输出
    output = torch.cat([output1, output2], dim=0)  # [32, seq_len, seq_len]
    return output

class DynamicPruning:
    def __init__(self):
        """
        初始化动态剪枝类。
        
        参数:
            threshold (float): 剪枝阈值，默认值为 0.5。
            alpha (float): 基于统计分布的阈值调整参数，默认值为 1.0。
        """
        # self.threshold = threshold
        # self.alpha = alpha

    def compute_importance_scores(self, attention_weights):
        """
        计算每个注意力头的重要性得分（基于注意力权重的方差）。
        
        参数:
            attention_weights (torch.Tensor): 注意力权重矩阵，形状为 (num_heads, seq_len, seq_len)。
        
        返回:
            importance_scores (torch.Tensor): 每个头的重要性得分，形状为 (num_heads,)。
        """
        # 计算每个头的注意力权重方差
        variances = torch.var(attention_weights, dim=(1, 2))  # 形状: (num_heads,)
        return variances

    def normalize_scores(self, importance_scores):
        """
        对重要性得分进行归一化处理。
        
        参数:
            importance_scores (torch.Tensor): 重要性得分，形状为 (num_heads,)。
        
        返回:
            normalized_scores (torch.Tensor): 归一化后的重要性得分，形状为 (num_heads,)。
        """
        min_score = torch.min(importance_scores)
        max_score = torch.max(importance_scores)
        normalized_scores = (importance_scores - min_score) / (max_score - min_score + 1e-9)  # 防止除零
        return normalized_scores

class MappingTable:
    def __init__(self):
        """
        初始化映射表类。
        
        参数:
            similarity_threshold (float): 近似度检索的阈值，默认值为 0.8。
        """
        self.table = {}  # 映射表，存储隐藏状态及其对应头的重要性得分
        # self.similarity_threshold = similarity_threshold

    def add_entry(self, hidden_state, contribution, file_path):
        """
        向映射表中添加新的条目。
        
        参数:
            hidden_state (torch.Tensor): 隐藏状态，形状为 (hidden_dim,)。
            importance_scores (torch.Tensor): 对应头的重要性得分，形状为 (num_heads,)。
        """
        hidden_state_list = hidden_state.tolist()
        contribution_list = contribution.detach().numpy().tolist()
        global map_id
        data = {"id": map_id,
                "hidden_state": hidden_state_list,
                "contribution": contribution_list}
        map_id += 1
        with open(file_path, "a", encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False)+"\n")
        # self.table[tuple(hidden_state.flatten().tolist())] = importance_scores
        
    def save_mapping_table_to_json(self, file_path):
        """
        将映射表存储为 JSON 文件。
        
        参数:
            file_path (str): JSON 文件路径。
        """
        serialized_table = {str(k): v.tolist() for k, v in self.table.items()}
        with open(file_path, 'w') as f:
            json.dump(serialized_table, f)

    def load_mapping_table_from_json(self, file_path):
        """
        从 JSON 文件加载映射表。
        
        参数:
            file_path (str): JSON 文件路径。
        
        返回:
            mapping_table (dict): 映射表，键为隐藏状态的元组，值为重要性得分的张量。
        """
        with open(file_path, 'r') as f:
            serialized_table = json.load(f)
        return {eval(k): torch.tensor(v) for k, v in serialized_table.items()}
    


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.dropout = config.attention_dropout
        self.enable_bias = config.enable_bias
        self.pruner = DynamicPruning()
        self.active_heads = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                             1, 1, 1, 1, 1, 1, 1, 1]  # OPT指定活跃的头
        self.active_heads_num = self.active_heads.count(1)
        self.computed_heads = [index for index, value in enumerate(self.active_heads) if value == 1]
        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        self.parallel = True
        self.dist_able = True
        self.process_new_hs = False

        # self.dist_group_number = 0
        self.mapping_table = MappingTable()
        self.table = self.mapping_table.table
        # if table_file_path is not None:
        #     self.table = self.mapping_table.load_mapping_table_from_json(table_file_path)
        
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        # s_time = time.time()
        pca_hs = np.mean(hidden_states[0].cpu().detach().numpy(), axis=1).reshape(-1, 1)
        # pca_hs, _ = pca(hidden_states[0], 1) # (287, 128)
        # e_time = time.time()
        # print("pca的时间：", e_time - s_time)
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling   
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        # print("query: ", query_states.size())
        # print("key: ", key_states.size())
        # print("value: ", value_states.size())       
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        # s_time = time.time()
        self.computed_heads = [index for index, value in enumerate(self.active_heads) if value == 1]
        if self.process_new_hs == True:
            self.computed_heads = process_new_hidden_state(pca_hs, mapping_table, cluster_centers, self.num_heads)
        # [0,1,0,1,0.....,0]
        # print("self.computed_heads!!!!: ", self.computed_heads)
        
        
        # if self.parallel == False:
        #     self.computed_heads = [index for index, value in enumerate(self.active_heads) if value == 1]
        
        
        # e_time = time.time()
        # print("获得可计算的头的时间：", e_time - s_time)
        
        # print("self.computed_heads: ", self.computed_heads)
        # print("query shape: ", query_states.shape)
        # print("query: ", query_states[0][0][0:3])
        # temp_attn_weights = torch.bmm(query_states, key_states.transpose(1, 2)) # [32, 10, 10]
        # print("temp_attn_weights: ", temp_attn_weights)

        
        # print("query: ", query_states.size()) # [32 , 10, 64]
        # print("key: ", key_states.size()) # [32 , 10, 64]
        # print("value: ", value_states.size()) # [32 , 10, 64]
        # 计算每个头的 QK，只计算活跃头
        qk_list = []
        q_list = []
        k_list = []
        v_list = []
        idx = 0
        # TODO:如果有问题,这这里注释掉
        # Q = query_states[self.computed_heads, :, :]
        # K = key_states[self.computed_heads, :, :]
        # V = value_states[self.computed_heads, :, :]

        # qk = torch.matmul(Q, K.transpose(-2, -1))

        # TODO:如果有问题,这这里取消注释
        if self.dist_able == True:
            active_mask = np.array(self.computed_heads, dtype=bool)

            masked_query = query_states[active_mask, :, :]
            # print("masked_query: ", len(masked_query))
            masked_key = key_states[active_mask, :, :]
            # q = masked_query[0:16, :, :].tolist()
            # k = masked_key[0:16, :, :].tolist()
            # temp_attn_weights = distributed_matmul(masked_query, masked_key)
            temp_attn_weights = run_distributed_inference(masked_query, masked_key)
            # self.dist_group_number += 1
            # print("temp_attn_weights: ", temp_attn_weights.shape)
        else:
            for head_idx in self.computed_heads:

                if head_idx == 1:
                    # 获取当前头的 Q 和 K
                    q = query_states[idx, :, :]  # 选定的头
                    k = key_states[idx, :, :]    # 选定的头
                    # q_list.append(q)
                    # k_list.append(k)
                    # 计算 QK 和缩放
                    # qk = torch.bmm(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
                    # TODO 分布式推理注释掉下一行
                    qk = torch.matmul(q, k.transpose(-2, -1))
                    idx += 1
                    # 将计算的 QK 添加到列表中
                    qk_list.append(qk)
            temp_attn_weights = torch.stack(qk_list, dim=0)  # 根据需要拼接各个头的输出
        # print("temp_attn_weights: ", temp_attn_weights.shape)
        # TODO:如果有问题,这这里取消注释
        # attn_weights = F.pad(temp_attn_weights, pad_size, mode='constant', value=0)
        self.active_heads_num = self.computed_heads.count(1)
        print("self.computed_heads: ", self.computed_heads)
        print("self.active_heads_num: ", self.active_heads_num)
        attn_weights = F.pad(temp_attn_weights, (0, 0, 0, 0, 0, self.num_heads - self.active_heads_num))
        print("attn_weights: ", attn_weights.shape)
        att_variances = self.pruner.compute_importance_scores(attn_weights)
        contribution_heads = self.pruner.normalize_scores(att_variances)
        
        if self.add_table == True:
            self.mapping_table.add_entry(pca_hs, contribution_heads, table_file_path)
        
        # print("add的时间：", e_time - s_time)
        # print("attn_weights: ", attn_weights.shape)
        # print("attn_weights shape: ", attn_weights.shape)
        # print("attn_weights: ", attn_weights[0][0][0:3])
        src_len = key_states.size(1)
        output_dim = key_states.size(-1)
        device = attention_mask.device
        attn_weights = attn_weights.to(device)
        attn_weights_reshaped = None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # TODO:如果有问题,这里取消注释
        # attn_output = torch.bmm(attn_probs, v_selected)
        # TODO:如果有问题,这里注释掉!
        attn_probs = attn_probs.to(dtype=torch.float16)
        value_states = value_states.to(dtype=torch.float16)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = attn_output.to(self.out_proj.weight.dtype)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value
    
        # # if key_value_states are provided this layer is used as a cross-attention layer
        # # for the decoder
        # is_cross_attention = key_value_states is not None

        # bsz, tgt_len, _ = hidden_states.size()

        # # get query proj
        # query_states = self.q_proj(hidden_states) * self.scaling
        # # get key, value proj
        # if is_cross_attention and past_key_value is not None:
        #     # reuse k,v, cross_attentions
        #     key_states = past_key_value[0]
        #     value_states = past_key_value[1]
        # elif is_cross_attention:
        #     # cross_attentions
        #     key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        # elif past_key_value is not None:
        #     # reuse k, v, self_attention
        #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # else:
        #     # self_attention
        #     key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        #     value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # if self.is_decoder:
        #     # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        #     # Further calls to cross_attention layer can then reuse all cross-attention
        #     # key/value_states (first "if" case)
        #     # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        #     # all previous decoder key/value_states. Further calls to uni-directional self-attention
        #     # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        #     # if encoder bi-directional self-attention `past_key_value` is always `None`
        #     past_key_value = (key_states, value_states)

        # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        # key_states = key_states.view(*proj_shape)
        # value_states = value_states.view(*proj_shape)

        # src_len = key_states.size(1)
        # attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        # if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        #     raise ValueError(
        #         f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
        #         f" {attn_weights.size()}"
        #     )

        # if attention_mask is not None:
        #     if attention_mask.size() != (bsz, 1, tgt_len, src_len):
        #         raise ValueError(
        #             f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
        #         )
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        #     attn_weights = torch.max(
        #         attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        #     )
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        # if attn_weights.dtype == torch.float16:
        #     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        # else:
        #     attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # if layer_head_mask is not None:
        #     if layer_head_mask.size() != (self.num_heads,):
        #         raise ValueError(
        #             f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
        #             f" {layer_head_mask.size()}"
        #         )
        #     attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # if output_attentions:
        #     # this operation is a bit awkward, but it's required to
        #     # make sure that attn_weights keeps its gradient.
        #     # In order to do so, attn_weights have to be reshaped
        #     # twice and have to be reused in the following
        #     attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        # else:
        #     attn_weights_reshaped = None

        # attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_output = torch.bmm(attn_probs, value_states)

        # if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        # attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # attn_output = attn_output.transpose(1, 2)

        # # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # # partitioned aross GPUs when using tensor-parallelism.
        # attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        # attn_output = self.out_proj(attn_output)

        # return attn_output, attn_weights_reshaped, past_key_value


class OptFlashAttention2(OPTAttention):
    """
    OPT flash attention module. This module inherits from `OPTAttention` as the weights of the module stays untouched.
    The only required change would be on the forward pass where it needs to correctly call the public API of flash
    attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, _, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_length = query_states.shape[1]
        tgt_len = key_states.shape[-2]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = query_states.view(bsz, query_length, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_dropout = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, query_length, dropout=attn_dropout
        )

        attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_weights_reshaped)

        if not output_attentions:
            attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped, past_key_value

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention,
    "flash_attention_2": OptFlashAttention2,
}


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = OPT_ATTENTION_CLASSES[config._attn_implementation](config=config, is_decoder=True)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            # print("past_key_values: ", past_key_values)
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (linear layer).

    [`OPTForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    OPT_START_DOCSTRING,
)
class OPTForSequenceClassification(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value


@add_start_docstrings(
    """
    The OPT Model transformer with a span classification head on top for extractive question-answering tasks like SQuAD
    (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    OPT_START_DOCSTRING,
)
class OPTForQuestionAnswering(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.model = OPTModel(config)
        self.qa_outputs = nn.Linear(config.word_embed_proj_dim, 2)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForQuestionAnswering
        >>> import torch

        >>> torch.manual_seed(4)  # doctest: +IGNORE_RESULT
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> # note: we are loading a OPTForQuestionAnswering from the hub here,
        >>> # so the head will be randomly initialized, hence the predictions will be random
        >>> model = OPTForQuestionAnswering.from_pretrained("facebook/opt-350m")

        >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

        >>> inputs = tokenizer(question, text, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> answer_start_index = outputs.start_logits.argmax()
        >>> answer_end_index = outputs.end_logits.argmax()

        >>> answer_offset = len(tokenizer(question)[0])

        >>> predict_answer_tokens = inputs.input_ids[
        ...     0, answer_offset + answer_start_index : answer_offset + answer_end_index + 1
        ... ]
        >>> predicted = tokenizer.decode(predict_answer_tokens)
        >>> predicted
        ' a nice puppet'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        logits = self.qa_outputs(hidden_states)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + transformer_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

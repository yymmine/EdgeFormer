import os
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OPTConfig
from transformers.models.opt.modeling_opt import OPTDecoderLayer
import time 
import pandas as pd

os.environ["RANK"] = "0"
os.environ['MASTER_ADDR'] = "192.168.1.101"
os.environ['MASTER_PORT'] = "12345"
os.environ['WORLD_SIZE'] = '3'



# 计算是否有共同单词
def has_common_words(pred, true):
    pred_words = set(pred.split())  # 将预测的答案分解为单词集合
    true_words = set(true.split())  # 将真实答案分解为单词集合
    common_words = pred_words & true_words  # 计算交集
    once_accuracy = 1 if  len(common_words) > 0 else 0
    return once_accuracy  # 如果交集大小大于0，说明有共同单词


def generate_input(row, datasets_name):
    if datasets_name == 'boolq':
        question = row['question']
        passage = row['passage']
        answer = row['answer']
        label = 0 if answer == False else 1
        # print("input_text: ", input_text)
        return question, passage, label
    elif datasets_name == 'record':
        question = row['query']
        passage = row['passage']
        answer = row['answers']
        # print("passage: ", passage)
        # num_sample += 1
        # 将问题和 passage 合并为输入文本
        input_text = f"Question: {question} Passage: {passage} Answer:"
        return input_text, answer
    

class FrontNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.front_layers)])
        
    def forward(self, input_ids):
        embeddings = self.embed_tokens(input_ids)
        positions = self.embed_positions(input_ids)
        hidden_states = embeddings + positions
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """OPT的位置嵌入实现"""
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(num_embeddings, embedding_dim)
    
    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        return super().forward(positions.unsqueeze(0).expand_as(input_ids))

class MiddleNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.middle_layers)])
    
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

class BackNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.back_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return self.lm_head(hidden_states)
    

def run_rank(rank, world_size):
    # 初始化分布式环境
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    datasets_name = 'record'
    # 加载模型配置
    config = OPTConfig.from_pretrained("/media/yym/work/code/pre-model/opt-1.3b")
    tokenizer = AutoTokenizer.from_pretrained("/media/yym/work/code/pre-model/opt-1.3b")
    tokenizer.pad_token = tokenizer.eos_token
    # 使用数据集
    if datasets_name == 'record':
        # 加载 SuperGLUE 的 数据集
        df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')
    elif datasets_name == 'boolq':
        df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')
    # 显示数据的前几行
    print("Data Preview:")
    print(df.head())
    # 获取行数
    num_rows = len(df)
    num_sample = 1
    total_infer_time = 0
    average_accuracy = 0
    for index, row in df.iterrows():

        start_time = time.time()
        # input_text = generate_record_input(row)
        print("row: ", row)
        input_text, true_answer = generate_input(row, datasets_name)
        # print("question: ", question)
        # print("passage: ", passage)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        if rank == 0:  # 前端
            # 只加载前端参数
            model = FrontNet(config)
            front_dict = torch.load("/media/yym/work/code/pre-model/opt_1.3b_front.pth")
            
            # 加载嵌入层
            model.embed_tokens.load_state_dict(front_dict['embed_tokens'])
            model.embed_positions.load_state_dict(front_dict['embed_positions'])
            
            # 加载Transformer层
            for i, layer in enumerate(model.layers):
                layer.load_state_dict(front_dict['layers'][i])
                
            # 处理输入
            input_ids = torch.tensor([[1, 2, 3]])  # 示例输入
            hidden_states = model(input_ids)
            dist.send(hidden_states, dst=1)
            shape_logits = torch.empty(3, dtype=torch.long, device="cuda:0")  # 假设是3D张量
            dist.recv(shape_logits, src=2)
            shape = tuple(shape_logits.tolist())  # 转换为Python元组
        
            # 2. 根据形状创建接收缓冲区
            logits = torch.empty(shape, dtype=torch.float16, device="cuda:0")
            dist.recv(logits, src=2)
            predicted_ids = torch.argmax(logits, dim=-1)
            model_answer = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            model_answer = str(model_answer)
            true_answer = str(true_answer)
            true_answer = true_answer.replace("'", "")
            # 参考文本和生成文本
            references = {
                '1': [true_answer]
            }
            hypotheses = {
                '1': [model_answer]
            }
            once_accuracy = has_common_words(model_answer, true_answer)
            print("true_answer: ", true_answer)
            print("model_answer: ", model_answer)
            print("========================")
            print("once_accuracy: ", once_accuracy)


            end_time = time.time()
            once_infer_time = end_time - start_time
            
            total_infer_time = (total_infer_time * (num_sample-1) + once_infer_time) / num_sample
            num_sample += 1
            print("========================")
            print("平均时长：", total_infer_time)
            print(f"[tested percent]: {num_sample}/{num_rows}")
            print()
            
        elif rank == 1:  # 中端
            model = MiddleNet(config) 
            middle_dict = torch.load("/media/yym/work/code/pre-model/opt_1.3b_middle.pth")
            
            for i, layer in enumerate(model.layers):
                layer.load_state_dict(middle_dict['layers'][i])
                
            # 接收前端输出
            hidden_states = torch.zeros((1, 3, config.hidden_size))  # 匹配输入形状
            dist.recv(hidden_states, src=0)
            
            # 处理并传递
            hidden_states = model(hidden_states)
            dist.send(hidden_states, dst=2)
            
        else:  # 后端

            model = BackNet(config)  # 包含layers和head
            back_dict = torch.load("/media/yym/work/code/pre-model/opt_1.3b_back.pth")
            
            for i, layer in enumerate(model.layers):
                layer.load_state_dict(back_dict['layers'][i])
            model.final_layer_norm.load_state_dict(back_dict['final_layer_norm'])
            model.lm_head.load_state_dict(back_dict['lm_head'])
            
            # 接收中端输出
            hidden_states = torch.zeros((1, 3, config.hidden_size))
            dist.recv(hidden_states, src=1)
            
            # 最终处理
            logits = model(hidden_states)
            shape_logits = torch.tensor(logits.shape, dtype=torch.long, device=logits.device)
            dist.send(shape_logits, dst=0)
            dist.send(logits, dst=0)
        
    dist.barrier()
    dist.destroy_process_group()


def init_distributed():
    # 从环境变量获取配置
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    
    # 初始化进程组
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

if __name__ == "__main__":
    run_rank(rank=1, world_size=3)
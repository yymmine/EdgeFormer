import torch
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# ✅ 初始化 accelerate
accelerator = Accelerator(mixed_precision="fp16")  # 启用 FP16

# ✅ 加载模型 & tokenizer
model_name = "/media/yym/work/code/pre-model/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)  # 启用 FP16
model.to(accelerator.device)


# ✅ 读取数据
datasets_name = "record"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')

# ✅ 设定 batch_size
batch_size = 1  # 根据 GPU 调整

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
        


def generate_predictions(batch_texts):
    """对 batch_texts 进行推理"""
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # **确保 inputs 在 GPU**
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"]).logits  # 计算 logits
    return logits

# **批量推理**
results = []
for i in range(0, len(df), batch_size):
    batch_rows = df.iloc[i : i + batch_size]
    batch_texts = [generate_input(row, datasets_name)[0] for _, row in batch_rows.iterrows()]  # 获取输入文本

    start_time = time.time()
    logits = generate_predictions(batch_texts)
    results.append(logits)
    end_time = time.time()

    print(f"Batch {i // batch_size + 1}, Time: {end_time - start_time:.2f}s")

    # ✅ 释放显存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # 确保释放完成
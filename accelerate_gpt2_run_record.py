import torch
import pandas as pd
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# ✅ 初始化 accelerate（自动检测设备，启用 FP16 以降低显存占用）
accelerator = Accelerator(mixed_precision="fp16")  # "fp16" 适用于推理，可减少一半显存

# ✅ 加载 GPT-2 模型 & tokenizer
model_name = "gpt2"  # 或 "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)  # 直接用 FP16 加载
model = accelerator.prepare(model)  # 让 accelerate 处理模型
tokenizer.pad_token = tokenizer.eos_token
model.eval()  # 进入推理模式

# ✅ 读取 RECORD 数据集
datasets_name = "record"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')

# ✅ 设定 batch_size（根据 GPU 调整）
batch_size = 2  # 8GB 显存建议 batch_size=1；12GB 可尝试 2~4；24GB 以上可尝试更大
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
    
    # ✅ 确保 inputs 在 GPU
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            max_new_tokens=400,  # 生成的最大 token 数
            do_sample=False,  # 关闭随机采样
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

    # ✅ 释放计算图，避免显存积累
    outputs = outputs.detach().cpu()
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# ✅ 逐 batch 推理
results = []
for i in range(0, len(df), batch_size):
    batch_rows = df.iloc[i : i + batch_size]
    batch_texts = [generate_input(row, datasets_name)[0] for _, row in batch_rows.iterrows()]  # 获取输入文本

    start_time = time.time()
    predictions = generate_predictions(batch_texts)
    results.extend(predictions)
    end_time = time.time()

    print(f"Batch {i // batch_size + 1}, Time: {end_time - start_time:.2f}s")

    # ✅ 释放显存，避免 OOM
    del batch_texts, predictions
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()  # 释放 CPU 内存

# ✅ 输出前 5 组预测结果
print(results[:5])

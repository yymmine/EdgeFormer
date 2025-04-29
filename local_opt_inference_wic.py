import torch
import time
import pandas as pd
from transformers import AutoTokenizer, OPTForCausalLM

# 加载模型和 tokenizer
model_name = "/media/yym/work/code/pre-model/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model.eval()

# 加载 WiC 验证集（请确保数据格式正确）
datasets_name = "wic"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')
print(f"总样本数: {len(df)}")
# 评估指标
correct = 0
total = 0
total_time = 0

# 遍历数据集进行推理
for _, row in df.iterrows():
    # 构造输入文本
    sentence1, sentence2, word, label = row["sentence1"], row["sentence2"], row["word"], row["label"]
    input_text = f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nWord: {word}\nSame meaning? Yes or No:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # 记录开始时间
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_new_tokens=200, use_cache=False, eos_token_id=tokenizer.eos_token_id)
    
    # 计算时延
    elapsed_time = time.time() - start_time
    total_time += elapsed_time

    # 解析输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取模型的回答
    if "Yes" in generated_text:
        pred_label = 1
    elif "No" in generated_text:
        pred_label = 0
    else:
        pred_label = -1  # 处理异常情况

    # 计算准确率
    if pred_label == label:
        correct += 1
    total += 1
    print(f"已推理数量: {total}")
    print(f"latency: {elapsed_time:.4f}s")
# 计算最终的评估结果
accuracy = correct / total
avg_latency = total_time / total

print(f"Average Latency: {avg_latency:.4f} seconds")
print(f"Accuracy: {accuracy:.4f}")

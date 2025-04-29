import torch
import pandas as pd
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和 Tokenizer
model_name = "/media/yym/work/code/pre-model/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()  # 进入推理模式

# 读取 MultiRC 数据集的验证集
datasets_name = "multirc"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')
print(df.head())


# 5. 设定 batch_size（OPT 显存占用较高）
batch_size = 1  # OPT-1.3B 占用较大，建议 batch_size=1~2

# 生成输入文本的函数
def generate_input(row):
    """
    将 MultiRC 数据格式转换为文本输入
    """
    passage = row["paragraph"]
    question = row["question"]
    answer = row["answer"]  # 真实答案
    return f"Passage: {passage}\nQuestion: {question}\nAnswer:", answer

# 4. 进行推理
correct = 0
total = 0
latencies = []
for index, row in df.iterrows():
    input_text, true_answer = generate_input(row)

    # Tokenize 输入
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    # 记录推理开始时间
    start_time = time.time()

    # 生成回答
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=400,  # 控制生成长度
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id
        )

    # 记录推理结束时间
    end_time = time.time()
    latencies.append(end_time - start_time)

    # 解码输出
    pred_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # 计算准确率（简单匹配）
    if true_answer.lower() in pred_text.lower():
        correct += 1
    total += 1
    print(f"推理时延: {end_time - start_time:.4f}s")

# 5. 输出最终结果
average_latency = sum(latencies) / len(latencies)
accuracy = correct / total

print(f"平均推理时延: {average_latency:.4f} 秒")
print(f"MultiRC 验证集准确率: {accuracy:.4%}")

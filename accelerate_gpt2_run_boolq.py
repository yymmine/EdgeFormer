import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# ✅ 初始化 `accelerator`
accelerator = Accelerator()

# ✅ 加载 GPT-2
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model = accelerator.prepare(model)  # 让模型支持 `accelerate`
model.eval()

# ✅ 加载 BoolQ 数据集
datasets_name = "boolq"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')

# ✅ 处理 BoolQ 格式
def generate_input(row):
    passage = row.get("passage", "No passage provided")
    question = row.get("question", "No question provided")
    answer = row.get("answer", None)  # 可能为空

    # 构造 GPT-2 需要的输入格式
    input_text = f"文章: {passage}\n问题: {question}\n答案是？"

    return input_text, answer

# ✅ 进行推理，并计算推理时延
predictions = []
for _, row in df.iterrows():
    input_text, true_answer = generate_input(row)

    # Tokenization
    inputs = tokenizer(input_text, return_tensors="pt").to(accelerator.device)

    # 记录推理开始时间
    start_time = time.time()

    # 推理
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)

    # 记录推理结束时间
    end_time = time.time()
    inference_time = end_time - start_time  # 计算时延

    # 解码答案
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 记录结果
    predictions.append((input_text, model_answer, true_answer, inference_time))

    # ✅ 打印每条数据的推理时间
    print(f"推理时延: {inference_time:.4f} 秒")
    # print(f"输入: {input_text}")
    # print(f"GPT-2 预测: {model_answer}")
    # print(f"真实答案: {'Yes' if true_answer else 'No'}\n")

# ✅ 统计平均推理时延
avg_time = sum([t for _, _, _, t in predictions]) / len(predictions)
print(f"\n平均推理时延: {avg_time:.4f} 秒/条数据")

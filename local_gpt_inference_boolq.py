import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")


model_name = "/media/yym/work/code/pre-model/gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_name = "gpt2"
# GPT-2 没有默认 pad token，需要手动指定
tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()


datasets_name = "record"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')


def generate_input(row):
    passage = row.get("passage", "No passage provided")
    question = row.get("question", "No question provided")
    answer = row.get("answer", None)  # 可能为空

    # 构造 GPT-2 需要的输入格式
    input_text = f"文章: {passage}\n问题: {question}\n答案是？"

    return input_text, answer


predictions = []
for _, row in df.iterrows():
    input_text, true_answer = generate_input(row)

    # Tokenization
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    # 记录推理开始时间
    start_time = time.time()

    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=201,  # 生成最多 200 个 Token
            min_new_tokens=200,
            eos_token_id=tokenizer.eos_token_id
        )

    # 记录推理结束时间
    end_time = time.time()
    inference_time = end_time - start_time  # 计算时延

    # 解码答案
    model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 记录结果
    predictions.append((input_text, model_answer, true_answer, inference_time))

    # print(f"输入: {input_text}")
    # print(f"GPT-2 预测 (长度: {len(model_answer)} 个字符): {model_answer}")
    # print(f"真实答案: {'Yes' if true_answer else 'No'}\n")
    print(f"推理时延: {inference_time:.4f} 秒")

avg_time = sum([t for _, _, _, t in predictions]) / len(predictions)
print(f"\n平均推理时延: {avg_time:.4f} 秒/条数据")

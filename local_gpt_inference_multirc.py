import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import time

# 加载模型和 tokenizer
model_name = "/media/yym/work/code/pre-model/gpt2-large"
# model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 避免 padding 报错
model = GPT2LMHeadModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载 MultiRC 验证集
datasets_name = "multirc"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')

# 推理函数（使用 generate 限制生成 token 数）
def infer_answer_generate(question, passage):
    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            use_cache=False,
            min_new_tokens=400,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_answer = full_output[len(prompt):].strip().lower()

    # 简单关键词判断 yes / no
    if any(x in generated_answer for x in ["yes", "true", "correct"]):
        return 1
    elif any(x in generated_answer for x in ["no", "false", "incorrect"]):
        return 0
    else:
        return 0  # fallback 兜底为 0（可改成随机）

# 评估函数
def evaluate_multirc(df):
    correct = 0
    total = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        passage = row["paragraph"]
        question = row["question"]
        label = row["label"]

        pred = infer_answer_generate(question, passage)
        if pred == label:
            correct += 1
        total += 1

    return correct / total

# 执行评估
s_time = time.time()
accuracy = evaluate_multirc(df)
e_time = time.time()
avg_latency = (e_time - s_time) / len(df)

print(f"\nAverage latency per example: {avg_latency:.4f}s")
print(f"Accuracy on MultiRC validation set: {accuracy * 100:.2f}%")

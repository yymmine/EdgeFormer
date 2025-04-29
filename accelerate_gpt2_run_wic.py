import torch
import pandas as pd
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# 1. 初始化 Accelerate
accelerator = Accelerator()
device = accelerator.device

# 2. 加载模型和 tokenizer
model_name = "/media/yym/work/code/pre-model/gpt2-large"  # 或 "gpt2", "gpt2-medium", "gpt2-xl"
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()

# 3. 加载 WiC 验证集
df = pd.read_parquet("/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/wic/validation-00000-of-00001.parquet")

# 4. 单条推理函数
def infer_one_wic(word, sentence1, sentence2):
    prompt = f'The word "{word}" appears in two sentences:\n1: {sentence1}\n2: {sentence2}\nDoes "{word}" have the same meaning in both sentences? Answer yes or no:\n'
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            min_new_tokens=150,
            max_new_tokens=201,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated = output_text[len(prompt):].strip().lower()

    if "yes" in generated:
        return 1
    elif "no" in generated:
        return 0
    else:
        return -1  # 无法判断

# 5. 推理并评估
correct = 0
total = 0
invalid = 0
start_all = time.time()

for idx, row in df.iterrows():
    word = row["word"]
    s1 = row["sentence1"]
    s2 = row["sentence2"]
    label = int(row["label"])

    start = time.time()
    pred = infer_one_wic(word, s1, s2)
    latency = time.time() - start

    print(f"\n[✓] Example {idx + 1}")
    print(f"Time: {latency:.2f}s")
    print(f"Predicted: {pred}, Label: {label}")

    if pred == -1:
        invalid += 1
    elif pred == label:
        correct += 1
    total += 1

    torch.cuda.empty_cache()
    gc.collect()

    if idx >= 9:  # 先测试前10条
        break

end_all = time.time()

print("\n====================")
print(f"Evaluated: {total} examples")
print(f"Correct: {correct}")
print(f"Invalid predictions: {invalid}")
print(f"Accuracy: {correct / total * 100:.2f}%")
print(f"Avg latency per example: {(end_all - start_all) / total:.2f}s")

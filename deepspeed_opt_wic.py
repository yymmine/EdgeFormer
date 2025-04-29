import torch
import pandas as pd
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

# 1. 配置 DeepSpeed
ds_config = {
    "train_batch_size": 1,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 0}
}
HfDeepSpeedConfig(ds_config)

# 2. 加载 OPT-1.3B 模型和 tokenizer
model_path = "/media/yym/work/code/pre-model/opt-1.3b"  # 修改为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model = deepspeed.init_inference(
    model,
    mp_size=1,
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=True
)
model.eval()

# 3. 加载 WiC 验证集
df = pd.read_parquet("/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/wic/validation-00000-of-00001.parquet")

# 4. 单条推理函数
def infer_one_wic(word, sentence1, sentence2):
    prompt = f"The word \"{word}\" appears in two sentences:\n1: {sentence1}\n2: {sentence2}\nDoes \"{word}\" have the same meaning in both sentences? Answer yes or no:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            min_new_tokens=200,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated = full_output[len(prompt):].strip().lower()

    if "yes" in generated:
        return 1
    elif "no" in generated:
        return 0
    else:
        return -1  # 无法判断

# 5. 推理并评估准确率
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

    # 只跑前几条测试
    if idx >= 9:
        break

end_all = time.time()

# 6. 输出评估结果
print("\n====================")
print(f"Evaluated: {total} examples")
print(f"Correct: {correct}")
print(f"Invalid predictions: {invalid}")
print(f"Accuracy: {correct / total * 100:.2f}%")
print(f"Avg latency per example: {(end_all - start_all) / total:.2f}s")

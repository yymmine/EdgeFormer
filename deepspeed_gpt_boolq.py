import torch
import pandas as pd
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

# ============ 1. DeepSpeed config ============
ds_config = {
    "train_batch_size": 1,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 0
    },
    "tensor_parallel": {
        "enabled": False
    }
}
HfDeepSpeedConfig(ds_config)

model_name = "/media/yym/work/code/pre-model/gpt2-large"
# model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model = deepspeed.init_inference(
    model,
    mp_size=1,
    dtype=torch.float32,
    replace_method="auto",
    replace_with_kernel_inject=True,
)
model.eval()

datasets_name = "boolq"
df = pd.read_parquet(
    f"/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet"
)
batch_size = 1

# ============ 4. 构造输入 ============
def generate_input(row):
    question = row["question"]
    passage = row["passage"]
    label = 0 if row["answer"] is False else 1
    input_text = f"Question: {question} Passage: {passage} Answer:"
    return input_text, label

# ============ 5. 推理函数 ============
def generate_predictions(batch_texts):
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            min_new_tokens=200,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# ============ 6. 推理 & 实时准确率 ============
correct_count = 0
total_count = 0
total_time = 0.0
results = []

for i in range(0, len(df), batch_size):
    batch_rows = df.iloc[i : i + batch_size]
    batch_texts, batch_labels = [], []

    for _, row in batch_rows.iterrows():
        input_text, label = generate_input(row)
        batch_texts.append(input_text)
        batch_labels.append(label)

    start_time = time.time()
    predictions = generate_predictions(batch_texts)
    end_time = time.time()

    results.extend(predictions)
    total_time += end_time - start_time

    # 简单规则：判断生成文本中是否包含 "yes"/"true" or "no"/"false"
    for pred, label in zip(predictions, batch_labels):
        pred_lower = pred.lower()
        if label == 1 and any(x in pred_lower for x in ["yes", "true"]):
            correct_count += 1
        elif label == 0 and any(x in pred_lower for x in ["no", "false"]):
            correct_count += 1
        total_count += 1

    acc = correct_count / total_count
    avg_latency = total_time / total_count
    print(f"Batch {i // batch_size + 1}, Time: {end_time - start_time:.4f}s, "
          f"Running Acc: {acc:.4f}, Avg Latency: {avg_latency:.4f}s")

    # 显存清理
    del batch_texts, predictions
    torch.cuda.empty_cache()
    gc.collect()

# ============ 7. 输出示例 ============
print("\nExample predictions:")
print("\n".join(results[:5]))

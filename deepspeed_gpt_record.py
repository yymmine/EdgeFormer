import torch
import pandas as pd
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed

# ============ 1. DeepSpeed 初始化配置 ============
from transformers import AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig

# ============ 2. 加载模型 ============
model_name = "/media/yym/work/code/pre-model/gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model = deepspeed.init_inference(
    model,
    mp_size=1,  # 单 GPU
    dtype=torch.float32,
    replace_method='auto',
    replace_with_kernel_inject=True,
)
model.eval()

# ============ 3. 加载数据 ============
datasets_name = "record"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')
batch_size = 1

# ============ 4. 数据构建 ============
def generate_input(row, datasets_name):
    if datasets_name == 'record':
        question = row['query']
        passage = row['passage']
        answer = row['answers']
        input_text = f"Question: {question} Passage: {passage} Answer:"
        return input_text, answer
    return "", []

def generate_predictions(batch_texts):
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.cuda() for k, v in inputs.items()}  # DeepSpeed 只在 CUDA 上跑
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            min_new_tokens=400,
            max_new_tokens=500,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    outputs = outputs.detach().cpu()
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# ============ 5. 推理 + 实时准确率 ============
results = []
correct_count = 0
total_count = 0
total_time = 0.0

for i in range(0, len(df), batch_size):
    batch_rows = df.iloc[i : i + batch_size]
    batch_texts = []
    batch_answers = []

    for _, row in batch_rows.iterrows():
        input_text, answer_list = generate_input(row, datasets_name)
        batch_texts.append(input_text)
        batch_answers.append(answer_list)

    start_time = time.time()
    predictions = generate_predictions(batch_texts)
    end_time = time.time()

    batch_time = end_time - start_time
    total_time += batch_time
    results.extend(predictions)

    for pred_text, gold_answers in zip(predictions, batch_answers):
        if any(ans.lower() in pred_text.lower() for ans in gold_answers):
            correct_count += 1
        total_count += 1

    acc = correct_count / total_count
    avg_latency = total_time / total_count
    print(f"Batch {i // batch_size + 1}, Time: {batch_time:.4f}s, "
          f"Running Acc: {acc:.4f}, Avg Latency: {avg_latency:.4f}s")

    # 清理缓存
    del batch_texts, predictions
    torch.cuda.empty_cache()
    gc.collect()

# ============ 6. 打印结果预览 ============
print("\nExample predictions:")
print("\n".join(results[:5]))

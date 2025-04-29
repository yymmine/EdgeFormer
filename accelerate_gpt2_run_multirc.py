import torch
import pandas as pd
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

# 1. 初始化 Accelerator（自动适配 GPU / CPU）
accelerator = Accelerator()

# 2. 设置设备
device = accelerator.device

# 3. 加载本地 GPT-2 模型 & tokenizer
model_path = "gpt2"  # 修改为本地模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 4. 解决 padding 报错（GPT-2 默认无 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 5. 加载 GPT-2 到 Accelerator
model = AutoModelForCausalLM.from_pretrained(model_path)
model = accelerator.prepare(model)
model.eval()

# 6. 加载 MultiRC 验证集
datasets_name = "multirc"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')


# 7. 设置 batch_size（GPT-2 需要较小 batch）
batch_size = 1 # 8GB 显存建议 4，12GB 可用 8，24GB 可用 16+

def generate_input(row, datasets_name):
    """
    生成 GPT-2 输入
    """
    if datasets_name.lower() == "multirc":
        # 兼容不同数据集格式
        passage = row.get("passage") or row.get("paragraph") or "No passage"
        question = row.get("question", "No question")
        answer = row.get("answer", "No answer")

        input_text = f"文章: {passage}\n问题: {question}\n答案: {answer}\n这个答案是正确的吗？"

        true_answer = row.get("label", None)  # 可能为空

        return input_text, true_answer

    else:
        raise ValueError(f"Unsupported dataset: {datasets_name}")



def generate_predictions(batch_texts):
    """对 batch_texts 进行推理"""
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # 8. 确保输入数据在 Accelerator 设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],  # ✅ 传入 attention_mask，确保正确计算
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False
        )

    outputs = accelerator.gather(outputs)  # ✅ 适配多 GPU
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# ✅ 9. 逐 batch 推理
results = []
for i in range(0, len(df), batch_size):
    batch_rows = df.iloc[i : i + batch_size]
    batch_texts = [generate_input(row, datasets_name)[0] for _, row in batch_rows.iterrows()]

    start_time = time.time()
    predictions = generate_predictions(batch_texts)
    results.extend(predictions)
    end_time = time.time()

    print(f"Batch {i // batch_size + 1}, Time: {end_time - start_time:.2f}s")

    # ✅ 10. 释放显存，避免 OOM
    del batch_texts, predictions
    torch.cuda.empty_cache()
    gc.collect()

# ✅ 11. 输出前 5 组预测结果
print(results[:5])

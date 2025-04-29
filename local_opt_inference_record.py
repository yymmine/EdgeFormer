import torch
import pandas as pd
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载 OPT-1.3B 模型 & tokenizer
model_name = "/media/yym/work/code/pre-model/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 解决 padding 报错
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # OPT 默认没有 pad_token，需要手动设置

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

model.to(device)
model.eval()

# 4. 读取 RECORD 数据集
datasets_name = "record"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')

# 5. 设定 batch_size（OPT 显存占用较高）
batch_size = 1  # OPT-1.3B 占用较大，建议 batch_size=1~2
past_key_values = None
def generate_input(row, datasets_name):
    if datasets_name == 'boolq':
        question = row['question']
        passage = row['passage']
        answer = row['answer']
        label = 0 if answer == False else 1
        # print("input_text: ", input_text)
        return question, passage, label
    elif datasets_name == 'record':
        question = row['query']
        passage = row['passage']
        answer = row['answers']
        # print("passage: ", passage)
        # num_sample += 1
        # 将问题和 passage 合并为输入文本
        input_text = f"Question: {question} Passage: {passage} Answer:"
        return input_text, answer
        


def generate_predictions(batch_texts):
    """对 batch_texts 进行推理"""
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # 6. 手动将数据加载到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model.generate(
            inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
            max_new_tokens=400,
            do_sample=False,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,  # OPT 需要手动设置 pad_token
            decoder_start_token_id=tokenizer.bos_token_id  # OPT 需要 decoder_start_token_id
        )

    outputs = outputs.detach().cpu()
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# ✅ 7. 逐 batch 推理
results = []
for i in range(0, len(df), batch_size):
    batch_rows = df.iloc[i : i + batch_size]
    batch_texts = [generate_input(row, datasets_name)[0] for _, row in batch_rows.iterrows()]

    start_time = time.time()
    predictions = generate_predictions(batch_texts)
    results.extend(predictions)
    end_time = time.time()

    print(f"Batch {i // batch_size + 1}, Time: {end_time - start_time:.2f}s")

    # ✅ 8. 释放显存，避免 OOM
    del batch_texts, predictions
    torch.cuda.empty_cache()
    gc.collect()

# ✅ 9. 输出前 5 组预测结果
print(results[:5])

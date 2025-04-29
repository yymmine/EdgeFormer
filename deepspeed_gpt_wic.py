import torch
import pandas as pd
import gc
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

# ============ 1. 配置 DeepSpeed ============
ds_config = {
    "train_batch_size": 1,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 0},
    "tensor_parallel": {"enabled": False}
}
HfDeepSpeedConfig(ds_config)

# ============ 2. 加载 GPT2-large 模型 ============
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

# ============ 3. 加载 WiC 验证集 ============
dataset_name = "wic"
df = pd.read_parquet(
    f"/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{dataset_name}/validation-00000-of-00001.parquet"
)

# ============ 4. 推理函数 ============
def infer_wic(sentence1, sentence2, word):
    prompt = (
        f'Word: "{word}"\n'
        f"Sentence 1: {sentence1}\n"
        f"Sentence 2: {sentence2}\n"
        f"Are the meanings of the word the same in both sentences? Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            min_new_tokens=400,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_text = output_text[len(prompt):].lower()

    if any(x in answer_text for x in ["yes", "true", "correct"]):
        return 1
    elif any(x in answer_text for x in ["no", "false", "incorrect"]):
        return 0
    else:
        return 0  # fallback 兜底

# ============ 5. 验证函数 ============
def evaluate_wic(df):
    correct = 0
    total = 0
    total_time = 0.0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        word = row["word"]
        label = row["label"]

        start = time.time()
        pred = infer_wic(s1, s2, word)
        total_time += (time.time() - start)

        if pred == label:
            correct += 1
        total += 1

        # 显存清理
        torch.cuda.empty_cache()
        gc.collect()

    return correct / total, total_time / total

# ============ 6. 执行验证 ============
accuracy, avg_latency = evaluate_wic(df)
print(f"\n[✓] Accuracy on WiC validation set: {accuracy * 100:.2f}%")
print(f"[✓] Average latency per sample: {avg_latency:.4f}s")

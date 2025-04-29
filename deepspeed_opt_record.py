import torch
import pandas as pd
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

# 1. DeepSpeed 配置
ds_config = {
    "train_batch_size": 1,
    "fp16": {"enabled": True},
    "zero_optimization": {"stage": 0}
}
HfDeepSpeedConfig(ds_config)

# 2. 加载模型和 tokenizer（OPT-1.3B）
model_path = "/media/yym/work/code/pre-model/opt-1.3b"  # 改成你的本地路径或 "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
model = deepspeed.init_inference(
    model,
    mp_size=1,
    dtype=torch.float32,
    replace_method="auto",
    replace_with_kernel_inject=True
)
model.eval()

# 3. 加载 RECORD 验证集
df = pd.read_parquet("/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/record/validation-00000-of-00001.parquet")

# 4. 单条推理函数
def infer_one_record(question, passage):
    prompt = f"Question: {question}\nPassage: {passage}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            min_new_tokens=400,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_output[len(prompt):].strip()

# 5. 遍历每条数据逐条推理
for idx, row in df.iterrows():
    question = row["query"]
    passage = row["passage"]

    start_time = time.time()
    output = infer_one_record(question, passage)
    elapsed = time.time() - start_time

    print(f"\n[✓] Example {idx + 1}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Answer: {output}")

    # 显存清理
    torch.cuda.empty_cache()
    gc.collect()

    # # 可设定只测试前几条
    # if idx >= 2:
    #     break

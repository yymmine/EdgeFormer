import torch
import deepspeed
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "/media/yym/work/code/pre-model/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()  # 进入推理模式

# 读取 MultiRC 数据集的验证集
datasets_name = "multirc"
df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')
print(df.head())

# 使用 DeepSpeed 加载模型
ds_engine = deepspeed.init_inference(
    model=model, dtype=torch.float16, replace_with_kernel_inject=True
)


model.eval()  # 设为推理模式

def generate_response(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = ds_engine.generate(
            inputs["input_ids"], max_new_tokens=500, min_new_tokens=400
        )
    latency = time.time() - start_time
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True), latency


total_latency = 0
total_samples = 0
correct_predictions = 0

for _, row in df.iterrows():
    input_text = f"Passage: {row['paragraph']}\nQuestion: {row['question']}\nAnswer:"
    predicted_answer, latency = generate_response(input_text)
    
    total_latency += latency
    total_samples += 1
    
    # 假设数据集中有 ground truth answer
    correct_predictions += int(predicted_answer.strip().lower() == row['answer'].strip().lower())
    print(f"Latency: {latency:.4f} seconds")
    # print(f"Predicted: {predicted_answer} | Actual: {row['answer']}")

# 计算平均时延和准确率
average_latency = total_latency / total_samples
accuracy = correct_predictions / total_samples

print(f"Average Latency: {average_latency:.4f} seconds")
print(f"Accuracy: {accuracy:.4f}")

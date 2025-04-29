import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# 1. 初始化模型
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, output_attentions=True).to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 2. 验证数据
val_samples = [
    "For four years we have waited expectantly for the pitter patter of tiny paws. Soon, that wait could finally be over. Tian Tian, the UK's only female giant panda, has conceived and could give birth to a cub as early as August. However Edinburgh Zoo, where the pandas live, have warned people 'not to get too excited' as the process is 'extremely complex'. Moreover, on the two previous occasions keepers inseminated Tian Tian - whose name means 'Sweetie' - she has failed to produce a panda cub. She was artificially inseminated again in March this year, but keepers at the zoo say implantation - when a fertilised egg attaches to the uterus - has not yet occurred."
]

# 3. 定义评估指标计算函数（修改为计算方差）
def compute_head_variance(inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取各层注意力权重 (n_layers, batch, n_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions)  
    _, seq_len = inputs["input_ids"].shape
    
    # 计算每个头的注意力方差
    head_variance = torch.zeros(model.config.n_layer, model.config.n_head)
    for layer in range(model.config.n_layer):
        for head in range(model.config.n_head):
            # 计算注意力分布的方差 (batch维度取平均)
            prob = attentions[layer, :, head].mean(0)  # (seq_len, seq_len)
            
            # 计算标准化方差（除以均匀分布的方差）
            uniform_var = 1.0 / seq_len  # 均匀分布的方差
            variance = torch.var(prob, dim=-1).mean() / uniform_var
            head_variance[layer, head] = variance
    
    return head_variance.cpu().numpy()

# 4. 执行评估
all_variance = []

for sample in tqdm(val_samples, desc="Analyzing heads"):
    inputs = tokenizer(sample, return_tensors="pt").to(device)
    variance = compute_head_variance(inputs)
    all_variance.append(variance)

# 5. 聚合结果
mean_variance = np.mean(all_variance, axis=0)

# 6. 自动识别冗余头 (基于方差)
THRESHOLD_VARIANCE = 0.3  # 低于此值认为注意力接近随机分布

redundant_heads = (mean_variance < THRESHOLD_VARIANCE)
redundant_ratio = redundant_heads.mean()
print(f"自动识别的冗余头比例: {redundant_ratio:.2%}")

# 7. 可视化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 注意力方差热力图
sns.heatmap(mean_variance, annot=True, fmt=".2f", cmap="YlGnBu", 
            ax=ax1, cbar_kws={'label': 'Normalized Attention Weight'})
cbar = ax1.collections[0].colorbar

# 设置颜色条标签字号为16
cbar.ax.yaxis.label.set_size(16)
# 设置刻度标签字号为16
cbar.ax.tick_params(labelsize=14)
# 设置Times New Roman字体
font_dict = {
    'fontsize': 16,          # 字号
    'fontweight': 'normal'   # 字重（可选bold）
}

ax1.set_title("Attention Head Weight", **font_dict)
ax1.set_xlabel("Head Index", **font_dict)
ax1.set_ylabel("Layer Index", **font_dict)
ax1.tick_params(axis='both', labelsize=14)
# 冗余头标记图
sns.heatmap(redundant_heads, cmap=["green", "red"], annot=True, 
            fmt="d", cbar=False, ax=ax2)
ax2.set_title(f"Redundant Heads (Red=True)\nTotal Redundant: {redundant_ratio:.2%}", **font_dict)
ax2.set_xlabel("Head Index", **font_dict)
ax2.set_ylabel("Layer Index", **font_dict)
ax1.tick_params(axis='both', labelsize=14)
plt.tight_layout()
plt.savefig("gpt2_head_variance_analysis.png", dpi=300)
plt.show()

# 8. 输出关键统计
print("\n注意力方差最高的5个头（最聚焦）：")
variance_df = pd.DataFrame(mean_variance)
print(variance_df.stack().sort_values(ascending=False).head(5))

print("\n注意力方差最低的5个头（最可能冗余）：")
print(variance_df.stack().sort_values().head(5))
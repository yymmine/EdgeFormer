import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import pandas as pd
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from transformers import GPT2LMHeadModel, GPT2Tokenizer
datasets_name = 'record'
# 选择模型，例如 GPT-2
model_name = "/media/yym/work/code/pre-model/gpt2-large"
# **初始化 tokenizer 和 GPT-2**
# model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# **移动模型到 GPU**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 计算是否有共同单词
def has_common_words(pred, true):
    pred_words = set(pred.split())  # 将预测的答案分解为单词集合
    true_words = set(true.split())  # 将真实答案分解为单词集合
    common_words = pred_words & true_words  # 计算交集
    once_accuracy = 1 if  len(common_words) > 0 else 0
    return once_accuracy  # 如果交集大小大于0，说明有共同单词


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
        

# 使用数据集
if datasets_name == 'record':
    # 加载 SuperGLUE 的 数据集
    df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')
elif datasets_name == 'boolq':
    df = pd.read_parquet(f'/media/yym/work/code/gpt-2-Pytorch/download_dataset/super_glue/{datasets_name}/validation-00000-of-00001.parquet')
# 显示数据的前几行
print("Data Preview:")
print(df.head())
# 获取行数
num_rows = len(df)
num_sample = 1
total_infer_time = 0
average_accuracy = 0
average_BLEU = 0
average_rouge = 0
average_cider = 0
for index, row in df.iterrows():

    start_time = time.time()
    # input_text = generate_record_input(row)
    print("row: ", row)
    input_text, true_answer = generate_input(row, datasets_name)

    s_time = time.time()
    # 进行推理
    # input_text = "DeepSpeed is a new feature that allows you to set the speed of your car to a certain speed. The speed of your car is determined by the speed of the car's engine. The speed of your car is determined by the speed of the engine."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output = model.generate(input_ids, 
                                max_length=1024, 
                                max_new_tokens=400, 
                                use_cache=False,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        print("推理完一次了")
    # print("question: ", question)
    # print("passage: ", passage)
    # inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # print("inputs: ", inputs)
    # print("inputs: ", inputs)
    # TODO: 原始：使用模型进行推理
    # with torch.no_grad():
    #     outputs = model.generate(inputs['input_ids'], max_new_tokens=20, eos_token_id=tokenizer.eos_token_id)
    #     model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    model_answer = tokenizer.decode(output[0], skip_special_tokens=True)
    model_answer = str(model_answer)
    true_answer = str(true_answer)
    true_answer = true_answer.replace("'", "")
    # # 初始化评估器
    # bleu_scorer = Bleu()
    # meteor_scorer = Meteor()
    # rouge_scorer = Rouge()
    # cider_scorer = Cider()
    # # 参考文本和生成文本
    # references = {
    #     '1': [true_answer]
    # }
    # hypotheses = {
    #     '1': [model_answer]
    # }
    # once_accuracy = has_common_words(model_answer, true_answer)
    # # 计算分数
    # bleu_score, _ = bleu_scorer.compute_score(references, hypotheses)
    # # meteor_score, _ = meteor_scorer.compute_score(references, hypotheses)
    # rouge_score, _ = rouge_scorer.compute_score(references, hypotheses)
    # cider_score, _ = cider_scorer.compute_score(references, hypotheses)
    # print("true_answer: ", true_answer)
    # print("model_answer: ", model_answer)

    # correct = 0
    # total = len(true_answer)
    # if model_answer.strip().lower() ==  true_answer.strip().lower():
    #     correct += 1
    # once_accuracy = correct / total
    print("========================")
    # print("label: ", label)


    #TODO: 注释掉的
    # print("once_accuracy: ", once_accuracy)
    # average_accuracy = (average_accuracy * (num_sample-1) + once_accuracy) / num_sample 
    # average_BLEU = (average_BLEU * (num_sample-1) + bleu_score[-1]) / num_sample
    # average_rouge = (average_rouge * (num_sample-1) + rouge_score) / num_sample
    # average_cider = (average_cider * (num_sample-1) + cider_score) / num_sample
    
    # print("average_accuracy: ", average_accuracy)
    # print("BLEU:", average_BLEU)
    # # print("METEOR:", meteor_score)
    # print("ROUGE-L:", average_rouge)
    # print("CIDEr:", average_cider)

    end_time = time.time()
    once_infer_time = end_time - start_time
    
    total_infer_time = (total_infer_time * (num_sample-1) + once_infer_time) / num_sample
    num_sample += 1
    # print(f"Question: {question}")
    # print(f"Passage: {passage}")
    print("========================")
    # 输出平均时延
    # print("once_infer_time: ", once_infer_time)
    print("平均时长：", total_infer_time)
 




    # # 输出结果
    # print(tokenizer.decode(output[0], skip_special_tokens=True))
    # e_time = time.time()
    # print("推理时延为：", e_time - s_time)
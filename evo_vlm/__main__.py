import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. load model
device = "cuda" if torch.cuda.is_available() else "cpu"
repo_id = "SakanaAI/EvoLLM-JP-v1-7B"
model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model.to(device)

# 入力テキストの準備
input_text = "関西弁で面白い冗談を言ってみて下さい。"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# テキストの生成
output_ids = model.generate(**inputs, max_length=400)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(generated_text)

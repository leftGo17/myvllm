from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = os.path.expanduser("~/hf/Qwen3-0.6B")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(model)

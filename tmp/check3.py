from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = os.path.expanduser("~/hf/Qwen3-0.6B")

tokenizer = AutoTokenizer.from_pretrained(model_name)


prompts = [
    "introduce yourself",  # * 15,
    "list all prime numbers within 100",  # * 15,
    "give me your opinion on the impact of artificial intelligence on society",  # * 15,
]  # * 30
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]
print(prompts)

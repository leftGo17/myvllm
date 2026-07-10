from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import os

# 关闭 Transformers 的普通日志/警告，只打印最后的模型回复。
logging.set_verbosity_error()

model_name = os.path.expanduser("~/hf/Qwen3-0.6B")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

# messages 是聊天格式输入。
# role: 说话者身份。这里 "user" 表示用户输入。
# content: 这条消息的文本内容。这里让模型回应 "hello"。
messages = [
    {
        "role": "user",
        "content": "hello",
    }
]

inputs = tokenizer.apply_chat_template(
    # messages: 上面定义的聊天消息列表。
    messages,
    # add_generation_prompt=True:
    # 在输入末尾加上“轮到 assistant 回复”的提示，让模型开始生成助手回复。
    add_generation_prompt=True,
    # enable_thinking=False:
    # Qwen3 支持 thinking 模式。关掉后，输出会直接是最终回答，不打印 <think> 内容。
    enable_thinking=False,
    # return_tensors="pt":
    # 返回 PyTorch Tensor，供 PyTorch 模型直接使用。
    return_tensors="pt",
    # return_dict=True:
    # 返回字典格式，例如 {"input_ids": ..., "attention_mask": ...}。
    return_dict=True,
)

# 把输入 Tensor 移动到模型所在设备。当前默认通常是 CPU；如果模型在 GPU，这里也会跟随到 GPU。
inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
print(inputs)
outputs = model.generate(
    # **inputs 会展开 input_ids、attention_mask 等模型需要的输入字段。
    **inputs,
    max_new_tokens=64,
    # do_sample=False:
    # 使用确定性生成，通常等价于每一步选择概率最高的 token。
    do_sample=False,
    # pad_token_id:
    # 生成时用于 padding 的 token id。这里用 eos_token_id，避免没有 pad token 时报警或报错。
    pad_token_id=tokenizer.eos_token_id,
)

# outputs 包含“原始输入 + 新生成内容”。
# inputs["input_ids"].shape[-1] 是输入长度；从这个位置开始切片，只取模型新生成的回复。
reply_ids = outputs[0, inputs["input_ids"].shape[-1] :]

# decode: 把 token id 转回字符串。
# skip_special_tokens=True: 去掉 <|...|> 这类特殊 token。
# strip(): 去掉开头和结尾多余空白。
reply = tokenizer.decode(reply_ids, skip_special_tokens=True)

# 打印模型对 "hello" 的回复。
print(reply)

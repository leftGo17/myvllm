from vllm import LLM, SamplingParams
import os

model_name = os.path.expanduser("~/hf/Qwen3-0.6B")

# LLM: vllm 的推理引擎入口，负责加载模型、管理 KV cache、调度请求。
llm = LLM(model=model_name)

# SamplingParams: 控制生成策略。
# temperature=0: 贪心解码，等价于每步选概率最高的 token，结果可复现。
# max_tokens: 最多生成的新 token 数。
sampling_params = SamplingParams(temperature=0, max_tokens=64)

conversations = [
    [{"role": "user", "content": "introduce yourself"}],
    [{"role": "user", "content": "list all prime numbers within 100"}],
]

# chat: 自动套用模型的 chat template（区别于直接 generate 裸文本），
# 再走和 generate 一样的批量推理流程。
outputs = llm.chat(conversations, sampling_params)

for output in outputs:
    print("prompt:", output.prompt)
    print("output:", output.outputs[0].text)
    print("-" * 40)

from vllm import LLM, SamplingParams
import os
import shutil

model_name = os.path.expanduser("~/hf/Qwen3-0.6B")
trace_dir = os.path.join(os.path.dirname(__file__), "trace")

# 每次跑之前清空旧的 trace，避免 trace 目录里堆积历史文件、混淆分析。
shutil.rmtree(trace_dir, ignore_errors=True)
os.makedirs(trace_dir, exist_ok=True)


llm = LLM(
    model=model_name,
    quantization="fp8",
    profiler_config={"torch_profiler_dir": trace_dir},
    # speculative_config={
    #     "method": "ngram",
    #     "num_speculative_tokens": 5,
    #     "prompt_lookup_max": 4,
    #     "prompt_lookup_min": 2,
    # },
    # 以下是不用手动开、vllm 本身默认就打开的快速推理手段：
    # enable_prefix_caching=True   前缀缓存，相同前缀的 KV block 直接复用
    # enable_chunked_prefill=True  chunked prefill，长 prompt 切块，和 decode 混批处理
    # enforce_eager=False          decode 阶段用 CUDA Graph 加速（配 cudagraph_mode=FULL_AND_PIECEWISE）
)

sampling_params_warm = SamplingParams(temperature=0, max_tokens=2)
sampling_params = SamplingParams(temperature=0, max_tokens=5)

conversations1 = [
    [{"role": "user", "content": "introduce yourself"}],
    [{"role": "user", "content": "list all prime numbers within 100"}],
]

conversations2 = [
    [{"role": "user", "content": "hello introduce yourself"}],
    [{"role": "user", "content": "please list all prime numbers within 100"}],
]

# start_profile/stop_profile: 只包住真正想采样的这段推理，
# 避免把模型加载、CUDA graph capture 等一次性开销也录进 trace。
llm.start_profile()

outputs = llm.chat(conversations1, sampling_params_warm)

outputs = llm.chat(conversations2, sampling_params)
llm.stop_profile()

for output in outputs:
    print("prompt:", output.prompt)
    print("output:", output.outputs[0].text)
    print("-" * 40)


# quantization="fp8": 在线（动态）量化。原始 checkpoint 仍是 bf16/fp16，
# vllm 在加载权重时实时量化成 fp8（权重用 per-tensor 静态 scale，
# 激活用 dynamic per-tensor scale），不需要提前准备量化好的 checkpoint。
#
# speculative_config: 投机解码。这里用 ngram 方法——不需要额外的draft模型，
# 直接在已生成的上下文里查找重复的 n-gram 来"猜"接下来的几个 token，
# 猜中就能一次验证多个 token，命中率取决于文本本身的重复度（复制粘贴/代码等场景效果最好）。
# 像 Qwen3-0.6B 这种本身已经很小的模型，没有合适的小 draft 模型可用，ngram 是最容易白嫖的方案。
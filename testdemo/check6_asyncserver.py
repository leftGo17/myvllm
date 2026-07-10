import asyncio
import os
import shutil

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

model_name = os.path.expanduser("~/hf/Qwen3-0.6B")
trace_dir = os.path.join(os.path.dirname(__file__), "trace")

# 每次跑之前清空旧的 trace，避免堆积历史文件。
shutil.rmtree(trace_dir, ignore_errors=True)
os.makedirs(trace_dir, exist_ok=True)

# AsyncLLM: 常驻引擎，vllm serve（OpenAI 兼容 server）内部用的就是这个类。
# 跟 LLM 不一样：这里引擎起来一次之后就在后台一直跑，
# 请求可以在任意时间点通过 asyncio 并发提交，不需要等上一批全部处理完。
engine_args = AsyncEngineArgs(
    model=model_name,
    quantization="fp8",
    profiler_config={"torch_profiler_dir": trace_dir},
)
engine = AsyncLLM.from_engine_args(engine_args)

tokenizer = engine.get_tokenizer()


def to_prompt(content: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


sampling_params = SamplingParams(temperature=0, max_tokens=5)


async def run_request(request_id: str, content: str, arrive_at: float) -> None:
    await asyncio.sleep(arrive_at)
    print(f"[t={arrive_at:.2f}s] 请求 {request_id} 到达，开始提交")
    final_output = None
    # generate() 是一个 async generator：每生成一步新 token 就 yield 一次当前的
    # RequestOutput（流式），这里只是简单地一直迭代到最后一个（完整结果）。
    async for output in engine.generate(to_prompt(content), sampling_params, request_id=request_id):
        final_output = output
    print(f"[完成] 请求 {request_id}: {final_output.outputs[0].text[:60]!r}...")


async def main() -> None:
    # 顺序执行：req-A 完全跑完（拿到最终结果）之后，才提交 req-B。
    # 跟 llm.chat() 调两次的关键区别是：引擎全程没有重启，
    # CUDA graph、torch.compile 缓存、KV cache 分配都还在，
    # req-B 是"引擎已经热了、但此刻系统里没有别的请求在跑"时进来的新请求，
    # 而不是像并发那种"req-A 还在 decode 时 req-B 插进来混批"的情况。
    await engine.start_profile()
    await run_request("req-A", "introduce yourself", arrive_at=0.0)
    await run_request("req-B", "list all prime numbers within 100", arrive_at=0.0)
    await engine.stop_profile()
    engine.shutdown()


asyncio.run(main())
# 显式清空引用，让垃圾回收在这里就发生，避免解释器退出阶段
# 再触发一次 AsyncLLM.__del__ -> shutdown()（这时模块全局变量可能已被清空，报错但无害）。
engine = None

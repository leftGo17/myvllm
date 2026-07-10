import torch.nn as nn
import torch
import torch.distributed as dist


# 基础类，按照传入input_size和output_size 在hbm中初始化一块内存
class LinearBase(nn.Module):
    """
    A base class for linear layers.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # set tp_dim, tp_rank, tp_world_size for tensor parallelism
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # create weight parameter with custom weight loader
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # create bias parameter
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")


"""
these functions are for is that we deploy a maybe randomly initialized model on GPU using some tensor/pipeline parallel method
then we wanna load a saved model checkpoint to it

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""


# 直接拷贝内存中参数到hbm里面
# the simpliest Linear layer: ReplicatedLinear(LinearBase)
# where we simply copy the weight as the weight_loader
# and run the forward as a normal linear layer
class ReplicatedLinear(LinearBase):
    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


# 列并行，激活值每个GPU中都有一份完整，weight被切分到每个GPU中，按照输出维度切分
# 也就是原本假设总的(5, 10) x (10, 20) = (5, 20)
# 每个GPU都有一份(10, 10)， 算出来(5, 10)
# 一般情况下这时候最后再all_reduce得到(5, 20)
# 但是显然还有后续操作，算完这个W之后，就是算attention，attention本来就是分head，不是整个一起算attn
# 所以这里不需要all_reduce，直接把(5, 10)传给后续的attention就行了
# attn完成之后需要统计结果，就相当于做一次output，就要做一次all_reduce

# 这个类输入输出size依然是全部大小，内部做了切分，然后再去调用基类，也就是这个类要循环调用
# 才能得到完整weigth加载


# 只有这个类需要管切分
# columnsplit Linear layer: ColumnParallelLinear(LinearBase)
# get the original full parameter
# compute the starting index of the column split
# compute the dim size of the full parameter
# copy the parameter slice to the local parameter
class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert (
            output_size % tp_size == 0
        ), "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size // tp_size, bias, tp_dim=0)

    # param: parameter after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        # full_dim on the output column
        full_data_output_size = loaded_weights.size(0)
        # dim size after sharding
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(
            0
        ), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


# 这个类自己不需要管切分，但是加载时候
# an extension of ColumnParallelLinear by merging several matrices
class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[
            int
        ],  # e.g. merge QKV matrices to compute MM together and then split
        bias: bool = True,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    # param: parameter to be reloaded after tensor parallelism
    # loaded_weights: the original full parameter to be loaded into param
    # the index of merged matrices (e.g. it's 0 for Q, 1 for K, 2 for V assuming QKV are merged together)
    def weight_loader(
        self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int
    ):
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        load to
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        ) which is also sharded by tp_size
        """
        param_data = param.data
        # compute offset
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        # compute size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size
        # find the correct slice to be loaded in the sharded parameter
        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # Calculate per-GPU output size
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # Pass TOTAL output size to parent (it will divide by tp_size)
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    # load_weight_id: q, k, v
    def weight_loader(
        self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str
    ):
        # batch_size * num_heads * num_token * head_size
        param_data = param.data
        # loaded_weights: batch_size * num_token * (head_size*num_heads)
        assert load_weight_id in [
            "q",
            "k",
            "v",
        ], "load_weight_id must be one of 'q', 'k', 'v'"
        # compute offset
        if load_weight_id == "q":
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == "k":
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == "v":
            offset = (
                self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            )
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        param_data = param_data.narrow(0, offset, shard_size)
        # shard the original full weight
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert (
            input_size % tp_size == 0
        ), "Input size must be divisible by tensor parallel size."
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        param_data = param.data
        # full_dim on the input row
        full_data_input_size = loaded_weights.size(1)
        # dim size after sharding
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(
            1
        ), "Shard size does not match parameter size."
        # starting index
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result


if __name__ == "__main__":
    # Example usage
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    layer = LinearBase(input_size=10, output_size=5)
    print("LinearBase layer initialized:", layer)

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import torch_npu
import sys
import json
import copy
import time

torch.npu.config.allow_internal_format = True

import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import Parameter

import torchair as tng
tng.patch_for_hcom()
import torchair._contrib.custom_torch_ops
sys.path.append("..")
from models.global_setting import _TODO_REQUIRE_API, PREFETCH_SIZE, FFN1_PREFETCH_SIZE, FFN2_PREFETCH_SIZE
from models.common import SuperKernelScope, NpuStreamSwitch, NpuLimitCoreNum, model_dtype

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)
local_rank_id = int(os.getenv("LOCAL_RANK", "1"))
device = torch.device("%s:%s" % ("npu", local_rank_id))
torch.npu.set_device(device)

class NpuMoERouterA8W8(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.rank_id = int(os.getenv("RANK_ID", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))

        self.dynamo = kwargs.get("dynamo", False)
        self.enable_print = kwargs.get("enable_print", False)
        self.enable_weight_nz = kwargs.get("enable_weight_nz", False)
        self.hidden_size = kwargs.get("hidden_size", 7168)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", 2048)
        self.experts_per_die = kwargs.get("experts_per_die", 1)
        self.enable_stream = self.dynamo

        print(f"[NpuMoERouterA8W8]rank:{self.rank_id} local_rank:{self.local_rank} "
              f"dynamo:{self.dynamo} enable_print:{self.enable_print} enable_weight_nz:{self.enable_weight_nz} "
              f"hidden_size:{self.hidden_size} moe_intermediate_size:{self.moe_intermediate_size} "
              f"experts_per_die:{self.experts_per_die}", flush=True) if self.enable_print else None

        self.up_weight = nn.Parameter(torch.ones((self.experts_per_die, 2*self.moe_intermediate_size, self.hidden_size), dtype=torch.int8, device=self.device), 
                                    requires_grad=False) # E, out, in
        self.down_weight = nn.Parameter(torch.ones((self.experts_per_die, self.hidden_size, self.moe_intermediate_size), dtype=torch.int8, device=self.device), 
                                    requires_grad=False) # E, out, in
        if self.enable_weight_nz:
            self.up_weight.data = self.up_weight.data.transpose(1, 2).contiguous()
            self.up_weight.data = torch_npu.npu_format_cast(self.up_weight.data, 29)
            self.down_weight.data = self.down_weight.data.transpose(1, 2).contiguous()
            self.down_weight.data = torch_npu.npu_format_cast(self.down_weight.data, 29)
        
        scale_1 = torch.ones(self.hidden_size, dtype=model_dtype)
        self.in_scale_1 = nn.Parameter(scale_1, requires_grad=False)
        scale_type = torch.float32
        self.out_scale_1 = nn.Parameter(torch.ones(size=(self.experts_per_die, 2*self.moe_intermediate_size),\
                                    dtype=scale_type), requires_grad=False)

        scale_2 = torch.ones(self.experts_per_die, self.moe_intermediate_size, dtype=model_dtype)
        self.in_scale_2 = nn.Parameter(scale_2, requires_grad=False)
        self.out_scale_2 = nn.Parameter(torch.ones(size=(self.experts_per_die, self.hidden_size),\
                                    dtype=model_dtype), requires_grad=False)
        
    def forward(self, x, expert_tokens, is_quant=True, dynamic_scale=None, avg_tokens_per_expert=None):
        if is_quant:
            h = x
            pertoken_scale = dynamic_scale
        else:
            h, pertoken_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=self.in_scale_1)

        if len(pertoken_scale.size()) > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            h = h.view(-1, hidden_size)

        weight = self.up_weight
        group_list_type = 2 if len(expert_tokens.shape) == 2 else 1
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [weight], bias=None, group_list=expert_tokens, 
                output_dtype=torch.int32, group_type=0, 
                split_item=3, group_list_type=group_list_type, act_type=0, tuning_config=avg_tokens_per_expert)[0]
        with NpuStreamSwitch(self.enable_stream, '33'):
            intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                mm1_mm3, weight_scale=self.out_scale_1,
                # activation_scale=pertoken_scale.squeeze(0), # 主线和rp2使用activation_scale, bbit分支使用activate_scale
                activate_scale=pertoken_scale.squeeze(0),
                bias=None, quant_scale=self.in_scale_2, quant_offset=None,
                group_index=expert_tokens, activate_left=False, quant_mode=1) # eager和dynamo已归一

        print(f"gmm1 result: {pertoken_scale} {pertoken_scale.shape}", flush=True) if not self.dynamo and self.enable_print else None
        weight = self.down_weight
        out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [weight], bias=None, group_list=expert_tokens, scale=[self.out_scale_2], offset=None,
                        per_token_scale=[pertoken_scale], output_dtype=torch.bfloat16, group_type=0, 
                        split_item=3, group_list_type=group_list_type, act_type=0, tuning_config=avg_tokens_per_expert)[0]
        print(f"gmm2 result: {out_hidden} {out_hidden.shape}", flush=True) if not self.dynamo and self.enable_print else None

        return out_hidden

class FFNModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.enable_prefetch = kwargs.get("enable_prefetch", False)
        self.expert_layers = kwargs.get("expert_layers", 58)
        print(f"enable_prefetch:{self.enable_prefetch} expert_layers:{self.expert_layers}", flush=True)

        self.layers = nn.ModuleList([NpuMoERouterA8W8(**kwargs) for _ in range(self.expert_layers)])

        Cin = 7168
        Cout = 7168
        self.dumy_nn = torch.nn.Linear(Cin, Cout, dtype=torch.bfloat16)

    def forward(self, x, expert_tokens, is_quant=False, dynamic_scale=None, avg_tokens_per_expert=None):
        """
        将专家按层展开成一个大图
        """
        for layer_idx in range(self.expert_layers):
            moe_layer = self.layers[layer_idx]
            if self.enable_prefetch:
                torch_npu.npu_prefetch(moe_layer.up_weight.data, x, PREFETCH_SIZE, 0)
                torch_npu.npu_prefetch(moe_layer.down_weight.data, x, PREFETCH_SIZE, 0)

            dynamic_scale = dynamic_scale * 0.1
            for batch_id in range(3):
                x = moe_layer(x, expert_tokens, is_quant, dynamic_scale, avg_tokens_per_expert)
                x = x.to(torch.int8)

        return x

def compile(model):
    import torchair as tng
    torch._logging.set_logs(recompiles=True)
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    dynamic = False
    model = torch.compile(model, dynamic=dynamic, fullgraph=True, backend=npu_backend)
    logging.info("in dynamo mode, dynamic=%s, fullgraph=%s, backend=npu" % (dynamic, True))
    logging.info("begin to compile...")

    return model

def define_profiling(profile_switch=False, profile_save_path="log"):
    if profile_switch:
        os.makedirs(profile_save_path, exist_ok=True)
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)

        profiler = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.NPU,
                    torch_npu.profiler.ProfilerActivity.CPU],
                with_stack=False,
                record_shapes=False,
                profile_memory=False,
                experimental_config=experimental_config,
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=4, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path))
    else:
        profiler = InferenceContextManager()
    return profiler

def test_ffn():
    num_per_layer = 1
    layer_num = 1
    expert_layers = 3
    experts_per_die = layer_num*num_per_layer
    enable_weight_nz = True
    dynamo = True
    enable_prefetch = True
    kwargs = {
        "experts_per_die": experts_per_die,
        "enable_weight_nz": enable_weight_nz,
        "dynamo": dynamo,
        "enable_print": False,
        "enable_prefetch": enable_prefetch,
        "expert_layers": expert_layers,
    }
    
    model = FFNModel(**kwargs)
    model.to(device)
    if enable_weight_nz:
        for i in range(expert_layers):
            model.layers[i].up_weight.data = torch_npu.npu_format_cast(model.layers[i].up_weight.data, 29)
            model.layers[i].down_weight.data = torch_npu.npu_format_cast(model.layers[i].down_weight.data, 29)

    if dynamo:
        model = compile(model)

    attn_die = 144
    tok = 8
    mbs = 30 * 2
    batch_size = attn_die * mbs * (tok + 1)
    x = torch.randint(low=0, high=2, size=(batch_size, 7168), dtype=torch.int8, device=device)
    dynamic_scale = torch.rand((batch_size,), dtype=torch.float32, device=device)
    # avg_token = attn_die * mbs * tok // 256
    avg_token = attn_die * mbs  // 32
    expert_token_nums = torch.zeros([experts_per_die, 2], dtype=torch.int64, device=device)
    for i in range(num_per_layer):
        expert_token_nums[i][0] = i
        expert_token_nums[i][1] = avg_token
    print(f"expert_token_nums:{expert_token_nums}", flush=True)
    avg_tokens_per_expert = [mbs]

    print(f"x:{x.shape} {x.dtype} dynamic_scale:{dynamic_scale.shape} {dynamic_scale.dtype}", flush=True)

    profile_save_path = f"prof"
    profiler = define_profiling(True, profile_save_path)
    with profiler as prof:
        for _ in range(10):
            with torch.no_grad():
                y = model(x, expert_token_nums, is_quant=True, dynamic_scale=dynamic_scale, avg_tokens_per_expert=avg_tokens_per_expert)
    print(f"moe router expert result:{y.shape}", flush=True)


if __name__ == "__main__":
    test_ffn()
    logging.info("model run success")
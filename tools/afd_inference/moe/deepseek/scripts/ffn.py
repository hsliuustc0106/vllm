# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
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

import llm_datadist.llm_wrapper as lw
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
import torchair as tng
tng.patch_for_hcom()
import torchair._contrib.custom_torch_ops
from models.global_setting import PREFETCH_SIZE
from models.common import SuperKernelScope, NpuStreamSwitch, NpuLimitCoreNum, model_dtype, ffn_mode, exe_mode, init_comm_group, fix_rand_seed, apply_quant
from models.local_window_utils import alloc_and_exchange_comm_window, get_local_window, call_ffn_win_size
from transformers.modeling_utils import PreTrainedModel

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)

micro_batch_number = int(os.getenv("MICRO_BATCH_NUMBER", "3")) if exe_mode == "dynamo" else 1

class NpuMoERouterA8W8(torch.nn.Module):
    def __init__(self, enable_stream, **kwargs) -> None:
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
        self.enable_stream = enable_stream and False

        logging.info(f"[NpuMoERouterA8W8]rank:{self.rank_id} local_rank:{self.local_rank} "
              f"dynamo:{self.dynamo} enable_print:{self.enable_print} enable_weight_nz:{self.enable_weight_nz} "
              f"hidden_size:{self.hidden_size} moe_intermediate_size:{self.moe_intermediate_size} "
              f"experts_per_die:{self.experts_per_die} enable_stream:{self.enable_stream}", flush=True) if self.enable_print else None

        epsilon = 1e-2
        fix_rand_seed()
        self.up_weight = nn.Parameter(torch.rand(self.experts_per_die, 2 * self.moe_intermediate_size, self.hidden_size), requires_grad=False)  # E, out, in
        fix_rand_seed()
        self.down_weight = nn.Parameter(torch.rand(self.experts_per_die, self.hidden_size, self.moe_intermediate_size), requires_grad=False)  # E, out, in
        self.up_weight.data = apply_quant(self.up_weight.data)
        self.down_weight.data = apply_quant(self.down_weight.data)
        if self.enable_weight_nz:
            self.up_weight.data = self.up_weight.data.transpose(1, 2).contiguous().to(self.device)
            self.up_weight.data = torch_npu.npu_format_cast(self.up_weight.data, 29)
            self.down_weight.data = self.down_weight.data.transpose(1, 2).contiguous().to(self.device)
            self.down_weight.data = torch_npu.npu_format_cast(self.down_weight.data, 29)

        fix_rand_seed()
        self.in_scale_1 = nn.Parameter(torch.rand(self.hidden_size, dtype=model_dtype) * (1 - epsilon) + epsilon, requires_grad=False)
        self.out_scale_1 = nn.Parameter(torch.rand(size=(self.experts_per_die, 2 * self.moe_intermediate_size), dtype=torch.float32) * (1 - epsilon) + epsilon, requires_grad=False)

        fix_rand_seed()
        self.in_scale_2 = nn.Parameter(torch.rand(self.experts_per_die, self.moe_intermediate_size, dtype=torch.float32) * (1 - epsilon) + epsilon, requires_grad=False)
        self.out_scale_2 = nn.Parameter(torch.rand(size=(self.experts_per_die, self.hidden_size), dtype=model_dtype) * (1 - epsilon) + epsilon, requires_grad=False)
        
    def forward(self, x, expert_tokens, is_quant=True, dynamic_scale=None, avg_tokens_per_expert=None):
        if is_quant:
            h = x
            pertoken_scale = dynamic_scale
        else:
            h, pertoken_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=self.in_scale_1)

        weight = self.up_weight
        group_list_type = 2 if len(expert_tokens.shape) == 2 else 1
        print(f"group_list_type:{group_list_type} expert_tokens:{expert_tokens.shape} {expert_tokens} "
              f"h.shape:{h.shape}",
              flush=True) if not self.dynamo else None
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [weight], bias=None, group_list=expert_tokens,
                output_dtype=torch.int32, group_type=0, 
                split_item=3, group_list_type=group_list_type, act_type=0, tuning_config=avg_tokens_per_expert)[0]
        with NpuStreamSwitch(self.enable_stream, '33'):
            intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                mm1_mm3, weight_scale=self.out_scale_1,
                activation_scale=pertoken_scale.squeeze(0), # 主线和rp2使用activation_scale, bbit分支使用activate_scale
                # activate_scale=pertoken_scale.squeeze(0),
                bias=None, quant_scale=self.in_scale_2, quant_offset=None,
                group_index=expert_tokens, activate_left=False, quant_mode=1) # eager和dynamo已归一

        print(f"gmm1 result: {pertoken_scale} {pertoken_scale.shape}", flush=True) if not self.dynamo and self.enable_print else None
        weight = self.down_weight
        out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [weight], bias=None, group_list=expert_tokens, scale=[self.out_scale_2], offset=None,
                        per_token_scale=[pertoken_scale], output_dtype=torch.bfloat16, group_type=0, 
                        split_item=3, group_list_type=group_list_type, act_type=0, tuning_config=avg_tokens_per_expert)[0]
        print(f"gmm2 result: {out_hidden} {out_hidden.shape}", flush=True) if not self.dynamo and self.enable_print else None

        return out_hidden

class NpuMoELayer(torch.nn.Module):
    def __init__(self, layer_idx, **kwargs) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.rank_id = int(os.getenv("RANK_ID", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))

        self.dynamo = kwargs.get("dynamo", False)
        self.enable_print = kwargs.get("enable_print", False)
        self.hidden_size = kwargs.get("hidden_size", 7168)
        self.enable_gmm_tune_config = kwargs.get("enable_gmm_tune_config", 0)
        self.batch_size = kwargs.get("batch_size", 1)
        self.shared_expert_rank_num = kwargs.get("shared_expert_rank_num", 1)
        self.pg_group_name = kwargs.get("pg_group_name", None)
        self.on_cloud = kwargs.get("on_cloud", 0)
        self.world_size = kwargs.get("world_size", 16)
        self.top_k = kwargs.get("top_k", 8)
        self.ffn_dies = int(os.getenv("FFN_DIES", "12"))
        self.attn_dies = int(os.getenv("ATTN_DIES", "4"))
        self.context_holder = kwargs.get("context_holder", None)
        self.schedule_context = kwargs.get("schedule_context", None)
        self.experts_per_die = kwargs.get("experts_per_die", 1)
        self.gmm_quant_mode = kwargs.get("gmm_quant_mode", 2) # 0: 非量化；1: 静态量化；2：动态量化
        self.avg_tokens_per_expert = kwargs.get("avg_tokens_per_expert", None)
        self.enable_superkernel = kwargs.get("enable_superkernel", False)
        self.batch_with_recv = kwargs.get("batch_with_recv", 0)
        self.f2a_event = kwargs.get("f2a_event", None)
        self.f2a_stream = kwargs.get("f2a_stream", None)
        self.use_real_actual_seq_len = int(os.getenv("USE_REAL_ACTUAL_SEQ_LEN", "0"))
        self.enable_stream = self.dynamo and False
        self.aiv_batch = 16
        self.aiv_f2a = 48 - self.aiv_batch
        self.enabel_split_core = False # True if self.enable_stream else False

        self.expert_func = NpuMoERouterA8W8(self.enable_stream, **kwargs)
        if self.enable_gmm_tune_config == 0:
            self.avg_tokens_per_expert = None
        print(f"avg_tokens_per_expert:{self.avg_tokens_per_expert}", flush=True)

        layer_out = os.getenv("LAYER_OUT", "FA")
        attn_die_offset = self.ffn_dies if layer_out.upper() == "FA" else 0
        self.attn_rank_table = torch.Tensor([i + attn_die_offset for i in range(self.attn_dies)]).to(torch.int32).npu()
        # self.max_out_shape = [self.attn_dies * self.batch_size * (self.top_k + 1), self.hidden_size] #{Y, H} Y=A*BS*Loc
        self.max_out_shape = [self.attn_dies , self.batch_size, self.top_k + 1, self.hidden_size] #{Y, H} Y=A*BS*Loc
        print(f"self.attn_dies:{self.attn_dies} batch_size:{self.batch_size} self.top_k:{self.top_k}"
              f" hidden_size:{self.hidden_size}")# al_E

    def _send_back_to_attn(self, hidden_states, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num):
        # 调用ffn_to_attention
        to_attn_kwargs = {
          "x": hidden_states,    # [Y,H]
          "session_ids": session_ids,    # [Y]
          "micro_batch_ids": micro_batch_ids,    # [Y]
          "token_ids": token_ids,  # [Y]
          "expert_offsets": expert_offsets, # [Y]
          "actual_token_num": actual_token_num, # int64 Tensor [1]
          "attn_rank_table": self.attn_rank_table, # [A]
          "group": self.pg_group_name,
          "world_size": self.world_size,
          "token_info_table_shape": [micro_batch_number, self.batch_size, self.top_k + 1],    # [M, BS, K+1]
          "token_data_shape": [micro_batch_number, self.batch_size, self.top_k + 1, self.hidden_size]        # [M, BS, K+1, HS]
        }
        if not self.dynamo and self.enable_print:
            print(f"npu_ffn_to_attention actual_token_num:{actual_token_num} session_ids:{session_ids}", flush=True)
            print(f"npu_ffn_to_attention micro_batch_ids:{micro_batch_ids}", flush=True)
            print(f"npu_ffn_to_attention token_ids:{token_ids}", flush=True)
            print(f"npu_ffn_to_attention expert_offsets:{expert_offsets}", flush=True)
            print(f"npu_ffn_to_attention attn_rank_table:{self.attn_rank_table}", flush=True)
            print(f"npu_ffn_to_attention pg_group_name:{self.pg_group_name} world_size:{self.world_size}", flush=True)
        torch_npu.npu_ffn_to_attention(**to_attn_kwargs)
        print(f"npu_ffn_to_attention success, micro batch:{micro_batch_ids}", flush=True) if not self.dynamo and self.enable_print else None

    def forward(self, x, batch_id, local_all_gather_out, is_quant=True):
        if not self.batch_with_recv:
            print("ffn_worker_scheduler begin.", flush=True) if not self.dynamo and self.enable_print else None
            torch_npu.ffn_worker_scheduler_(self.schedule_context, sync_group_size=1 if self.use_real_actual_seq_len else self.attn_dies)
            print("ffn_worker_scheduler submit end.", flush=True) if not self.dynamo and self.enable_print else None
            torch.npu.synchronize() if not self.dynamo and self.enable_print else None
            print("ffn_worker_scheduler finish.", flush=True) if not self.dynamo and self.enable_print else None

        # 调用FFNWorkerBatching
        batching_kwargs = {
            "schedule_context": self.schedule_context,
            "expert_num": self.experts_per_die,
            "max_out_shape": self.max_out_shape,
            "token_dtype": 2 if self.gmm_quant_mode == 2 else 1 if model_dtype == torch.bfloat16 else 0,
            "need_schedule": 1 if self.batch_with_recv else 0
        }
        print(f"begin npu_ffn_worker_batching...", flush=True) if not self.dynamo and self.enable_print else None

        if local_all_gather_out is not None and self.dynamo:
            tng.scope.npu_wait_tensor(batching_kwargs.get("schedule_context"), local_all_gather_out)

        # TODO:BATCHING当前不支持SK
        with NpuLimitCoreNum(self.enabel_split_core, 24, self.aiv_batch):
            with SuperKernelScope(self.enable_superkernel, f'moe', "option_xxx"):
                hidden_states, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num \
                    = torch_npu.npu_ffn_worker_batching(**batching_kwargs)
        # if not self.dynamo and self.enable_print:
        if not self.dynamo:
            torch.npu.synchronize()
            print(f"[ffn_worker_batching] hidden_states:{hidden_states.shape} {hidden_states.dtype} {hidden_states}", flush=True)
            print(f"[ffn_worker_batching] actual_token_num:{actual_token_num} group_list:{group_list} ", flush=True)
            print(f"[ffn_worker_batching] dynamic_scale:{dynamic_scale} ", flush=True)
            print(f"[ffn_worker_batching] session_ids:{session_ids} ", flush=True)
            print(f"[ffn_worker_batching] micro_batch_ids:{micro_batch_ids} ", flush=True)
            print(f"[ffn_worker_batching] token_ids:{token_ids} ", flush=True)
            print(f"[ffn_worker_batching] expert_offsets:{expert_offsets} ", flush=True)

        kwargs = {
            "x": hidden_states,
            "expert_tokens": group_list,
            "is_quant": True,
            "dynamic_scale": dynamic_scale,
            "avg_tokens_per_expert": self.avg_tokens_per_expert,
        }
        print(f"[rank:{self.rank_id} moe expert input x: {kwargs}", flush=True) if not self.dynamo and self.enable_print else None
        with SuperKernelScope(self.enable_superkernel, f'moe', "option_xxx"):
            out_hidden = self.expert_func(**kwargs)
        if self.dynamo:
            with NpuStreamSwitch(self.enable_stream, '33'):
                with NpuLimitCoreNum(self.enabel_split_core, 24, self.aiv_f2a):
                    with SuperKernelScope(self.enable_superkernel, f'moe', "option_xxx"):
                        self._send_back_to_attn(out_hidden, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num)
        else:
            self.f2a_event.record()
            with torch.npu.stream(self.f2a_stream):
                self.f2a_event.wait()
                self._send_back_to_attn(out_hidden, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num)
        return out_hidden


class FFNModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.enable_prefetch = kwargs.get("enable_prefetch", False)
        self.expert_layers = kwargs.get("expert_layers", 58)
        self.global_world_size = kwargs.get("world_size", 16)
        self.spec_len = kwargs.get("spec_len", 1)
        self.start_sync = kwargs.get("start_sync", False)

        self.layers = nn.ModuleList([NpuMoELayer(layer_idx, **kwargs) for layer_idx in range(self.expert_layers)])
        print("FFNModel finished")

    def forward(self, x):
        """
        将专家按层展开成一个大图
        """
        for layer_idx in range(self.expert_layers):
            # 临时方案，用于全局强制同步
            local_all_gather_out = None
            if self.start_sync:
                if (layer_idx == 0 and self.spec_len == 1) or (layer_idx == 1 and self.spec_len == 2):
                    local_tensor = torch.ones([1], dtype=torch.int32).npu()
                    local_all_gather_out = torch.zeros([self.global_world_size], dtype=torch.int32).npu()
                    dist.all_gather_into_tensor(local_all_gather_out, local_tensor)

            moe_layer = self.layers[layer_idx]
            if self.enable_prefetch:
                torch_npu.npu_prefetch(moe_layer.expert_func.up_weight.data, x, PREFETCH_SIZE, 0)
                torch_npu.npu_prefetch(moe_layer.expert_func.down_weight.data, x, PREFETCH_SIZE, 0)
            for batch_id in range(micro_batch_number):
                x = moe_layer(x, batch_id, local_all_gather_out)

        return x

class FFNForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.rank_id = int(os.getenv("RANK_ID", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.f2a_event = torch.npu.Event()
        self.f2a_stream = torch.npu.Stream()

        self.attn_dies = int(os.getenv("ATTN_DIES", "4"))
        self.next_n = int(os.getenv("NEXT_N", "0")) if not hasattr(config, "next_n") else config.next_n
        self.spec_len = self.next_n + 1
        self.batch_size = int(os.getenv("BATCH_SIZE", "4")) // self.attn_dies // micro_batch_number * self.spec_len
        self.ffn_dies = int(os.getenv("FFN_DIES", "12"))
        self.world_size = int(os.getenv("WORLD_SIZE", "16"))
        expert_layers = config.num_hidden_layers - config.first_k_dense_replace # 稠密层数为3
        self.expert_layers = expert_layers + self.next_n if self.next_n != 0 else expert_layers # 考虑开MTP时的稀疏层
        self.hidden_size = config.hidden_size
        self.shared_expert_rank_num = int(os.getenv("EXPERTS_SHARE_NUM_COPY", "1"))
        self.experts_per_layer = int(os.getenv("N_ROUTED_EXPERTS_PER_RANK", "1")) # 在每个层Die上部署的路由专家数
        print(f"FFNForCausalLM shared_expert_rank_num:{self.shared_expert_rank_num} "
              f"experts_per_layer:{self.experts_per_layer}")
        self.enable_gmm_tune_config = int(os.getenv("ENABLE_GMM_TUNE_CONFIG", "0"))
        self.enable_prefetch = int(os.getenv("ENABLE_PREFETCH", "0"))
        self.batch_with_recv = int(os.getenv("ENABLE_BATCH_WITH_RECV", "0"))
        self.top_k = config.num_experts_per_tok
        self.gmm_quant_mode = 2 # 0: FP16 1: BF16 2: dynamic_quant_int8
        self.on_cloud = int(os.getenv("ON_CLOUD", "0"))
        self.dynamo = True if ffn_mode == "dynamo" else False
        self.enable_weight_nz = True
        self.enable_print = False
        self.enable_cache_compile = False #int(os.getenv("ENABLE_CACHE_COMPILE", "0")) and ffn_mode =="dynamo"
        self.enable_superkernel = int(os.getenv("ENABLE_SUPERKERNEL", "1")) and self.dynamo # eager模式不支持superkernel
        self.ffn_need_wait = int(os.getenv("FFN_NEED_WAIT", "0"))
        self.start_sync = int(os.getenv("ATTN_FFN_START_SYNC", "0"))
        self.remainder_router_expert = int(os.getenv("REMAINDER_ROUTER_EXPERT", "0"))
        self.ffn_die_for_remaind_expert = 1 if self.remainder_router_expert > 0 else 0
        layer_out = os.getenv("LAYER_OUT", "FA")
        ffn_die_offset = 0 if layer_out.upper() == "FA" else self.attn_dies
        if self.rank_id < self.shared_expert_rank_num + ffn_die_offset:
            self.experts_per_die = 1 * 1 # 共享专家平铺
            self.avg_tokens_per_expert = [self.batch_size * self.attn_dies // self.shared_expert_rank_num]
            print(f"share expert experts_per_die:{self.experts_per_die} self.batch_size:{self.batch_size} "
                  f"self.attn_dies:{self.attn_dies} "
                  f"self.avg_tokens_per_expert:{self.avg_tokens_per_expert}", flush=True)
        elif self.rank_id == self.shared_expert_rank_num + ffn_die_offset and self.remainder_router_expert > 0:
            self.experts_per_die = 1 * self.remainder_router_expert
            n_routed_experts = config.n_routed_experts if self.on_cloud else 8
            self.avg_tokens_per_expert = [self.batch_size *self.top_k*self.attn_dies // n_routed_experts]
            print(f"remaind router expert", flush=True)
        else:
            self.experts_per_die = 1 * self.experts_per_layer
            n_routed_experts = config.n_routed_experts if self.on_cloud else 8
            self.avg_tokens_per_expert = [self.batch_size*self.top_k*self.attn_dies // n_routed_experts]
            print(f"router expert n_routed_experts:{n_routed_experts} "
                  f"self.batch_size:{self.batch_size} "
                  f"self.attn_dies:{self.attn_dies}"
                  f"self.avg_tokens_per_expert:{self.avg_tokens_per_expert}", flush=True)

        "创建D/C算子通信域"
        print("ffn begin to init moe_all_to_all_group", flush=True)
        moe_pg_group = init_comm_group(
            global_rank=self.rank_id,
            group_num=1,
            world_size=self.world_size,
            group_stride=1,
            group_name="moe_all_to_all_group_name",
        )
        self.pg_group_name = moe_pg_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id)
        print("ffn success to init moe_all_to_all_group", flush=True)

        ffn_win_size = call_ffn_win_size(session_num=self.attn_dies, micro_batch_num=micro_batch_number,
            micro_batch_size=self.batch_size, selected_expert_num=config.num_experts_per_tok+1,
            hidden_size=config.hidden_size, quant_mode=self.gmm_quant_mode)
        layer_out = os.getenv("LAYER_OUT", "FA")
        peer_offset = self.ffn_dies if layer_out.upper() == "FA" else 0
        peer_ranks = [peer_offset + i for i in range(self.attn_dies)]
        alloc_and_exchange_comm_window(peer_ranks=peer_ranks, win_size=ffn_win_size, group=moe_pg_group)
        ffn_window, ffn_window_size = get_local_window()
        print(f"get local window success, ffn_window={ffn_window}, ffn_window_size={ffn_window_size}", flush=True)

        kwargs = self.update_kwargs()
        
        normal_ffn_dies = self.ffn_dies - self.shared_expert_rank_num - self.ffn_die_for_remaind_expert # 去除共享专家和冗余路由专家部署的die
        expert_num = normal_ffn_dies * self.experts_per_layer + self.shared_expert_rank_num + self.remainder_router_expert
        if self.gmm_quant_mode == 2:
            attn_to_ffn_token_size = (config.hidden_size + 4 + 511) // 512 * 512  # 512:为了对齐 4: 量化scale fp32
        else:
            attn_to_ffn_token_size = config.hidden_size * 2
        ffn_to_attn_token_size = config.hidden_size * 2
        context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode=0, session_num=self.attn_dies,
            micro_batch_num=micro_batch_number, micro_batch_size=self.batch_size, selected_expert_num=config.num_experts_per_tok+1,
            expert_num=expert_num, attn_to_ffn_token_size=attn_to_ffn_token_size,
            ffn_to_attn_token_size=ffn_to_attn_token_size, ffn_window=ffn_window, ffn_window_size=ffn_window_size)
        print(f"=========context_holder:{context_holder.get_schedule_context_info()}", flush=True)
        schedule_context = context_holder.get_schedule_context_tensor()

        kwargs.update({"schedule_context": schedule_context,
                       "context_holder": context_holder,})

        local_expert_table = torch.full((1, self.experts_per_layer), -1, dtype=torch.int32).npu()
        if self.rank_id < self.shared_expert_rank_num + ffn_die_offset:
            for layer_id in range(1):
                local_expert_table[layer_id][0] = normal_ffn_dies * self.experts_per_layer + self.remainder_router_expert
        elif self.rank_id == self.shared_expert_rank_num + ffn_die_offset and self.remainder_router_expert > 0:
            for layer_id in range(1):
                for i in range(self.remainder_router_expert):
                    local_expert_table[layer_id][i] = normal_ffn_dies * self.experts_per_layer + i
        else:
            for layer_id in range(1):
                for i in range(self.experts_per_layer):
                    local_expert_table[layer_id][i] = (self.rank_id - self.shared_expert_rank_num - self.ffn_die_for_remaind_expert - ffn_die_offset) * self.experts_per_layer + i
        local_expert_table_all_gather_out = torch.zeros([self.world_size, 1, self.experts_per_layer], dtype=torch.int32).npu()
        dist.all_gather_into_tensor(local_expert_table_all_gather_out, local_expert_table.unsqueeze(0), group=_get_default_group())
        print(f"local_expert_table:{local_expert_table} {local_expert_table_all_gather_out}", flush=True)

        # attn_die_offset = self.ffn_dies if layer_out.upper() == "FA" else 0
        # self.attn_rank_table = torch.Tensor([i + attn_die_offset for i in range(self.attn_dies)]).to(torch.int32).npu()

        self.model = FFNModel(config, **kwargs)

        if self.enable_cache_compile:
            import torchair as tng
            logging.info("enable_cache_compile...")
            tng_config = tng.CompilerConfig()
            tng_config.experimental_config.frozen_parameter = True
            tng_config.experimental_config.topology_sorting_strategy = "StableRDFS"
            para_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(f"{para_dir}/models/", os.getenv("CASE_NAME", "./"))
            self.cached_model = tng.inference.cache_compile(self.ffn, config=tng_config, cache_dir=cache_dir, ge_cache=True)

        # if ((self.ffn_need_wait and self.on_cloud) or self.expert_layers > 30) and exe_mode == "dynamo":
        #     logging.info("begin sleep 1200")
        #     time.sleep(600)  # 临时规避方案，避免attn编译过慢导致ffn侧同步超时
        #     logging.info("end sleep 1200")

    def update_kwargs(self):
        kwargs = {
                    "batch_size": self.batch_size,
                    "world_size": self.world_size,
                    "expert_layers": self.expert_layers,
                    "shared_expert_rank_num": self.shared_expert_rank_num,
                    "experts_per_die": self.experts_per_die,
                    "enable_gmm_tune_config": self.enable_gmm_tune_config,
                    "avg_tokens_per_expert": self.avg_tokens_per_expert,
                    "enable_weight_nz": self.enable_weight_nz,
                    "dynamo": self.dynamo,
                    "enable_superkernel": self.enable_superkernel,
                    "start_sync": self.start_sync,
                    "enable_cache_compile": self.enable_cache_compile,
                    "hidden_size": self.hidden_size,
                    "pg_group_name": self.pg_group_name, 
                    "enable_prefetch": self.enable_prefetch,
                    "top_k": self.top_k,
                    "gmm_quant_mode": self.gmm_quant_mode,
                    "enable_print": self.enable_print,
                    "on_cloud": self.on_cloud,
                    "batch_with_recv": self.batch_with_recv,
                    "spec_len": self.spec_len,
                    "f2a_event": self.f2a_event,
                    "f2a_stream": self.f2a_stream,
                }
        return kwargs    
    
    def ffn_to_attn(self, hidden_states, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num):
        to_attn_kwargs = {
          "x": hidden_states,    # [Y,H]
          "session_ids": session_ids,    # [Y]
          "micro_batch_ids": micro_batch_ids,    # [Y]
          "token_ids": token_ids,  # [Y]
          "expert_offsets": expert_offsets, # [Y]
          "actual_token_num": actual_token_num, # int64 Tensor [1]
          "attn_rank_table": self.attn_rank_table, # [A]
          "group": self.pg_group_name,
          "world_size": self.world_size,
          "token_info_table_shape": [micro_batch_number, self.batch_size, self.top_k + 1],    # [M, BS, K+1]
          "token_data_shape": [micro_batch_number, self.batch_size, self.top_k + 1, self.hidden_size]        # [M, BS, K+1, HS]
        }
        if not self.dynamo and self.enable_print:
            print(f"npu_ffn_to_attention actual_token_num:{actual_token_num} session_ids:{session_ids}", flush=True)
            print(f"npu_ffn_to_attention micro_batch_ids:{micro_batch_ids}", flush=True)
            print(f"npu_ffn_to_attention token_ids:{token_ids}", flush=True)
            print(f"npu_ffn_to_attention expert_offsets:{expert_offsets}", flush=True)
            print(f"npu_ffn_to_attention attn_rank_table:{self.attn_rank_table}", flush=True)
            print(f"npu_ffn_to_attention pg_group_name:{self.pg_group_name} world_size:{self.world_size}", flush=True)
        torch_npu.npu_ffn_to_attention(**to_attn_kwargs)
        print(f"npu_ffn_to_attention success, micro batch:{micro_batch_ids}", flush=True) if not self.dynamo and self.enable_print else None

    def ffn(self, x):
        return self.model(x)

    def forward(self, x):
        return self.cached_model(x) if self.enable_cache_compile else self.model(x)

# coding=utf-8
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch DeepSeek model."""
import os
import math
import copy
from typing import List, Optional, Tuple, Union
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

import torch_npu
import torchair as tng
tng.patch_for_hcom()
from torch.distributed.distributed_c10d import _world

from .configuration_deepseek import DeepseekV2Config
from .global_setting import _TODO_REQUIRE_API, _PAGE_ATTENTION_SETTING, PREFETCH_SIZE, FFN2_PREFETCH_SIZE, LM_HEAD_PREFETCH_SIZE
from .common import model_dtype, exe_mode, apply_quant, DeepseekV2RMSNorm, _init_rope, NpuStreamSwitch, SuperKernelScope, rope_single, NpuLinear, \
    DeepseekV2PreTrainedModel, DeepseekV2_START_DOCSTRING, init_comm_group, half_batch, one_third_batch, fix_rand_seed
from .local_window_utils import alloc_and_exchange_comm_window, get_local_window, call_attn_win_size

logger = logging.get_logger(__name__)

expert_rank_table = None  # 当前MTP场景通过全局变量规避重复allgather
moe_all_to_all_group = None
moe_all_to_all_group_name = None  # 当前MTP场景通过全局变量规避重复创建moe_group和获取hccl_comm_name
context_holder = None
schedule_context = None

micro_batch_number = int(os.getenv("MICRO_BATCH_NUMBER", "3")) if exe_mode == "dynamo" else 1


class DeepseekV2MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None, mlp_type="mlp", **kwargs):
        super().__init__()
        self.layer_flag = mlp_type
        self.world_size = kwargs.get("world_size")
        self.experts_tp_size = kwargs.get("experts_tp_size")
        self.die_num_per_node = kwargs.get("die_num_per_node")
        self.dense_tp_size = kwargs.get("dense_tp_size")
        self.enable_prefetch = kwargs.get("enable_prefetch")
        self.dynamic_quant_mode = kwargs.get("dynamic_quant_mode")
        self.enable_combine_dequant = kwargs.get("enable_combine_dequant", 0)
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = (
            config.intermediate_size if intermediate_size is None else intermediate_size
        )

        if self.layer_flag == "moe_share":
            self.intermediate_size_per_rank = self.intermediate_size // self.experts_tp_size
        else:  # dense mlp layer
            self.intermediate_size_per_rank = self.intermediate_size // self.dense_tp_size

        # gate_weight, up_weight in the order of swiglu_api
        self.merge_up_gate_proj = NpuLinear(self.hidden_size, self.intermediate_size_per_rank * 2, bias=False).to(model_dtype)
        self.down_proj = NpuLinear(self.intermediate_size_per_rank, self.hidden_size, bias=False).to(model_dtype)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, is_quant=False, dynamic_scale=None, allow_combine_dequant=True):
        if is_quant:
            merged_x, out_scale, pertoken_scale, out_shape = self.merge_up_gate_proj(x, is_quant, dynamic_scale, throw_dequant=True)
            group_index = torch.tensor([merged_x.shape[0]], dtype=torch.int64, device="npu")
            kwargs = {
                        "weight_scale": out_scale,
                        "bias": None,
                        "quant_scale": self.down_proj.in_scale,
                        "quant_offset": None,
                        "activation_scale": pertoken_scale,
                        "group_index": group_index,
                        "activate_left": False,
                        "quant_mode": 1
                    }
            intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(merged_x, **kwargs)
            if allow_combine_dequant and self.enable_combine_dequant:
                down_proj, out_scale, _, _ = self.down_proj(intermediate_h, is_quant, pertoken_scale, throw_dequant=True)
            else:
                down_proj = self.down_proj(intermediate_h, is_quant, pertoken_scale, throw_quant=True, out_shape=out_shape)
        else:
            merged_x = self.merge_up_gate_proj(x)
            intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
            down_proj = self.down_proj(intermediate_hidden_states)
            pertoken_scale = None
            out_scale = None
        return down_proj, pertoken_scale, out_scale


class MoEGate(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = 256 #config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.batch_size = kwargs.get("batch_size")
        self.enable_micro_batch = kwargs.get("enable_micro_batch")
        if self.enable_micro_batch:
            self.batch_size = self.batch_size // micro_batch_number
        self.input_len = kwargs.get("input_len")
        self.world_size = kwargs.get("world_size")
        self.global_rank = kwargs.get("global_rank")
        self.experts_tp_size = kwargs.get("experts_tp_size", 1)
        self.ep_size = kwargs.get("ep_size", 1)
        self.spec_len = kwargs.get("spec_len", 1)
        self.n_routed_experts_per_rank = kwargs.get("n_routed_experts_per_rank", 1)
        self.on_cloud = kwargs.get("on_cloud", 0)
        self.ffn_dies = int(os.getenv("FFN_DIES", "12"))
        self.attn_tp_size = int(os.getenv("ATTN_TP_SIZE", "4"))

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        fix_rand_seed()
        self.weight = nn.Parameter(
            torch.rand((self.n_routed_experts, config.hidden_size), dtype=torch.float32)
        )        
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.rand((self.n_routed_experts), dtype=torch.float32)
            )
        
        # uniform topk for each token
        # local_cur_topk_list = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7] for _ in range(self.batch_size * self.spec_len // self.world_size)]).int().npu()

        # step = self.batch_size // self.world_size * self.top_k * self.spec_len
        # cloud_cur_topk_list = [i % self.n_routed_experts for i in range(self.global_rank * step, (self.global_rank + 1) * step)]
        # cloud_cur_topk_list = torch.Tensor(cloud_cur_topk_list).int().view(self.batch_size * self.spec_len // self.world_size, -1).npu()
        # self.cur_topk_list = cloud_cur_topk_list if self.on_cloud else local_cur_topk_list

        # new enforce balance
        # local_cur_topk_list = torch.tensor([i % (self.n_routed_experts_per_rank * (self.ffn_dies - 1)) \
        self.experts_share_num_copy = kwargs.get("experts_share_num_copy")
        n_routed_experts = self.n_routed_experts_per_rank * (self.ffn_dies - self.experts_share_num_copy)
        print(f"n_routed_experts:{n_routed_experts}")
        local_cur_topk_list = torch.tensor([i % n_routed_experts \
                                            for i in range(self.batch_size * self.spec_len // self.world_size * self.top_k)]).int().npu()
        self.cur_topk_list = local_cur_topk_list.view(self.batch_size * self.spec_len // self.world_size, -1)
        print(f"MoeGate self.batch_size:{self.batch_size} self.enable_micro_batch:{self.enable_micro_batch} self.cur_topk_list:{self.cur_topk_list.shape} {self.cur_topk_list }", flush=True)

    def forward(self, hidden_states):
        return self.forward_gate(hidden_states)

    def forward_gate(self, hidden_states):
        bsz, seq_len, hidden_dim = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.to(torch.float).view(-1, hidden_dim)
        logits = F.linear(
            hidden_states, self.weight, None
        )
        topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
            logits, 
            k=self.top_k, # topk当前写8
            bias=self.e_score_correction_bias, 
            k_group=self.topk_group, # fix: 4 
            group_count=self.n_group, # fix 8
            group_select_mode=1, # 0: group中的最大; 1: topk2.sum(fix)
            renorm=0, # 0: softmax->topk(fix); 1: topk->softmax
            norm_type=1, # 0: softmax; 1: sigmoid(fix) 
            # out_flag=False, # 第三个输出是否输出
            routed_scaling_factor=self.routed_scaling_factor, 
            eps=float(1e-20))
        row_idx = None
        aux_loss = None

        topk_idx = self.cur_topk_list  # set uniform distributation of experts
        return topk_idx, topk_weight, aux_loss, row_idx

class npu_DeepseekV2MoE_ep(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, layer_idx, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_flag = "moe"
        self.config = config
        self.hidden_dim = config.hidden_size
        self.next_n = kwargs.get("next_n")
        self.spec_len = kwargs.get("spec_len")
        self.global_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.world_size = kwargs.get("world_size")
        self.global_rank = kwargs.get("global_rank")
        self.input_len = kwargs.get("input_len")
        self.dp_size = kwargs.get("dp_size")
        self.experts_tp_size = kwargs.get("experts_tp_size")
        self.die_num_per_node = kwargs.get("die_num_per_node")
        self.ep_size = kwargs.get("ep_size")
        self.route_ep_size = kwargs.get("route_ep_size")
        self.experts_share_num_copy = kwargs.get("experts_share_num_copy")
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts_per_rank = kwargs.get("n_routed_experts_per_rank", 1)
        self.route_share_on_same_card = kwargs.get("route_share_on_same_card", False)
        self.n_shared_experts = config.n_shared_experts or 0
        self.enable_gmm_tune_config =  int(os.getenv("ENABLE_GMM_TUNE_CONFIG", "0"))
        self.dynamic_quant_mode = kwargs.get("dynamic_quant_mode", 3)
        self.enable_prefetch = int(os.getenv("ENABLE_PREFETCH", "0"))
        self.attn_dies = int(os.getenv("ATTN_DIES", "1")) 
        self.ffn_dies = int(os.getenv("FFN_DIES", "1"))
        self.dense_layer_num = config.first_k_dense_replace
        self.enable_stream = _TODO_REQUIRE_API["enable_stream"]
        self.router_expert_num = kwargs.get("router_expert_num", 1)

        self.gate = MoEGate(config, **kwargs)
        self.experts = None
        self.shared_experts = None

        self.moe_all_to_all_group_name = kwargs.get("moe_all_to_all_group_name")

        self.enable_superkernel = kwargs.get("enable_superkernel", 0)
        self.enable_combine_dequant = kwargs.get("enable_combine_dequant", 0)
        self.on_cloud = kwargs.get("on_cloud", 0)
        self.attn_tp_size = kwargs.get("attn_tp_size")

        self.moe_expert_num = self.router_expert_num
        layer_out = os.getenv("LAYER_OUT", "FA")
        self.ffn_start_rank_id = self.attn_dies if layer_out.upper() == "AF" else 0
        self.shared_expert_rank_num = self.n_shared_experts * self.experts_share_num_copy
        if self.dynamic_quant_mode == 5:
            epsilon = 1e-2
            fix_rand_seed()
            all_experts_scale_1 = torch.rand((self.moe_expert_num + (0 if self.route_share_on_same_card else 1), config.hidden_size), dtype=torch.float32) * (1 - epsilon) + epsilon
            self.all_experts_scale_1 = nn.Parameter(all_experts_scale_1, requires_grad=False)
        self.batch_size = int(os.getenv("BATCH_SIZE", "1")) // self.attn_dies
        self.enable_micro_batch = kwargs.get("enable_micro_batch")
        if self.enable_micro_batch:
            self.batch_size = self.batch_size // micro_batch_number
        self.schedule_context = kwargs.get("schedule_context")
        self.context_holder = kwargs.get("context_holder")
        self.expert_rank_table = kwargs.get("expert_rank_table")
        self.selected_expert_num = self.top_k + 1
        self.layer_id_tensor = torch.Tensor([layer_idx]).to(torch.int32).npu()
        # self.moe_layer_id_tensor = torch.Tensor([layer_idx - self.dense_layer_num]).to(torch.int32).npu()
        self.moe_layer_id_tensor = torch.Tensor([0]).to(torch.int32).npu()
        attn_die_offset = 0 if layer_out.upper() == "AF" else self.ffn_dies
        self.session_id = torch.Tensor([self.global_rank - attn_die_offset]).to(torch.int32).npu()
        self.use_real_actual_seq_len = int(os.getenv("USE_REAL_ACTUAL_SEQ_LEN", "0"))

    @torch.no_grad()
    def forward(self, hidden_states, mb_id, last_combine_kwargs=None):
        topk_idx, topk_weight, _, row_idx = self.gate(hidden_states)
        combine_output = self.moe_infer_fusion(hidden_states, topk_idx, topk_weight, mb_id, last_combine_kwargs)
        return combine_output

    def moe_infer_fusion(self, x, topk_ids, topk_weight, mb_id, last_combine_kwargs):
        _, _, hidden_size = x.shape
        hidden_states = x.view(-1, hidden_size)
        batch_size = hidden_states.size(0)
        quant_mode = 2
        if quant_mode == 2:
            # 动态量化实际占用大小是self.hidden_size + 4, 然后对齐到512
            attantion_token_size = (hidden_size + 4 + 512 - 1) // 512 * 512
        else:
            attantion_token_size = hidden_size

        kwargs = {
            "x": hidden_states.unsqueeze(0), # [1*BS*H]
            "session_id": self.session_id,
            "micro_batch_id": mb_id,
            "layer_id": self.moe_layer_id_tensor,
            "expert_ids": topk_ids.unsqueeze(0),  # [1*n*topk]
            "expert_rank_table": self.expert_rank_table,  # 生成rank table
            "group": self.moe_all_to_all_group_name,
            "world_size": self.attn_dies + self.ffn_dies,
            "ffn_token_info_table_shape": [self.attn_dies, micro_batch_number, 1 + 1 + batch_size * self.selected_expert_num], # [A, M , F] TODO 确定按照字节还是int32
            "ffn_token_data_shape": [self.attn_dies, micro_batch_number, batch_size, self.selected_expert_num, attantion_token_size],  # [A, M, BS, K+1, HS]
            "attn_token_info_table_shape": [micro_batch_number, batch_size, self.selected_expert_num],  # [M, BS, K+1]
            "scales": None if self.dynamic_quant_mode != 5 else self.all_experts_scale_1, # 量化系数
            "quant_mode": quant_mode,  # 0: 非量化；1: 静态量化；2：动态量化
            "sync_flag": self.use_real_actual_seq_len,  # 0:同步， 1：异步
            "moe_expert_num": self.moe_expert_num,
            "ffn_start_rank_id": self.ffn_start_rank_id,
        }
        logger.info(f"npu_attention_to_ffn session_id:{self.session_id} x.shape:{hidden_states.shape}") if exe_mode != "dynamo" else None
        tng.scope.npu_wait_tensor(kwargs.get("x"), topk_weight) if exe_mode == "dynamo" else None
        attn2ffn_out = torch_npu.npu_attention_to_ffn(**kwargs)

        # with NpuStreamSwitch(self.enable_stream, '33'):
        #     schedule_context = self.schedule_context
        #     if self.enable_stream:
        #         schedule_context = tng.scope.npu_wait_tensor(schedule_context, hidden_states)
        #     torch_npu.attention_worker_scheduler_(schedule_context)

        combine_kwargs = {
            "schedule_context": self.schedule_context,
            "expert_scales": topk_weight,
            "layer_id": self.layer_id_tensor,
            "hidden_size": hidden_size,
            "token_dtype": 1 if model_dtype == torch.bfloat16 else 0,  # 0: FP16 1: BF16
            "need_schedule": 1
        }

        if self.enable_micro_batch and exe_mode == "dynamo":
            combine_output = hidden_states
            if last_combine_kwargs is not None:
                combine_output, _ = torch_npu.npu_attention_worker_combine(**last_combine_kwargs)
        else:
            combine_output, _ = torch_npu.npu_attention_worker_combine(**combine_kwargs)

        return combine_output, attn2ffn_out, combine_kwargs


class DeepseekV2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV2Config, layer_idx: Optional[int] = None, **kwargs):
        super().__init__()
        self.enable_stream = kwargs.get("enable_stream")
        self.enable_prefetch = kwargs.get("enable_prefetch")
        self.world_size = kwargs.get("world_size")
        self.batch_size = kwargs.get("batch_size")
        self.enable_micro_batch = kwargs.get("enable_micro_batch")
        if layer_idx < config.first_k_dense_replace:
            self.enable_micro_batch = 0
        if self.enable_micro_batch:
            self.batch_size = self.batch_size // micro_batch_number
        self.attn_tp_size = kwargs.get("attn_tp_size")
        self.attn_dp_size = kwargs.get("attn_dp_size")
        self.o_proj_tp_size = kwargs.get("o_proj_tp_size")
        self.o_proj_group = kwargs.get("o_proj_group")
        if self.enable_micro_batch:
            self.all2all_o_proj_shape = kwargs.get("all2all_o_proj_shape_mb")
            self.reduce_scatter_o_proj_shape = kwargs.get("reduce_scatter_o_proj_shape_mb")
        else:
            self.all2all_o_proj_shape = kwargs.get("all2all_o_proj_shape")
            self.reduce_scatter_o_proj_shape = kwargs.get("reduce_scatter_o_proj_shape")
        self.dynamic_quant_mode = kwargs.get("dynamic_quant_mode")
        self.div_mode = kwargs.get("div_mode")
        self.use_merge = int(os.getenv("USE_MERGE", "0"))
        self.next_n = kwargs.get("next_n", 0)
        self.spec_len = kwargs.get("spec_len", 0)
        self.use_fa_tensor = int(os.getenv("USE_FA_TENSOR", "0"))
        self.enable_fa_quant = int(os.getenv("ENABLE_FA_QUANT", "0")) and exe_mode =="dynamo"

        self.config = config
        self.out_dtype = config.torch_dtype
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.moe_flag = (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
        )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.num_key_value_heads_per_rank = self.num_heads_per_rank

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.kvcache_nz = int(os.getenv("KVCACHE_NZ", "0"))

        if not self.use_merge:
            if self.q_lora_rank is None:
                self.q_proj = NpuLinear(
                    self.hidden_size, self.num_heads_per_rank * self.q_head_dim, bias=False
                ).to(model_dtype)
            else:
                self.q_a_proj = NpuLinear(
                    self.hidden_size, config.q_lora_rank, bias=config.attention_bias
                ).to(model_dtype)
                self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
                self.q_b_proj = NpuLinear(
                    config.q_lora_rank, self.num_heads_per_rank * self.q_head_dim, bias=False
                ).to(model_dtype)

            self.kv_a_proj_with_mqa = NpuLinear(
                self.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                bias=config.attention_bias,
            ).to(model_dtype)
        else:
            self.merged_qkv_a_proj = NpuLinear(
                self.hidden_size,
                config.q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim,
                bias=config.attention_bias,
            ).to(model_dtype)
            self.q_a_layernorm = DeepseekV2RMSNorm(config.q_lora_rank)
            self.q_b_proj = NpuLinear(
                config.q_lora_rank, self.num_heads_per_rank * self.q_head_dim, bias=False
            ).to(model_dtype)

        norm_res_bsz = self.batch_size * self.attn_tp_size // self.world_size
        q_len = self.spec_len if self.next_n > 0 else 1
        self.norm_res = torch.zeros([norm_res_bsz, q_len, config.q_lora_rank], dtype=model_dtype, device="npu")

        if _TODO_REQUIRE_API["enable_pa"]:
            self.max_len = _PAGE_ATTENTION_SETTING["max_length"]
            self.block_size = _PAGE_ATTENTION_SETTING["block_size"]
            self.cache_len = self.max_len // self.block_size

            batch_size = self.batch_size * self.attn_tp_size // self.world_size
            block_table = torch.arange(0, batch_size * self.cache_len).reshape(batch_size, -1) + 1
            block_table_full = torch.nn.functional.pad(block_table, (0, math.ceil(self.max_len / self.block_size) - self.cache_len), "constant", 1)
            self.block_table = torch.zeros_like(block_table_full) - 1 + block_table_full
            self.block_table = self.block_table.to(torch.int32).npu()
            assert _TODO_REQUIRE_API["actual_seq_len"] <= self.max_len

        self.kv_a_layernorm = DeepseekV2RMSNorm(config.kv_lora_rank)
        fix_rand_seed()
        self.kv_b_proj_w_k = nn.Parameter(torch.rand((self.num_heads_per_rank, self.qk_nope_head_dim, self.kv_lora_rank), dtype=model_dtype))
        self.kv_b_proj_w_v = nn.Parameter(torch.rand((self.num_heads_per_rank, self.kv_lora_rank, self.v_head_dim), dtype=model_dtype))
        self.kv_b_proj = None
        self.o_proj = NpuLinear(
            self.num_heads_per_rank * self.v_head_dim // self.o_proj_tp_size,
            self.hidden_size,
            bias=config.attention_bias,
        ).to(model_dtype)
        self.softmax_scale = self.q_head_dim ** (-0.5)

        if _TODO_REQUIRE_API["enable_mla_prolog"] and model_dtype == torch.bfloat16 and (self.dynamic_quant_mode == 2 or self.enable_fa_quant):
            fix_rand_seed()
            self.quant_scale_ckv = nn.Parameter(torch.rand(1, dtype=torch.float), requires_grad=False)
            self.quant_scale_ckv_fa = nn.Parameter(torch.rand(self.kv_lora_rank, dtype=torch.float), requires_grad=False)
        else:
            self.quant_scale_q = None
            self.quant_scale_ckv = None
            self.kv_scale = None
            
    def compute_q_kv_decode_mla_prolog(
        self, hidden_states, in_scale, 
        weight_qa, out_scale_qa, 
        weight_kva, out_scale_kva, 
        cos_sin, past_key_value, kv_len, 
        weight_qb, in_scale_qb, out_scale_qb,
        **kwargs):
        bsz, q_len, _ = hidden_states.size()
        if self.enable_fa_quant:
            hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        cos, sin = cos_sin
        cos = cos.squeeze(1)
        sin = sin.squeeze(1)

        cache_index = kv_len.view(bsz, -1)
        kv_cache = past_key_value[self.layer_idx][0].unsqueeze(2)            # past_key_value[self.layer_idx][0]
        kr_cache = past_key_value[self.layer_idx][1].unsqueeze(2)             # past_key_value[self.layer_idx][1]
        dequant_scale_query = None
        if self.dynamic_quant_mode == 2:
            mla_q_nope, mla_q_pe, mla_k_nope, mla_k_rope = torch.ops.npu.npu_mla_prolog(token_x = hidden_states,
                weight_dq = weight_qa, weight_uq_qr = weight_qb,
                weight_uk = self.kv_b_proj_w_k, weight_dkv_kr = weight_kva,
                rmsnorm_gamma_cq = self.q_a_layernorm.weight, rmsnorm_gamma_ckv = self.kv_a_layernorm.weight,
                rope_sin = sin, rope_cos = cos, cache_index = cache_index, kv_cache = kv_cache, kr_cache = kr_cache,
                dequant_scale_x = None if in_scale is None else torch.reciprocal(in_scale.repeat(hidden_states.shape[0] // in_scale.shape[0]).unsqueeze(1)), # pertensor quant, only use the first element of the vector , reciprocal
                dequant_scale_w_dq = None if out_scale_qa is None else out_scale_qa.unsqueeze(0),  # pertensor quant, only use the first element of the vector, reciprocal
                dequant_scale_w_uq_qr = None if out_scale_qb is None else out_scale_qb.unsqueeze(0), # pertensor quant, only use the first element of the vector, reciprocal
                dequant_scale_w_dkv_kr = None if out_scale_kva is None else out_scale_kva.unsqueeze(0), # pertensor quant, only use the first element of the vector, reciprocal
                smooth_scales_cq = None if in_scale_qb is None else in_scale_qb.unsqueeze(0), # pertensor quant, only use the first element of the vector , not reciprocal
                rmsnorm_epsilon_cq = self.q_a_layernorm.variance_epsilon,
                rmsnorm_epsilon_ckv = self.kv_a_layernorm.variance_epsilon,
                quant_scale_q = None if self.quant_scale_q is None else self.quant_scale_q.unsqueeze(1).repeat(1, self.kv_lora_rank),  # (N) --> (N, D)
                quant_scale_ckv = None if self.quant_scale_ckv is None else self.quant_scale_ckv.unsqueeze(0),
                cache_mode = "PA_NZ")
        elif self.enable_fa_quant:
            mla_q_nope, mla_q_pe, mla_k_nope, mla_k_rope, dequant_scale_q_nope = torch.ops.npu.npu_mla_prolog_v2(token_x = hidden_states,
                weight_dq = weight_qa, weight_uq_qr = weight_qb,
                weight_uk = self.kv_b_proj_w_k, weight_dkv_kr = weight_kva,
                rmsnorm_gamma_cq = self.q_a_layernorm.weight, rmsnorm_gamma_ckv = self.kv_a_layernorm.weight,
                rope_sin = sin, rope_cos = cos, cache_index = cache_index, kv_cache = kv_cache, kr_cache = kr_cache,
                dequant_scale_x = pertoken_scale.view(-1, 1), # pertoken quant
                dequant_scale_w_dq = out_scale_qa.view(1, -1),  # pertensor quant, only use the first element of the vector, reciprocal
                dequant_scale_w_uq_qr = out_scale_qb.view(1, -1), # pertensor quant, only use the first element of the vector, reciprocal
                dequant_scale_w_dkv_kr = out_scale_kva.view(1, -1), # pertensor quant, only use the first element of the vector, reciprocal
                quant_scale_ckv=self.quant_scale_ckv_fa.view(1, -1),
                quant_scale_ckr=None,
                smooth_scales_cq = None,
                rmsnorm_epsilon_cq = self.q_a_layernorm.variance_epsilon,
                rmsnorm_epsilon_ckv = self.kv_a_layernorm.variance_epsilon,
                cache_mode = "PA_NZ")
            dequant_scale_query = dequant_scale_q_nope.view(bsz, q_len, -1)
        else:
            mla_q_nope, mla_q_pe, mla_k_nope, mla_k_rope = torch.ops.npu.npu_mla_prolog(token_x = hidden_states,
                weight_dq = weight_qa, weight_uq_qr = weight_qb,
                weight_uk = self.kv_b_proj_w_k, weight_dkv_kr = weight_kva,
                rmsnorm_gamma_cq = self.q_a_layernorm.weight, rmsnorm_gamma_ckv = self.kv_a_layernorm.weight,
                rope_sin = sin, rope_cos = cos, cache_index = cache_index, kv_cache = kv_cache, kr_cache = kr_cache,
                dequant_scale_x = None if in_scale is None else torch.reciprocal(in_scale.unsqueeze(1)), # pertensor quant, only use the first element of the vector , reciprocal
                dequant_scale_w_dq = None if out_scale_qa is None else out_scale_qa.unsqueeze(0),  # pertensor quant, only use the first element of the vector, reciprocal
                dequant_scale_w_uq_qr = None if out_scale_qb is None else out_scale_qb.unsqueeze(0), # pertensor quant, only use the first element of the vector, reciprocal
                dequant_scale_w_dkv_kr = None if out_scale_kva is None else out_scale_kva.unsqueeze(0), # pertensor quant, only use the first element of the vector, reciprocal
                smooth_scales_cq = None if in_scale_qb is None else in_scale_qb.unsqueeze(0), # pertensor quant, only use the first element of the vector , not reciprocal
                rmsnorm_epsilon_cq = self.q_a_layernorm.variance_epsilon,
                rmsnorm_epsilon_ckv = self.kv_a_layernorm.variance_epsilon,
                # quant_scale_q = None if self.quant_scale_q is None else self.quant_scale_q.unsqueeze(1).repeat(1, self.kv_lora_rank),  # (N) --> (N, D)
                quant_scale_ckv = None if self.quant_scale_ckv is None else self.quant_scale_ckv.unsqueeze(0),
                cache_mode = "PA_NZ")

        mla_query_states = [mla_q_nope, mla_q_pe]
        mla_key_states = [mla_k_nope, mla_k_rope]  # (bs, 1, qlen, kv_lora_rank + qk_rope_head_dim)

        return mla_query_states, mla_key_states, dequant_scale_query

    def compute_q_kv_decode_multistream(
        self, hidden_states, in_scale,
        weight_qa, out_scale_qa, 
        weight_kva, out_scale_kva, 
        cos_sin, past_key_value, kv_len, enable_stream=False, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        cos, sin = cos_sin
        if self.dynamic_quant_mode in [2, 3]:
            q_lowrank = torch_npu.npu_quant_matmul(
                hidden_states, weight_qa, out_scale_qa, bias=None,
                output_dtype=model_dtype)

            with NpuStreamSwitch(enable_stream, '11'):
                kv = hidden_states
                if enable_stream:
                    kv = tng.scope.npu_wait_tensor(kv, q_lowrank)
                kv = torch_npu.npu_quant_matmul(
                    kv, weight_kva, out_scale_kva, bias=None,
                    output_dtype=model_dtype)
        else:
            q_lowrank = torch.matmul(hidden_states, weight_qa)
            with NpuStreamSwitch(enable_stream, '11'):
                kv = hidden_states
                if enable_stream:
                    kv = tng.scope.npu_wait_tensor(kv, q_lowrank)
                kv = torch.matmul(kv, weight_kva)
                
        q = self.q_a_layernorm(q_lowrank, self.norm_res)[0]
        q = q.view(bsz, q_len, -1)
        q = self.q_b_proj(q)
        # q = torch.matmul(q, weight_qb)
        if self.next_n > 0:  # TODO, need to check for q_len under spec
            q_len = self.spec_len
        q = q.view(bsz, q_len, self.num_heads_per_rank, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,s,n,d

        if exe_mode == "dynamo" and self.kv_b_proj_w_k.shape[0] * self.kv_b_proj_w_k.shape[1] <= 65535:
            q_nope = q_nope.view(-1, self.num_heads_per_rank, self.qk_nope_head_dim) # bs,n,d -> n,bs,d
            q_nope = torch_npu.npu_transpose_batchmatmul(q_nope, self.kv_b_proj_w_k, bias=None, scale=None,
                                                         perm_x1=(1,0,2), perm_x2=(0,1,2), perm_y=(1,0,2))
            q_nope = q_nope.view(bsz, q_len, self.num_heads_per_rank, -1)
        else:
            q_nope = q_nope.view(-1, self.num_heads_per_rank, self.qk_nope_head_dim).transpose(0, 1) # bs,n,d -> n,bs,d
            q_nope = (
                torch.matmul(q_nope, self.kv_b_proj_w_k)
                .transpose(1, 0)
                .view(bsz, q_len, self.num_heads_per_rank, -1)
            )

        with NpuStreamSwitch(enable_stream, '11'):
            if not _TODO_REQUIRE_API["enable_pa"]:
                # need view (BS, N, 1, D) cause rope_single not support BSND
                kv = kv.view(bsz*q_len, 1, 1, -1)
                kv_len = kv_len.view(bsz, -1)
                cos = cos.view(bsz*q_len, 1, 1, self.qk_rope_head_dim)
                sin = sin.view(bsz*q_len, 1, 1, self.qk_rope_head_dim)
                k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                    kv, self.kv_a_layernorm.weight, 
                    cos, sin, kv_len, 
                    past_key_value[self.layer_idx][1], past_key_value[self.layer_idx][0],
                    epsilon=self.kv_a_layernorm.variance_epsilon)
            else:
                # need view (BS, N, 1, D) cause rope_single not support BSND
                kv = kv.view(bsz*q_len, 1, 1, -1)
                kv_len = kv_len.view(-1)
                cos = cos.view(bsz*q_len, 1, 1, self.qk_rope_head_dim)
                sin = sin.view(bsz*q_len, 1, 1, self.qk_rope_head_dim)
                cache_rope = past_key_value[self.layer_idx][1].unsqueeze(2)
                cache_nope = past_key_value[self.layer_idx][0].unsqueeze(2)
                k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                    kv, self.kv_a_layernorm.weight,
                    cos, sin, kv_len.view(-1),
                    cache_rope, cache_nope,
                    epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ" if self.kvcache_nz else "PA")
                k_rope = k_rope.squeeze(2)
                k_nope = k_nope.squeeze(2)

            # need view (BS, N, 1, D) cause rope_single not support BSND
            q_pe = q_pe.view(bsz*q_len, self.num_heads_per_rank, 1, self.qk_rope_head_dim)
            q_pe = rope_single(q_pe, cos, sin)
            q_pe = q_pe.view(bsz, q_len, self.num_heads_per_rank, self.qk_rope_head_dim)

        query_states = [q_nope, q_pe]
        key_states = [k_nope, k_rope]  # (bs, 1, qlen, kv_lora_rank + qk_rope_head_dim)
        return query_states, key_states, None
    
    def apply_attention_out_npu_decode(
        self,
        bsz, q_len, attn_output
    ):
        if self.o_proj_tp_size > 1:
            # TP split K-dim all2allv
            attn_output = attn_output.view(bsz*q_len, self.o_proj_tp_size, -1).transpose(0,1).contiguous()
            all2all_o_proj = torch.empty(self.all2all_o_proj_shape, dtype=model_dtype, device="npu")
            dist.all_to_all_single(all2all_o_proj, attn_output.view(-1), group=self.o_proj_group)
            attn_output = self.o_proj(all2all_o_proj.view(bsz*q_len*self.o_proj_tp_size, -1))
            reduce_scatter_o_proj = torch.empty(self.reduce_scatter_o_proj_shape, dtype=model_dtype, device="npu")
            dist.reduce_scatter_tensor(reduce_scatter_o_proj, attn_output, group=self.o_proj_group)
            attn_output = reduce_scatter_o_proj.view(bsz, q_len, -1)
        else:
            attn_output = self.o_proj(attn_output.view(-1, attn_output.shape[-1]))
            attn_output = attn_output.view(bsz, q_len, -1)

        return attn_output


    def apply_attention_npu_decode(
        self,
        query_states, key_states, value_states, kv_seq_len,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        past_key_value: Optional[Cache] = None,
        dequant_scale_query = None
    ):
        bsz, q_len, _, q_dim = query_states[0].size()
        attn_mask = attention_mask
        sparse_mode = 3
        if self.next_n == 0:
            attn_mask = None
            sparse_mode = 0
        if True:
            use_BSND_NBSD = exe_mode == "dynamo" and q_len in [1, 2]
            npu_flash_attention = torch.ops.npu.npu_fused_infer_attention_score_v2
            if not _TODO_REQUIRE_API["enable_pa"]:
                # use torch_npu FIA kernel
                attn_output, _ = npu_flash_attention(
                        query_states[0], key_states[0], value_states[0], query_rope=query_states[1], key_rope=key_states[1],
                        num_heads=self.num_heads_per_rank,
                        num_key_value_heads=1, input_layout="BSND",
                        atten_mask=attention_mask, 
                        actual_seq_lengths_kv=actual_seq_lengths_kv, scale=self.softmax_scale,
                        antiquant_mode=0, antiquant_scale=None)
            else:
                if self.enable_fa_quant:
                    attn_output, _ = npu_flash_attention(query_states[0], key_states[0],
                                                         value_states[0], query_rope=query_states[1],
                                                         key_rope=key_states[1],
                                                         atten_mask=attn_mask,
                                                         actual_seq_kvlen=actual_seq_lengths_kv,
                                                         block_table=self.block_table,
                                                         dequant_scale_query=dequant_scale_query,
                                                         dequant_scale_key=self.quant_scale_ckv,
                                                         dequant_scale_value=self.quant_scale_ckv,
                                                         num_query_heads=self.num_heads_per_rank,
                                                         num_key_value_heads=1,
                                                         softmax_scale=self.softmax_scale,
                                                         input_layout="BSND_NBSD" if use_BSND_NBSD else "BSND",
                                                         sparse_mode=sparse_mode,
                                                         block_size=self.block_size,
                                                         query_quant_mode=3, key_quant_mode=0,
                                                         value_quant_mode=0
                                                         )
                else:
                    if self.dynamic_quant_mode == 2:
                        dequant_scale1 = torch.reciprocal(self.quant_scale_q).view(1, 1, -1).repeat(bsz, q_len, 1)
                        dequant_scale2 = torch.reciprocal(self.quant_scale_ckv)
                    else:
                        dequant_scale1, dequant_scale2 = None, None
                    # use torch_npu Paged attention kernel
                    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                            query_states[0], key_states[0], value_states[0], query_rope=query_states[1], key_rope=key_states[1],
                            dequant_scale1=dequant_scale1,
                            dequant_scale2=dequant_scale2,
                            num_heads=self.num_heads_per_rank,
                            num_key_value_heads=1, input_layout="BSND_NBSD" if use_BSND_NBSD else "BSND",
                            atten_mask=attn_mask, scale=self.softmax_scale,
                            antiquant_mode=0, antiquant_scale=None,
                            block_table=self.block_table,
                            block_size=self.block_size,
                            actual_seq_lengths_kv=actual_seq_lengths_kv,
                            sparse_mode=sparse_mode
                        )
        else:
            query_states = torch.cat(query_states, dim=-1)
            key_states = torch.cat(key_states, dim=-1)
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
            )
            # if attention_mask is not None:
            #    attn_weights = attn_weights + attention_mask[..., :attn_weights.shape[-1]]

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.attention_dropout, training=self.training
            )
            attn_output = torch.matmul(attn_weights, value_states[0])
        
        if use_BSND_NBSD:
            attn_output = attn_output.view(self.num_heads_per_rank, bsz*q_len, self.kv_lora_rank)
        else:
            attn_output = attn_output.view(bsz*q_len, self.num_heads_per_rank, self.kv_lora_rank).transpose(0, 1)
        # attn_output = (
        #     torch.matmul(attn_output, self.kv_b_proj_w_v)  # grahp fusion stage will produce a matmul+transpose operator
        #     .transpose(1, 0)
        #     .reshape(bsz, q_len, -1)
        # )

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        cos_sin: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ):
        kwargs.update({
            "hidden_states": hidden_states,
            "cos_sin": cos_sin,
            "past_key_value": past_key_value,
            "kv_len": kv_len
        })
        enable_mla_prolog = _TODO_REQUIRE_API["enable_mla_prolog"] and model_dtype == torch.bfloat16 and self.dynamic_quant_mode in [4, 5, 2]
        if enable_mla_prolog:
            decode_func = self.compute_q_kv_decode_mla_prolog
        else:
            kwargs.update({"enable_stream": self.enable_stream})
            decode_func = self.compute_q_kv_decode_multistream
        query_states, key_states, dequant_scale_query = decode_func(**kwargs)
        
        # update kv cache
        if self.kvcache_nz:
            key_states = self.modify_kvcache_layout(key_states)

        value_states = key_states
        kv_seq_len = self.config.max_position_embeddings

        hidden_states = self.apply_attention_npu_decode(
            query_states=query_states, key_states=key_states, value_states=value_states,
            kv_seq_len=kv_seq_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            dequant_scale_query=dequant_scale_query,
        )
        return hidden_states

    def modify_kvcache_layout(self, key_states):
        '''
        modify kvcache layout to NZ format
        '''
        block_num = key_states[0].size()[0]
        key_states[0] = key_states[0].view(block_num, 1, self.kv_lora_rank // (32 if key_states[0].dtype == torch.int8 else 16), self.block_size, (32 if key_states[0].dtype == torch.int8 else 16))
        key_states[1] = key_states[1].view(block_num, 1, self.qk_rope_head_dim // 16, self.block_size, 16)
        return key_states


ATTENTION_CLASSES = {
    "eager": DeepseekV2Attention,
}
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV2Config, layer_idx: int, **kwargs):
        super().__init__()
        self.config = config
        self.enable_stream = kwargs.get("enable_stream")
        self.enable_prefetch = kwargs.get("enable_prefetch")
        self.world_size = kwargs.get("world_size")
        self.local_rank = kwargs.get("local_rank")
        self.global_rank = kwargs.get("global_rank")
        self.dynamic_quant_mode = kwargs.get("dynamic_quant_mode")
        self.batch_size = kwargs.get("batch_size")
        self.micro_mode = kwargs.get("micro_mode")
        self.enable_micro_batch = kwargs.get("enable_micro_batch")
        if layer_idx < config.first_k_dense_replace:
            self.enable_micro_batch = 0
        if self.enable_micro_batch:
            self.batch_size = self.batch_size // micro_batch_number
        self.input_len = kwargs.get("input_len")
        self.attn_tp_size = kwargs.get("attn_tp_size")
        self.attn_dp_size = kwargs.get("attn_dp_size")
        self.die_num_per_node = kwargs.get("die_num_per_node", 16)
        self.div_mode = kwargs.get("div_mode", 1)
        self.enable_superkernel = kwargs.get("enable_superkernel", 0)
        self.enable_combine_dequant = kwargs.get("enable_combine_dequant", 0)
        self.enable_fa_quant = int(os.getenv("ENABLE_FA_QUANT", "0")) and exe_mode =="dynamo"

        self.vocab_ranks_comm_group = kwargs.get("vocab_ranks_comm_group")
        self.all_gather_dense_shape = kwargs.get("all_gather_dense_shape")
        self.all_gather_scale_shape = kwargs.get("all_gather_scale_shape", None)
        self.reduce_scatter_dense_shape = kwargs.get("reduce_scatter_dense_shape")
        self.dense_gather_group = kwargs.get("dense_gather_group", None)
        self.attn_gather_scatter_group = kwargs.get("attn_gather_scatter_group")
        if self.enable_micro_batch:
            self.attn_input_decode_shape = kwargs.get("attn_input_decode_shape_mb")
        else:
            self.attn_input_decode_shape = kwargs.get("attn_input_decode_shape")

        self.all_gather_dense = torch.empty(self.all_gather_dense_shape, dtype=torch.int8).npu()
        self.all_gather_scale = torch.empty(self.all_gather_scale_shape, dtype=torch.float).npu()
        self.reduce_scatter_dense = torch.empty(self.reduce_scatter_dense_shape, dtype=model_dtype).npu()
        self.attn_input_decode = torch.empty(self.attn_input_decode_shape, dtype=torch.int8 if self.dynamic_quant_mode in [2, 3] else model_dtype).npu()
        
        self.dense_tp_size = kwargs.get("dense_tp_size", 16)
        self.hidden_size = config.hidden_size
        self.next_n = kwargs.get("next_n", 0)
        self.spec_len = kwargs.get("spec_len", 0)
        self.is_spec = kwargs.get("is_spec", False)

        if config.model_type in ["deepseek_v2", "deepseek_v3"]:
            self.self_attn = ATTENTION_CLASSES[config._attn_implementation](
                config=config, layer_idx=layer_idx, **kwargs
            )
        else:
            raise Exception("only support deepseek MLA attention!")

        self.moe_flag = (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
        )
        self.layer_idx = layer_idx

        self.mlp = (
            npu_DeepseekV2MoE_ep(config, layer_idx, **kwargs) if self.moe_flag else DeepseekV2MLP(config, **kwargs)
        )
        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.init_decode()
    
    def init_quant_weight(self, param, in_features, out_features_q, enable_quant: int = 0):
        '''
        enable_quant: int,  0: no quant; 1: pertoken quant; 2: pertensor quant
        '''
        weight_qa = copy.deepcopy(param.weight)
        weight_qa.requires_grad = False
        in_scale = None
        out_scale_q = None
        epsilon = 1e-2
        if enable_quant in [1, 2]:
            weight_qa.data = apply_quant(weight_qa.data)
            fix_rand_seed()
            if enable_quant == 2:
                in_scale = nn.Parameter(1 / (torch.rand(self.batch_size * self.spec_len // self.world_size, dtype=torch.float) * (1 - epsilon) + epsilon), requires_grad=False)
            else:
                in_scale = nn.Parameter(1 / (torch.rand(in_features, dtype=torch.float) * (1 - epsilon) + epsilon), requires_grad=False)
            out_scale_q = nn.Parameter(torch.rand(out_features_q, dtype=torch.float) * (1 - epsilon) + epsilon, requires_grad=False)
            if model_dtype == torch.float16:
                out_scale_q = torch_npu.npu_trans_quant_param(out_scale_q.npu(), None)  # to uint64

        return in_scale, out_scale_q, weight_qa

    def init_decode(self):
        if self.dynamic_quant_mode == 3 or self.enable_fa_quant:
            enable_quant = 1  # 0: no quant; 1: pertoken quant; 2: pertensor quant
        elif self.dynamic_quant_mode == 2:
            enable_quant = 2
        else:
            enable_quant = 0

        self.in_scale, self.out_scale_qa, self.self_attn.q_a_proj.weight = self.init_quant_weight(
            self.self_attn.q_a_proj, 
            self.hidden_size, 
            self.config.q_lora_rank, enable_quant)

        self.in_scale, self.out_scale_kva, self.self_attn.kv_a_proj_with_mqa.weight = self.init_quant_weight(
            self.self_attn.kv_a_proj_with_mqa, 
            self.hidden_size, 
            self.config.kv_lora_rank + self.config.qk_rope_head_dim, enable_quant)

        if self.dynamic_quant_mode in [2, 5]:
            enable_quant = 1
        self.in_scale_qb, self.out_scale_qb, self.self_attn.q_b_proj.weight = self.init_quant_weight(
            self.self_attn.q_b_proj, 
            self.config.q_lora_rank, 
            self.self_attn.num_heads_per_rank * self.self_attn.q_head_dim,
            enable_quant)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        pass

    def forward_decode_part_one(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_residual: Optional[torch.Tensor] = None,
        batch_id=1,
        combine_kwargs=None,
        q_len=None,
        hidden_size=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        enable_superkernel = self.enable_superkernel and (not self.is_spec) # and (self.layer_idx != self.config.first_k_dense_replace)
        if enable_superkernel:
            if self.enable_micro_batch:
                label = f'mla'
                option = "option_yyy"
            else:
                label = f'mlp_{self.layer_idx}_{batch_id}'
                option = "option_xxx"
        else:
            label, option = None, None

        kwargs = {
            "in_scale": self.in_scale,
            "weight_qa": self.self_attn.q_a_proj.weight,
            "out_scale_qa": self.out_scale_qa,
            "weight_kva": self.self_attn.kv_a_proj_with_mqa.weight,
            "out_scale_kva": self.out_scale_kva,
            "weight_qb": self.self_attn.q_b_proj.weight,
            "in_scale_qb": self.in_scale_qb,
            "out_scale_qb": self.out_scale_qb
        }

        ####### STEP 1 #####
        # DP288
        bsz, q_len, hidden_size = hidden_states.size()
        with SuperKernelScope(enable_superkernel and not self.enable_micro_batch, label, option):
            hidden_states, residual = self.input_layernorm(hidden_states, past_residual)
            hidden_states = hidden_states.view(-1, hidden_size)

            if self.attn_tp_size > 1:
                hidden_states = hidden_states.view(-1, self.hidden_size)
                dist.all_gather_into_tensor(self.attn_input_decode, hidden_states, group=self.attn_gather_scatter_group)
                hidden_states = self.attn_input_decode

        hidden_states = hidden_states.view(-1, q_len, hidden_size)

        if self.layer_idx <= self.config.first_k_dense_replace and batch_id == 1:
            aic_num = 24
            aiv_num = 48
        else:
            aic_num = 16
            aiv_num = 32
        with SuperKernelScope(enable_superkernel, label, option):
            if exe_mode == "dynamo":
                with tng.scope.limit_core_num(aic_num, aiv_num):
                    hidden_states = self.self_attn.forward(hidden_states, kv_len, actual_seq_lengths_kv, cos_sin, attention_mask, past_key_value, **kwargs)
            else:
                hidden_states = self.self_attn.forward(hidden_states, kv_len, actual_seq_lengths_kv, cos_sin, attention_mask, past_key_value, **kwargs)
        outputs = (residual, hidden_states)

        enable_superkernel = self.enable_superkernel and (not self.is_spec)
        if self.enable_micro_batch:
            label = f'mlp_{self.layer_idx}_{batch_id}'
            option = "feed-sync-all=1"
        else:
            label = f'mlp_{self.layer_idx}_{batch_id}'
            option = "option_xxx"
        with SuperKernelScope(enable_superkernel, label, option):
            hidden_states = (
                torch.matmul(hidden_states, self.self_attn.kv_b_proj_w_v)  # grahp fusion stage will produce a matmul+transpose operator
                .transpose(1, 0)
                .reshape(bsz * self.attn_tp_size, q_len, -1)
            )
        return outputs
        
    def forward_decode_part_two(
        self,
        hidden_states: torch.Tensor,
        bsz:int, 
        q_len:int, 
        hidden_size:int,
        residual: Optional[torch.Tensor] = None,
        batch_id=1,
        mb_id=torch.Tensor([1]),
        last_combine_kwargs=None,
        combine_out=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        enable_superkernel = self.enable_superkernel and (not self.is_spec)
        if self.enable_micro_batch:
            label = f'mlp_{self.layer_idx}_{batch_id}'
            option = "feed-sync-all=1"
        else:
            label = f'mlp_{self.layer_idx}_{batch_id}'
            option = "option_xxx"
        with SuperKernelScope(enable_superkernel, label, option):
            # hidden_states = (
            #     torch.matmul(hidden_states, self.self_attn.kv_b_proj_w_v)  # grahp fusion stage will produce a matmul+transpose operator
            #     .transpose(1, 0)
            #     .reshape(bsz * self.attn_tp_size, q_len, -1)
            # )
            hidden_states = self.self_attn.apply_attention_out_npu_decode(bsz * self.attn_tp_size, q_len, hidden_states)
            if self.attn_tp_size > 1:
                hidden_states = hidden_states.view(-1, hidden_size) # bs//dp, hid
                hidden_states = dist._functional_collectives.reduce_scatter_tensor(hidden_states,
                                        "sum", scatter_dim=0, group=self.attn_gather_scatter_group)

            hidden_states = hidden_states.view(-1, q_len, hidden_size)
        
        ####### STEP 2 #####
        # EP288
        if combine_out is not None and exe_mode == "dynamo":
            tng.scope.npu_wait_tensor(hidden_states, combine_out)
        combine_kwargs = None
        attn2ffn_out = None
        if not self.moe_flag:
            if self.enable_fa_quant:
                with SuperKernelScope(enable_superkernel, label, option):
                    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
                    hidden_states = hidden_states.view(-1, hidden_size)
                    x, x_scale = torch_npu.npu_dynamic_quant(hidden_states, smooth_scales=self.mlp.merge_up_gate_proj.in_scale)

                dist.all_gather_into_tensor(self.all_gather_scale, x_scale.view(-1), group=self.dense_gather_group)
                dist.all_gather_into_tensor(self.all_gather_dense, x.reshape(-1, x.size(-1)), group=self.dense_gather_group)
                torch_npu.npu_prefetch(self.mlp.down_proj.weight.data, hidden_states, FFN2_PREFETCH_SIZE, 0)
                with SuperKernelScope(enable_superkernel, f'mlp_{self.layer_idx}_{batch_id}_1', 'option_xxx'):
                    hidden_states = self.mlp(self.all_gather_dense, is_quant=True, dynamic_scale=self.all_gather_scale, allow_combine_dequant=False)[0]

                    # dense mlp (DP, TP) --> (DP288)
                    hidden_states_for_prefetch = hidden_states.view(-1, hidden_size)
                    dist.reduce_scatter_tensor(self.reduce_scatter_dense, hidden_states_for_prefetch, group=self.dense_gather_group)
                    hidden_states = self.reduce_scatter_dense.view(bsz, q_len, hidden_size)
            else:
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
                all_gather_dense = torch.empty(self.all_gather_dense_shape, dtype=model_dtype, device="npu")
                dist.all_gather_into_tensor(all_gather_dense, hidden_states.view(-1, hidden_size), group=self.dense_gather_group)
                hidden_states = all_gather_dense
                hidden_states = self.mlp(hidden_states, allow_combine_dequant=False)[0]
                hidden_states_for_prefetch = hidden_states.view(-1, hidden_size)
                dist.reduce_scatter_tensor(self.reduce_scatter_dense, hidden_states_for_prefetch, group=self.dense_gather_group)
                hidden_states = self.reduce_scatter_dense.view(bsz, q_len, hidden_size)
        else:
            with SuperKernelScope(enable_superkernel, label, option):
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
                hidden_states, attn2ffn_out, combine_kwargs = self.mlp(hidden_states, mb_id, last_combine_kwargs)
                hidden_states_for_prefetch = None

        hidden_states = hidden_states.view(-1, q_len, hidden_size)
        # DP288 out
        outputs = (residual, hidden_states, hidden_states_for_prefetch, attn2ffn_out, combine_kwargs)
        return outputs
    
    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_residual: Optional[torch.Tensor] = None,
        batch_id=1,
        mb_id=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:

        bsz, q_len, hidden_size = hidden_states.size()
        residual, hidden_states = self.forward_decode_part_one(
            hidden_states=hidden_states,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            cos_sin=cos_sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            past_residual=past_residual,
            batch_id=batch_id
        )

        residual, hidden_states, hidden_states_for_prefetch, _, _ = self.forward_decode_part_two(
            hidden_states=hidden_states,
            bsz=bsz, 
            q_len=q_len,
            hidden_size=hidden_size,
            residual=residual,
            batch_id=batch_id,
            mb_id=mb_id,
            **kwargs
        )

        # DP288 out
        outputs = (residual, hidden_states, hidden_states_for_prefetch)
        return outputs



@add_start_docstrings(
    "The bare DeepseekV2 Model outputting raw hidden-states without any specific head on top.",
    DeepseekV2_START_DOCSTRING,
)
class DeepseekV2Model(DeepseekV2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV2DecoderLayer`]

    Args:
        config: DeepseekV2Config
    """

    def __init__(self, config: DeepseekV2Config, **kwargs):
        super().__init__(config)
        self.config = config
        self.use_fa_tensor = int(os.getenv("USE_FA_TENSOR", "0"))
        self.next_n = kwargs.get("next_n", 0)
        self.spec_len = kwargs.get("spec_len", 0)
        self.is_spec = kwargs.get("is_spec", 0)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.dynamic_quant_mode = int(os.getenv("QUANT_MODE", "3"))
        assert self.dynamic_quant_mode in [2, 3, 4, 5]
        self.enable_stream = _TODO_REQUIRE_API["enable_stream"]
        self.attn_dp_size = int(os.getenv("ATTN_DP_SIZE", "1"))
        self.attn_tp_size = int(os.getenv("ATTN_TP_SIZE", "1"))
        self.global_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.world_size = int(os.getenv("ATTN_DIES", "1"))
        self.die_num_per_node = int(os.getenv("MA_NUM_GPUS", "16")) if self.world_size >= 16 else int(os.getenv("ATTN_DIES", "1"))
        # self.die_num_per_node = int(os.getenv("MA_NUM_GPUS", "16")) if self.world_size > 16 else int(os.getenv("ATTN_DIES", "1"))
        self.ffn_dies = int(os.getenv("FFN_DIES", "1"))
        self.attn_dies = int(os.getenv("ATTN_DIES", "1"))
        self.div_mode = int(os.getenv("DIV_MODE", "1"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "1"))
        self.input_len = int(os.getenv("INPUT_MAX_LEN", "2048"))
        self.dp_size = int(os.getenv("DP_SIZE", "1"))
        self.experts_tp_size = int(os.getenv("EXPERTS_TP_SIZE", "1"))
        self.dense_tp_size = int(os.getenv("DENSE_TP_SIZE", "16"))
        self.o_proj_tp_size = int(os.getenv("OPROJ_TP_SIZE", "1"))
        self.enable_prefetch = int(os.getenv("ENABLE_PREFETCH", "0"))
        self.remainder_router_expert = int(os.getenv("REMAINDER_ROUTER_EXPERT", "0"))
        self.route_share_on_same_card = int(os.getenv("ROUTE_SHARE_ON_SAME_CARD", "0"))
        self.n_routed_experts_per_rank = int(os.getenv("N_ROUTED_EXPERTS_PER_RANK", "0"))
        self.experts_share_num_copy = kwargs.get("experts_share_num_copy")
        self.shared_expert_rank_num = config.n_shared_experts * self.experts_share_num_copy
        self.ffn_die_for_remaind_expert = 1 if self.remainder_router_expert > 0 else 0
        self.normal_ffn_dies = self.ffn_dies - self.shared_expert_rank_num - self.ffn_die_for_remaind_expert # 去除共享专家和冗余路由专家部署的die
        self.router_expert_num = self.normal_ffn_dies * self.n_routed_experts_per_rank + self.remainder_router_expert
        self.expert_num = self.router_expert_num + self.shared_expert_rank_num
        self.top_k = config.num_experts_per_tok
        if self.o_proj_tp_size > 1:
            assert self.attn_tp_size == 1 # attn_tp_size should be closed if use o_proj_tp
        assert self.experts_tp_size > 0
        self.ep_size = self.world_size // self.experts_tp_size
        self.enable_superkernel = int(os.getenv("ENABLE_SUPERKERNEL", "0")) and exe_mode =="dynamo" and self.dynamic_quant_mode == 5
        self.enable_combine_dequant = int(os.getenv("ENABLE_COMBINE_DEQUANT", "0"))
        self.enable_micro_batch = int(os.getenv("ENABLE_MICRO_BATCH", "0")) and exe_mode =="dynamo"
        self.start_sync = int(os.getenv("ATTN_FFN_START_SYNC", "0"))
        # if self.is_spec:
        #     self.enable_micro_batch = 0
        self.micro_mode = 1
        print("DSV2 model micro mode ", self.micro_mode)
        self.on_cloud = int(os.getenv("ON_CLOUD", "0"))
        self.dense_layer_num = config.first_k_dense_replace
        
        kwargs = {**kwargs,
                 "local_rank": self.local_rank,
                 "rank_offset": self.rank_offset,
                 "global_rank": self.global_rank,
                 "world_size": self.world_size,
                 "die_num_per_node": self.die_num_per_node,
                 "attn_dp_size": self.attn_dp_size,
                 "attn_tp_size": self.attn_tp_size,
                 "ep_size": self.ep_size,
                 "div_mode": self.div_mode,
                 "dynamic_quant_mode": self.dynamic_quant_mode,
                 "batch_size": self.batch_size,
                 "input_len": self.input_len,
                 "enable_stream": self.enable_stream,
                 "enable_prefetch": self.enable_prefetch,
                 "dp_size": self.dp_size,
                 "experts_tp_size": self.experts_tp_size,
                 "dense_tp_size": self.dense_tp_size,
                 "enable_superkernel": self.enable_superkernel,
                 "enable_combine_dequant": self.enable_combine_dequant,
                 "on_cloud": self.on_cloud,
                 "o_proj_tp_size": self.o_proj_tp_size,
                 "enable_micro_batch": self.enable_micro_batch,
                 "micro_mode": self.micro_mode,
                 "router_expert_num": self.router_expert_num,
                 }
        
        self.hidden_size = config.hidden_size
        self.vocab_size_per_rank = config.vocab_size
        fix_rand_seed()
        self.embed_tokens = nn.Embedding(
            self.vocab_size_per_rank, self.hidden_size, config.pad_token_id
        ).to(model_dtype)

        global expert_rank_table, context_holder, schedule_context
        kwargs = self.update_kwargs(config, **kwargs)

        if not self.is_spec:
            moe_all_to_all_group = kwargs.get("moe_all_to_all_group")
            attn_win_size = call_attn_win_size(micro_batch_num=micro_batch_number,
                micro_batch_size=(self.batch_size * self.spec_len // self.attn_dies // micro_batch_number),
                selected_expert_num=self.top_k+1, hidden_size=self.hidden_size)
            layer_out = os.getenv("LAYER_OUT", "FA")
            peer_offset = 0 if layer_out.upper() == "FA" else self.attn_dies
            peer_ranks = [peer_offset + i for i in range(self.ffn_dies)]
            alloc_and_exchange_comm_window(peer_ranks=peer_ranks, win_size=attn_win_size, group=moe_all_to_all_group)

            self.moe_layer_num = config.num_hidden_layers - config.first_k_dense_replace
            local_expert_table = torch.full((1, self.n_routed_experts_per_rank), -1, dtype=torch.int32).unsqueeze(0).npu()
            local_expert_table_all_gather_out = torch.zeros([self.global_world_size, 1, self.n_routed_experts_per_rank], dtype=torch.int32).npu()
            dist.all_gather_into_tensor(local_expert_table_all_gather_out, local_expert_table, group=_world.default_pg)
            expert_rank_table = self.gen_expert_rank_table(local_expert_table_all_gather_out)
            print(f"local_expert_table:{local_expert_table} {local_expert_table_all_gather_out} {expert_rank_table} {expert_rank_table.shape}", flush=True)

            attn_to_ffn_token_size = (self.hidden_size + 4 + 511) // 512 * 512   # 512字节对齐
            ffn_to_attn_token_size = self.hidden_size * 2
            attn_window, attn_window_size = get_local_window()

            print(f"get attn window success, attn_window={attn_window}, attn_window_size={attn_window_size}",
                  flush=True)
            context_holder = torch_npu._afd.create_schedule_context_holder(schedule_mode=1, session_num=self.attn_dies,
                                                                           micro_batch_num=micro_batch_number,
                                                                           micro_batch_size=(self.batch_size * self.spec_len // self.attn_dies // micro_batch_number),
                                                                           selected_expert_num=self.top_k + 1,
                                                                           expert_num=self.expert_num,
                                                                           attn_to_ffn_token_size=attn_to_ffn_token_size,
                                                                           ffn_to_attn_token_size=ffn_to_attn_token_size,
                                                                           attention_window=attn_window,
                                                                           attention_window_size=attn_window_size)
            schedule_context = context_holder.get_schedule_context_tensor()
        kwargs.update({
            "schedule_context": schedule_context,
            "context_holder": context_holder,
            "expert_rank_table": expert_rank_table,
            })
        print("kwargs = {}".format(kwargs), flush=True)

        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(config, layer_idx, **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        _init_rope(self)

    def gen_expert_rank_table(self, local_expert_tables):
        # 全局表初始化： [layer_num, moe_expert_num+share_expert_num, share_copy_num*2+1]
        # 可能存在路由专家存在不均衡部署情况(也即最后一个Die部署路由专家个数小于其他Die)
        expert_num = self.normal_ffn_dies * self.n_routed_experts_per_rank + self.remainder_router_expert + self.config.n_shared_experts 
        expert_rank_table = torch.zeros([1, expert_num, max(self.experts_share_num_copy, 1) * 2 + 1], dtype=torch.int32).npu()

        for rank_id in range(self.global_world_size):
            local_expert_id = 0
            for layer_id in range(1):
                for i in range(self.n_routed_experts_per_rank):
                    global_expert_id = local_expert_tables[rank_id][layer_id][i]
                    if global_expert_id == -1:
                        continue
                    expert_rank_table[layer_id][global_expert_id][0] += 1
                    instance_num = expert_rank_table[layer_id][global_expert_id][0]
                    assert instance_num < (max(self.experts_share_num_copy, 1) * 2 + 1)
                    expert_rank_table[layer_id][global_expert_id][instance_num * 2 - 1] = rank_id
                    expert_rank_table[layer_id][global_expert_id][instance_num * 2] = local_expert_id
                    local_expert_id += 1
        return expert_rank_table

    def update_kwargs(self, config: DeepseekV2Config, **kwargs_tmp):
        layer_out = os.getenv("LAYER_OUT", "FA")
        attn_worker_offset = 0 if layer_out.upper() == "AF" else self.ffn_dies

        # init comm_group
        attn_gather_scatter_group = init_comm_group(
            global_rank=self.global_rank,
            group_num=self.attn_dp_size,
            world_size=self.world_size,
            group_stride=1,
            group_name="attn_gather_scatter_group",
            rank_offset=attn_worker_offset
        )
        print(f"jcz tp_group attn_gather_scatter_group:{attn_gather_scatter_group}")
        dense_gather_group = init_comm_group(
            global_rank=self.global_rank,
            group_num=self.world_size // self.dense_tp_size,
            world_size=self.world_size,
            group_stride=1,
            group_name="dense_gather_group",
            rank_offset=attn_worker_offset
        )
        print(f"jcz tp_group dense_gather_group:{dense_gather_group}")

        if self.o_proj_tp_size > 1 and (self.o_proj_tp_size != self.attn_tp_size):
            o_proj_group = init_comm_group(
                global_rank=self.global_rank,
                group_num=self.world_size // self.o_proj_tp_size,
                world_size=self.world_size,
                group_stride=1,
                group_name="o_proj_group",
                rank_offset=attn_worker_offset
            )
            print(f"jcz tp_group o_proj_group:{o_proj_group}")

        self.mtp_proj_tp_size = 4
        if self.is_spec > 0:
            self.mtp_proj_group = init_comm_group(
                global_rank=self.global_rank,
                group_num=self.world_size // self.mtp_proj_tp_size,
                world_size=self.world_size,
                group_stride=1,
                group_name="mtp_proj_group",
                rank_offset=attn_worker_offset
            )
        else:
            self.mtp_proj_group = None
        print(f"jcz tp_group mtp_proj_group:{self.mtp_proj_group}")

        bs_per_rank = self.batch_size // self.world_size
        bs_per_rank_mb = bs_per_rank // micro_batch_number
        all_gather_dense_shape = [bs_per_rank * self.dense_tp_size * self.spec_len, config.hidden_size]
        reduce_scatter_dense_shape = [bs_per_rank * self.spec_len, config.hidden_size]
        if self.dynamic_quant_mode in [2, 3, 5]:
            all_gather_scale_shape = [self.dense_tp_size * bs_per_rank * self.spec_len]
        else:
            all_gather_scale_shape = []
        
        # attention
        attn_bs = self.batch_size
        attn_input_decode_shape = [attn_bs // self.attn_dp_size * self.spec_len, self.hidden_size]
        attn_input_decode_shape_mb = [attn_bs // micro_batch_number // self.attn_dp_size * self.spec_len, self.hidden_size]
        self.mb_id_one = torch.Tensor([0]).to(torch.int32).npu()
        self.mb_id_two = torch.Tensor([1]).to(torch.int32).npu()
        self.mb_id_three = torch.Tensor([2]).to(torch.int32).npu()

        if (self.o_proj_tp_size > 1) and (self.o_proj_tp_size != self.attn_tp_size):
            self.v_head_dim = config.v_head_dim
            self.num_heads = config.num_attention_heads
            all2all_o_proj_shape = [self.o_proj_tp_size * bs_per_rank * self.spec_len * self.num_heads * self.v_head_dim // self.o_proj_tp_size]
            reduce_scatter_o_proj_shape = [bs_per_rank * self.spec_len, self.hidden_size]
            all2all_o_proj_shape_mb = [self.o_proj_tp_size * bs_per_rank_mb * self.spec_len * self.num_heads * self.v_head_dim // self.o_proj_tp_size]
            reduce_scatter_o_proj_shape_mb = [bs_per_rank_mb * self.spec_len, self.hidden_size]

            kwargs_tmp.update(
                    {
                        "all2all_o_proj_shape": all2all_o_proj_shape,
                        "reduce_scatter_o_proj_shape": reduce_scatter_o_proj_shape,
                        "all2all_o_proj_shape_mb": all2all_o_proj_shape_mb,
                        "reduce_scatter_o_proj_shape_mb": reduce_scatter_o_proj_shape_mb,
                        "o_proj_group": o_proj_group,
                    }
                )

        self.mtp_proj = None
        if self.is_spec:
            self.norm_tsfm = DeepseekV2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.norm_emb = DeepseekV2RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
            self.mtp_proj = NpuLinear(
                    self.hidden_size * 2 // self.mtp_proj_tp_size, self.hidden_size, bias=None
            ).to(model_dtype)

            self.all2all_mtp_proj = torch.zeros([self.batch_size // self.world_size * self.spec_len * self.mtp_proj_tp_size * self.hidden_size * 2 // self.mtp_proj_tp_size], dtype=model_dtype).npu()
            self.mtp_proj_rs_out = torch.zeros([self.batch_size // self.world_size * self.spec_len, self.hidden_size], dtype=model_dtype).npu()

        kwargs_tmp.update(
                        {
                            "attn_gather_scatter_group": attn_gather_scatter_group,
                            "dense_gather_group": dense_gather_group,
                            "all_gather_dense_shape": all_gather_dense_shape,
                            "all_gather_scale_shape": all_gather_scale_shape,
                            "reduce_scatter_dense_shape": reduce_scatter_dense_shape,
                            "attn_input_decode_shape": attn_input_decode_shape,
                            "attn_input_decode_shape_mb": attn_input_decode_shape_mb,
                        }
                    )               
        return kwargs_tmp
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward_decode_microbatch_new(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        layer_num = len(self.layers)
        for layer_idx in range(self.dense_layer_num):
            residual, hidden_states, hidden_states_for_prefetch = self.layers[layer_idx].forward_decode(
                hidden_states,
                kv_len,
                actual_seq_lengths_kv,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                past_residual=residual
            )

        # MOE层，开启microbatch
        bsz, q_len, hidden_size = hidden_states.size()
        hidden_states_batch_1, hidden_states_batch_2, hidden_states_batch_3 = one_third_batch(hidden_states, 0)
        residual_1, residual_2, residual_3 = one_third_batch(residual, 0)
        cos, sin = cos_sin
        cos_1, cos_2, cos_3 = one_third_batch(cos, 0)
        sin_1, sin_2, sin_3 = one_third_batch(sin, 0)
        cos_sin_1 = (cos_1, sin_1)
        cos_sin_2 = (cos_2, sin_2)
        cos_sin_3 = (cos_3, sin_3)
        kv_len_size = kv_len.size()[0]
        kv_len_batch_1, kv_len_batch_2, kv_len_batch_3 = one_third_batch(kv_len, 0)
        # 双路共用同一份kvcache，偏移通过kv_len决定，当前kv_len直接用初始化的值
        past_key_values_1 = past_key_values
        past_key_values_2 = past_key_values
        past_key_values_3 = past_key_values
        if not self.use_fa_tensor:
            actual_seq_lengths_kv_1 = actual_seq_lengths_kv[:(kv_len_size // 3)]
            actual_seq_lengths_kv_2 = actual_seq_lengths_kv[(kv_len_size // 3):(kv_len_size * 2 // 3)]
            actual_seq_lengths_kv_3 = actual_seq_lengths_kv[(kv_len_size * 2 // 3):]
        else:
            actual_seq_lengths_kv_1, actual_seq_lengths_kv_2, actual_seq_lengths_kv_3 = one_third_batch(actual_seq_lengths_kv, 0)

        if self.start_sync and not self.is_spec:
            # 临时方案，用于全局强制同步
            local_tensor = torch.ones([1], dtype=torch.int32).npu()
            local_all_gather_out = torch.zeros([self.global_world_size], dtype=torch.int32).npu()
            dist.all_gather_into_tensor(local_all_gather_out, local_tensor)
            hidden_states_batch_1 = tng.scope.npu_wait_tensor(hidden_states_batch_1, local_all_gather_out)

        # 双路batch1的前半层。Dense不开双路时，这里不能被掩盖，用满核
        residual_1, hidden_states_batch_1_tmp = self.layers[self.dense_layer_num].forward_decode_part_one(
            hidden_states_batch_1,
            kv_len_batch_1,
            actual_seq_lengths_kv_1,
            cos_sin=cos_sin_1,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values_1,
            past_residual=residual_1
        )
        ori_aic = 24
        ori_aiv = 48
        new_aic_part_1 = 16
        new_aiv_part_1 = 32
        new_aic_part_2 = ori_aic - new_aic_part_1
        new_aiv_part_2 = ori_aiv - new_aiv_part_1

        combine_kwargs_3 = None

        combine_out_2 = None
        for layer_idx in range(self.dense_layer_num, layer_num):
            if layer_idx < (layer_num - 1):
                func_batch_1_part_1 = self.layers[layer_idx + 1].forward_decode_part_one
            func_batch_1_part_2 = self.layers[layer_idx].forward_decode_part_two
            func_batch_2_part_1 = self.layers[layer_idx].forward_decode_part_one
            func_batch_2_part_2 = self.layers[layer_idx].forward_decode_part_two
            func_batch_3_part_1 = self.layers[layer_idx].forward_decode_part_one
            func_batch_3_part_2 = self.layers[layer_idx].forward_decode_part_two

            with NpuStreamSwitch(True, '22'):
                with tng.scope.limit_core_num(new_aic_part_2, new_aiv_part_2):
                    residual_1, combine_out_3, _, attn2ffn_out_1, combine_kwargs_1 = func_batch_1_part_2(
                        hidden_states_batch_1_tmp,
                        bsz // micro_batch_number,  # bsz is fullbatch batch size
                        q_len,
                        hidden_size,
                        residual_1,
                        batch_id=1,
                        mb_id=self.mb_id_one,
                        last_combine_kwargs=combine_kwargs_3,
                        combine_out=combine_out_2
                    )

            hidden_states_batch_2 = hidden_states_batch_2 if layer_idx == self.dense_layer_num else combine_out_2
            residual_2, hidden_states_batch_2_tmp = func_batch_2_part_1(
                hidden_states_batch_2,
                kv_len_batch_2,
                actual_seq_lengths_kv_2,
                cos_sin=cos_sin_2,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values_2,
                past_residual=residual_2,
                batch_id=2,
                q_len=q_len,
                hidden_size=hidden_size,
            )

            with NpuStreamSwitch(True, '22'):
                with tng.scope.limit_core_num(new_aic_part_2, new_aiv_part_2):
                    residual_2, combine_out_1, _, attn2ffn_out_2, combine_kwargs_2 = func_batch_2_part_2(
                        hidden_states_batch_2_tmp,
                        bsz // micro_batch_number,  # bsz is fullbatch batch size
                        q_len,
                        hidden_size,
                        residual_2,
                        batch_id=2,
                        mb_id=self.mb_id_two,
                        last_combine_kwargs=combine_kwargs_1,
                        combine_out=combine_out_3
                    )

            hidden_states_batch_3 = hidden_states_batch_3 if layer_idx == self.dense_layer_num else combine_out_3
            residual_3, hidden_states_batch_3_tmp = func_batch_3_part_1(
                hidden_states_batch_3,
                kv_len_batch_3,
                actual_seq_lengths_kv_3,
                cos_sin=cos_sin_3,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values_3,
                past_residual=residual_3,
                batch_id=3,
                q_len=q_len,
                hidden_size=hidden_size,
            )

            if self.enable_prefetch:
                if layer_idx < (layer_num - 1):
                    torch_npu.npu_prefetch(self.layers[layer_idx + 1].self_attn.q_a_proj.weight.data, hidden_states_batch_3_tmp, PREFETCH_SIZE, 0)
                    torch_npu.npu_prefetch(self.layers[layer_idx + 1].self_attn.kv_a_proj_with_mqa.weight.data, hidden_states_batch_3_tmp, PREFETCH_SIZE, 0)
                    torch_npu.npu_prefetch(self.layers[layer_idx + 1].self_attn.q_b_proj.weight.data, hidden_states_batch_3_tmp, PREFETCH_SIZE, 0)
                    torch_npu.npu_prefetch(self.layers[layer_idx + 1].self_attn.kv_b_proj_w_k.data, hidden_states_batch_3_tmp, PREFETCH_SIZE, 0)

            with NpuStreamSwitch(True, '22'):
                if layer_idx == (layer_num - 1):
                    new_aic_part_2 = 24
                    new_aiv_part_2 = 48
                with tng.scope.limit_core_num(new_aic_part_2, new_aiv_part_2):
                    residual_3, combine_out_2, _, attn2ffn_out_3, combine_kwargs_3 = func_batch_3_part_2(
                        hidden_states_batch_3_tmp,
                        bsz // micro_batch_number,  # bsz is fullbatch batch size
                        q_len,
                        hidden_size,
                        residual_3,
                        batch_id=3,
                        mb_id=self.mb_id_three,
                        last_combine_kwargs=combine_kwargs_2,
                        combine_out=combine_out_1
                    )

            if layer_idx < (layer_num - 1):
                residual_1, hidden_states_batch_1_tmp = func_batch_1_part_1(
                    combine_out_1,
                    kv_len_batch_1,
                    actual_seq_lengths_kv_1,
                    cos_sin=cos_sin_1,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values_1,
                    past_residual=residual_1,
                    batch_id=1,
                    q_len=q_len,
                    hidden_size=hidden_size,
                )

        if combine_kwargs_3 is not None:
            tng.scope.npu_wait_tensor(combine_kwargs_3.get("schedule_context"), combine_out_2)
            combine_out_3, _ = torch_npu.npu_attention_worker_combine(**combine_kwargs_3)
            combine_out_3 = combine_out_3.view(-1, q_len, hidden_size)
        hidden_states = torch.cat([combine_out_1, combine_out_2, combine_out_3], dim=0)
        if residual_1 is not None and residual_2 is not None and residual_3 is not None:
            residual = torch.cat([residual_1, residual_2, residual_3], dim=0)
        outputs = (residual, hidden_states)
        return outputs

    def forward_decode_microbatch(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        layer_num = len(self.layers)
        for layer_idx in range(self.dense_layer_num):
            residual, hidden_states, hidden_states_for_prefetch = self.layers[layer_idx].forward_decode(
                hidden_states,
                kv_len,
                actual_seq_lengths_kv,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                past_residual=residual
            )

        # MOE层，开启microbatch
        bsz, q_len, hidden_size = hidden_states.size()
        hidden_states_batch_1, hidden_states_batch_2 = half_batch(hidden_states, 0)
        residual_1, residual_2 = half_batch(residual, 0)
        cos, sin = cos_sin
        cos_1, cos_2 = half_batch(cos, 0)
        sin_1, sin_2 = half_batch(sin, 0)
        cos_sin_1 = (cos_1, sin_1)
        cos_sin_2 = (cos_2, sin_2)
        kv_len_size = kv_len.size()[0]
        kv_len_batch_1, kv_len_batch_2 = half_batch(kv_len, 0)
        # 双路共用同一份kvcache，偏移通过kv_len决定，当前kv_len直接用初始化的值
        past_key_values_1 = past_key_values
        past_key_values_2 = past_key_values
        if not self.use_fa_tensor:
            actual_seq_lengths_kv_1 = actual_seq_lengths_kv[:(kv_len_size // 2)]
            actual_seq_lengths_kv_2 = actual_seq_lengths_kv[(kv_len_size // 2):]
        else:
            actual_seq_lengths_kv_1, actual_seq_lengths_kv_2 = half_batch(actual_seq_lengths_kv, 0)
        # 双路batch1的前半层。Dense不开双路时，这里不能被掩盖，用满核
        residual_1, hidden_states_batch_1_tmp = self.layers[self.dense_layer_num].forward_decode_part_one(
            hidden_states_batch_1,
            kv_len_batch_1,
            actual_seq_lengths_kv_1,
            cos_sin=cos_sin_1,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values_1,
            past_residual=residual_1
        )
        new_aic = 16
        new_aiv = 32
        # 开启双路，使用半核。FA使用二进制复用，全局指定不能继续生效，这里重新设置
        with tng.scope.limit_core_num(new_aic, new_aiv):
            for layer_idx in range(self.dense_layer_num, layer_num):
                if layer_idx < (layer_num - 1):
                    func_batch_1_part_1 = self.layers[layer_idx + 1].forward_decode_part_one
                func_batch_1_part_2 = self.layers[layer_idx].forward_decode_part_two
                func_batch_2_part_1 = self.layers[layer_idx].forward_decode_part_one
                func_batch_2_part_2 = self.layers[layer_idx].forward_decode_part_two

                residual_2, hidden_states_batch_2_tmp = func_batch_2_part_1(
                    hidden_states_batch_2,
                    kv_len_batch_2,
                    actual_seq_lengths_kv_2,
                    cos_sin=cos_sin_2,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values_2,
                    past_residual=residual_2,
                    batch_id=2,
                )

                with NpuStreamSwitch(True, '22'):
                    residual_1, hidden_states_batch_1, _ = func_batch_1_part_2(
                        hidden_states_batch_1_tmp,
                        bsz // 2,  # bsz is fullbatch batch size
                        q_len,
                        hidden_size,
                        residual_1,
                        batch_id=1,
                        mb_id=self.mb_id_one
                    )

                if layer_idx < (layer_num - 1):
                    residual_1, hidden_states_batch_1_tmp = func_batch_1_part_1(
                        hidden_states_batch_1,
                        kv_len_batch_1,
                        actual_seq_lengths_kv_1,
                        cos_sin=cos_sin_1,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values_1,
                        past_residual=residual_1,
                        batch_id=1,
                    )

                    with NpuStreamSwitch(True, '22'):
                        residual_2, hidden_states_batch_2, _ = func_batch_2_part_2(
                            hidden_states_batch_2_tmp,
                            bsz // 2,  # bsz is fullbatch batch size
                            q_len,
                            hidden_size,
                            residual_2,
                            batch_id=2,
                            mb_id=self.mb_id_two
                        )

        # 双路batch2的后半层，不能被llm_head掩盖，用满核
        with tng.scope.limit_core_num(new_aic, new_aiv):
            with NpuStreamSwitch(True, '22'):
                residual_2, hidden_states_batch_2, _ = func_batch_2_part_2(
                    hidden_states_batch_2_tmp,
                    bsz // 2,  # bsz is fullbatch batch size
                    q_len,
                    hidden_size,
                    residual_2,
                    batch_id=2,
                    mb_id=self.mb_id_two
                )

        hidden_states = torch.cat([hidden_states_batch_1, hidden_states_batch_2], dim=0)
        if residual_1 is not None and residual_2 is not None:
            residual = torch.cat([residual_1, residual_2], dim=0)
        outputs = (residual, hidden_states)
        return outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_tsfm: torch.FloatTensor = None
    ):
        batch_size, seq_length = input_ids.shape
        past_key_values_length = past_key_values[0][0].size(-2)

        if (position_ids is None) or (seq_length > 1):
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=input_ids.device
            )
        position_ids = position_ids.view(-1, seq_length).long()

        # embedding DP for input_ids, for mtp cases, chunk tsfm to DP
        if self.world_size > 1:
            new_input_ids = input_ids.flatten()
            tsfm_seq_length = seq_length
            inputs_embeds = self.embed_tokens(new_input_ids)  # (bs*qlen/world_size, hidden_size)
            if self.is_spec:
                input_ids_tsfm = input_ids_tsfm.view(-1, self.hidden_size)
                if seq_length > 1 and (seq_length != self.spec_len):
                    input_ids_tsfm = torch.chunk(input_ids_tsfm, self.world_size, dim=0)[self.global_rank]
                input_ids_tsfm = input_ids_tsfm.view(-1, tsfm_seq_length, self.hidden_size)

            inputs_embeds = inputs_embeds.view(-1, seq_length, self.hidden_size)
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        logger.info(f"forward 1 hidden_states:{hidden_states.shape}") if exe_mode != "dynamo" else None

        if self.is_spec:
            hidden_states_tsfm = self.norm_tsfm(input_ids_tsfm)
            hidden_embeds = self.norm_emb(hidden_states)
            hidden_mtp_in = torch.cat([hidden_states_tsfm, hidden_embeds], dim=-1) # concat hidden, output B,S,2H

            hidden_mtp_in = hidden_mtp_in.view(-1, self.mtp_proj_tp_size, self.hidden_size * 2 // self.mtp_proj_tp_size).transpose(0, 1).contiguous()
            dist.all_to_all_single(self.all2all_mtp_proj, hidden_mtp_in.flatten(), group=self.mtp_proj_group)
            hidden_mtp_in = self.all2all_mtp_proj.view(-1, self.hidden_size * 2 // self.mtp_proj_tp_size)
            hidden_states = self.mtp_proj(hidden_mtp_in) #  (4*B*S, H/2) --> (4*B*S, H)
            dist.reduce_scatter_tensor(self.mtp_proj_rs_out, hidden_states, group=self.mtp_proj_group)
            hidden_states = self.mtp_proj_rs_out.view(-1, seq_length, self.hidden_size)  # (B*S, H)

        cos_sin = self.rotary_emb(batch_size * self.attn_tp_size, seq_length, kv_len, self.config.max_position_embeddings)
        residual = None
        logger.info(f"forward 2 hidden_states:{hidden_states.shape}") if exe_mode != "dynamo" else None
        if not self.enable_micro_batch:
            for decoder_layer in self.layers:
                residual, hidden_states, hidden_states_for_prefetch = decoder_layer.forward_decode(
                    hidden_states,
                    kv_len,
                    actual_seq_lengths_kv,
                    cos_sin=cos_sin,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    past_residual=residual,
                    mb_id=self.mb_id_one
                )
        else:
            micro_batch_func = self.forward_decode_microbatch_new if micro_batch_number == 3 else self.forward_decode_microbatch
            residual, hidden_states = micro_batch_func(
                hidden_states,
                kv_len,
                actual_seq_lengths_kv,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                residual=residual
            )
            

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class DeepseekV2ForCausalLM(DeepseekV2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.global_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.world_size = int(os.getenv("ATTN_DIES", "1"))
        self.ffn_dies = int(os.getenv("FFN_DIES", "1"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.batch_size = int(os.getenv("BATCH_SIZE", "1"))
        self.input_len = int(os.getenv("INPUT_MAX_LEN", "2048"))
        self.dynamic_quant_mode = int(os.getenv("QUANT_MODE", "3"))
        self.experts_tp_size = int(os.getenv("EXPERTS_TP_SIZE", "1"))
        self.die_num_per_node = int(os.getenv("MA_NUM_GPUS", "16")) if self.world_size >= 16 else int(os.getenv("ATTN_DIES", "1"))
        # self.die_num_per_node = int(os.getenv("MA_NUM_GPUS", "16")) if self.world_size > 16 else int(os.getenv("ATTN_DIES", "1"))
        self.attn_dp_size = int(os.getenv("ATTN_DP_SIZE", "1"))
        self.attn_tp_size = int(os.getenv("ATTN_TP_SIZE", "1"))
        self.next_n = int(os.getenv("NEXT_N", "0")) if not hasattr(config, "next_n") else config.next_n
        self.is_spec = config.is_spec
        self.spec_len = self.next_n + 1
        self.enable_prefetch = int(os.getenv("ENABLE_PREFETCH", "0"))
        self.enable_cache_compile = int(os.getenv("ENABLE_CACHE_COMPILE", "0")) and exe_mode =="dynamo"
        self.enable_prof = int(os.getenv("ENABLE_PROFILE", "0"))
        self.use_fa_tensor = int(os.getenv("USE_FA_TENSOR", "0"))
        self.enable_fa_quant = int(os.getenv("ENABLE_FA_QUANT", "0")) and exe_mode =="dynamo"
        self.ep_size = self.global_world_size // self.experts_tp_size

        # set_expert
        enable_expert_adpt = int(os.getenv("ENABLE_EXPERT_ADPT", "0"))
        experts_share_num_copy = int(os.getenv("EXPERTS_SHARE_NUM_COPY", "1"))
        n_routed_experts_per_rank = int(os.getenv("N_ROUTED_EXPERTS_PER_RANK", "0"))
        route_share_on_same_card = int(os.getenv("ROUTE_SHARE_ON_SAME_CARD", "0"))
        kwargs = self.compute_expert_conf(config, enable_expert_adpt, n_routed_experts_per_rank,
                                          experts_share_num_copy, route_share_on_same_card)

        print("config.n_routed_experts={}, config.n_shared_experts={}, config.num_experts_per_tok={}".format(
               config.n_routed_experts, config.n_shared_experts, config.num_experts_per_tok
              ), flush=True)
        kwargs.update({
                        "next_n": self.next_n,
                        "spec_len": self.spec_len,
                        "is_spec": self.is_spec
                      })

        # init all gather out tensor for lm_head 16 TP
        self.lm_head_tp = self.die_num_per_node # 16 TP in a node server
        # self.lm_head_tp = 1 # 16 TP in a node server
        self.lm_head_dp = self.world_size // self.lm_head_tp
        assert self.lm_head_dp > 0
        self.all_gather_out_incre = torch.zeros([self.batch_size // self.lm_head_dp, self.spec_len, config.hidden_size],
                                                dtype=config.torch_dtype, device="npu")

        self.vocab_size = config.vocab_size
        print(f"self.vocab_size:{self.vocab_size} self.lm_head_tp:{self.lm_head_tp}")
        assert self.vocab_size % self.lm_head_tp == 0
        self.vocab_size_per_rank = self.vocab_size // self.lm_head_tp
        self.all_gather_shape = [self.batch_size // self.world_size * self.die_num_per_node * (self.next_n + 1)]
        self.all_gather_max_out_incre_final = torch.zeros(self.all_gather_shape, dtype=config.torch_dtype, device="npu")
        self.all_gather_index_out_incre_final = torch.zeros(self.all_gather_shape, dtype=torch.int32, device="npu")
        self.lm_head_rank_offset = (self.local_rank % self.lm_head_tp) * self.vocab_size_per_rank
        self.lm_head = NpuLinear(config.hidden_size, self.vocab_size_per_rank, bias=False).to(model_dtype)

        global moe_all_to_all_group_name, moe_all_to_all_group
        if not self.is_spec:
            print("attn begin to init moe_all_to_all_group", flush=True)
            moe_all_to_all_group = init_comm_group(
                global_rank=self.global_rank,
                group_num=self.experts_tp_size,
                world_size=self.global_world_size,
                group_stride=self.experts_tp_size,
                group_name="moe_all_to_all_group_name",
            )

            moe_all_to_all_group_name = moe_all_to_all_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.global_rank)
            print("attn success to init moe_all_to_all_group", flush=True)

        layer_out = os.getenv("LAYER_OUT", "FA")
        attn_worker_offset = 0 if layer_out.upper() == "AF" else self.ffn_dies
        print(f"self.die_num_per_node:{self.die_num_per_node} self.world_size:{self.world_size} "
              f"self.global_world_size:{self.global_world_size} self.lm_head_tp:{self.lm_head_tp}")
        if self.global_world_size > 16:
            self.vocab_ranks_comm_group = init_comm_group(
                global_rank=self.global_rank,
                group_num=self.world_size // self.lm_head_tp,
                world_size=self.world_size,
                group_stride=1,
                group_name="vocab_ranks_comm_group",
                rank_offset=attn_worker_offset
            )
        else:
            self.vocab_ranks_comm_group = dist.new_group(range(self.ffn_dies, self.global_world_size))
        print(f"self.vocab_ranks_comm_group:{self.vocab_ranks_comm_group} "
              f"world_size:{dist.get_world_size(self.vocab_ranks_comm_group)}")
        kwargs.update({
                        "vocab_ranks_comm_group": self.vocab_ranks_comm_group,
                        "moe_all_to_all_group_name": moe_all_to_all_group_name,
                        "moe_all_to_all_group": moe_all_to_all_group
                      }
                     )

        self.model = DeepseekV2Model(config, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

        if self.enable_cache_compile == 1:
            tng_config = tng.CompilerConfig()
            tng_config.experimental_config.frozen_parameter = True
            tng_config.experimental_config.tiling_schedule_optimize = True
            tng_config.experimental_config.topology_sorting_strategy = "StableRDFS"
            cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.getenv("CASE_NAME", "./"))
            print(f"begin cache_compile dir:{cache_dir}")
            self.cached_spec_decode = tng.inference.cache_compile(self.spec_decode, config=tng_config, cache_dir=cache_dir, ge_cache=True)
            self.cached_main_decode = tng.inference.cache_compile(self.main_decode, config=tng_config, cache_dir=cache_dir, ge_cache=True)
            print(f"end cache_compile dir:{cache_dir}")

        # Initalize kvcache
        self.kv_cache = self.init_cache()
    
    def compute_expert_conf(self, config, enable_expert_adpt, n_routed_experts_per_rank, 
                            experts_share_num_copy, route_share_on_same_card):
        n_shared_experts_per_rank = config.n_shared_experts or 0
        kwargs = {
                    "n_routed_experts_per_rank": n_routed_experts_per_rank,
                    "n_shared_experts_per_rank": n_shared_experts_per_rank,
                    "experts_share_num_copy": experts_share_num_copy,
                    "route_ep_size": config.n_routed_experts,
                    "route_share_on_same_card": route_share_on_same_card,
                }
        return kwargs

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def spec_decode(self, *args, **kwargs):
        if exe_mode == "dynamo":
            return self._forward(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def main_decode(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_tsfm: torch.FloatTensor = None,
        *args,
        **kwargs,
    ):
        if self.enable_cache_compile == 0:
            forward_decode = self.spec_decode if self.is_spec else self.main_decode
        else:
            forward_decode = self.cached_spec_decode if self.is_spec else self.cached_main_decode

        return forward_decode(input_ids, kv_len, actual_seq_lengths_kv, attention_mask, position_ids, past_key_values,
                                inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict,
                                input_ids_tsfm, *args, **kwargs)

    def _forward(
        self,
        input_ids: torch.LongTensor = None,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        input_ids_tsfm: torch.FloatTensor = None,
        *args,
        **kwargs
    ):
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV2ForCausalLM

        >>> model = DeepseekV2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            input_ids_tsfm=input_ids_tsfm
        )
        
        hidden_states = outputs
        seq_length = hidden_states.size(1)
        if self.lm_head_tp > 1:
            if self.enable_prefetch:
                torch_npu.npu_prefetch(self.lm_head.weight.data, hidden_states, LM_HEAD_PREFETCH_SIZE, 0)
            # (bs / dp / tp, h) -> (bs / dp, h)
            dist.all_gather_into_tensor(self.all_gather_out_incre, hidden_states, group=self.vocab_ranks_comm_group)
            hidden_states = self.all_gather_out_incre
        hidden_states_tsfm = hidden_states

        if exe_mode == "dynamo":
            with tng.scope.limit_core_num(24, 48):
                logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states)

        max_logits_per_rank, max_logits_index_per_rank = torch.max(logits, dim=-1)
        max_logits_index_per_rank = max_logits_index_per_rank.to(torch.int32)
        max_logits_index_per_rank += self.lm_head_rank_offset

        if self.lm_head_tp > 1:
            max_logits_per_rank = max_logits_per_rank.view(-1)
            dist.all_to_all_single(self.all_gather_max_out_incre_final, max_logits_per_rank, group=self.vocab_ranks_comm_group)
            max_logits = self.all_gather_max_out_incre_final.reshape(self.die_num_per_node, -1)
            max_logits = max_logits.transpose(0,1).reshape(-1, seq_length, self.die_num_per_node)

            max_logits_index_per_rank = max_logits_index_per_rank.view(-1)
            dist.all_to_all_single(self.all_gather_index_out_incre_final, max_logits_index_per_rank, group=self.vocab_ranks_comm_group)
            max_logits_index = self.all_gather_index_out_incre_final.reshape(self.die_num_per_node, -1)
            max_logits_index = max_logits_index.transpose(0,1).reshape(-1, seq_length, self.die_num_per_node)

        max_logits = max_logits.float()

        if self.next_n > 0:
            outputs = (max_logits, max_logits_index, hidden_states_tsfm)
        else:
            outputs = max_logits, max_logits_index

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "kv_len": kv_len,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "input_ids_tsfm": hidden_states_tsfm,
            "generate_ids": kwargs.get("generate_ids"),
            "input_lens": kwargs.get("input_lens")
        }

        output_dict = self.model_output_update(outputs, model_inputs, self.next_n)
        return output_dict, logits

    def init_cache(self, device="npu"):
        cache_seq_len = self.config.max_position_embeddings
        dtype = self.config.torch_dtype

        past_key_values = ()

        if _TODO_REQUIRE_API["enable_pa"]:
            self.max_len = _PAGE_ATTENTION_SETTING["max_length"]
            self.block_size = _PAGE_ATTENTION_SETTING["block_size"]
            self.cache_len = self.max_len // self.block_size
            num_block = math.ceil(self.max_len / self.block_size) * self.batch_size // self.attn_dp_size
            cache_nope_shape = (
                            num_block,
                            _PAGE_ATTENTION_SETTING["block_size"],
                            self.config.kv_lora_rank
            )
            cache_rope_shape = (
                            num_block,
                            _PAGE_ATTENTION_SETTING["block_size"],
                            self.config.qk_rope_head_dim
            )

        else:
            cache_nope_shape = (
                            self.batch_size,
                            cache_seq_len,
                            1,
                            self.config.kv_lora_rank
                        )
            cache_rope_shape = (
                            self.batch_size,
                            cache_seq_len,
                            1,
                            self.config.qk_rope_head_dim
                        )

        for i in range(self.config.num_hidden_layers):
            nope_cache = torch.zeros(cache_nope_shape, dtype=torch.int8 if (self.dynamic_quant_mode == 2 or self.enable_fa_quant) else dtype, device=device)
            rope_cache = torch.zeros(cache_rope_shape, dtype=dtype, device=device)
            past_key_values += ((nope_cache, rope_cache),)

        return past_key_values

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_prefill=None,
        kv_len=None,
        share_mask_tril=None,
        world_size=1,
        cur_position_id=None,
        input_ids_tsfm=None,
        generate_ids=None,
        actual_seq_lengths_kv=None,
        **kwargs
    ):
        if past_key_values is None:
            past_key_values = self.init_cache()

        if self.next_n > 0:
            pass
        else:
            attention_mask = None
        position_ids = kv_len

        model_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "kv_len": kv_len,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
                "input_ids_tsfm": input_ids_tsfm,
                "generate_ids": generate_ids,
                "input_lens": kwargs.get("input_lens")
            }
        # self.model_input_dict = model_inputs
        return model_inputs

    def model_output_update(self, outputs, input_dict, next_n=0):
        die_batch = self.batch_size // self.world_size
        input_dict['input_lens'] = input_dict['input_lens'] + 1
        input_dict['kv_len'] += 1

        if next_n > 0:
            max_logits, max_logits_index, tsfm_logits = outputs
        else:
            max_logits, max_logits_index = outputs
            input_dict['attention_mask'] = None
        input_dict['past_key_values'] = input_dict.get("past_key_values")
        input_dict['share_mask_tril'] = None

        if self.next_n > 0:
            max_logits_index = max_logits_index[0:die_batch, -1]
            max_logits = max_logits[0:die_batch, -1]
        else:
            max_logits_index = max_logits_index.squeeze(1)
            max_logits = max_logits.squeeze(1)

        gather_index = torch.argmax(max_logits, dim=-1, keepdim=True)
        next_tokens = torch.gather(max_logits_index, 1, gather_index)

        if next_n > 0:
            tsfm_logits = tsfm_logits[0:die_batch, -(next_n + 1):, :]
            input_dict['input_ids_tsfm'] = tsfm_logits

            # history ids update
            old_generate_ids = input_dict['generate_ids'][0:die_batch, 1:]
            input_dict['generate_ids'] = torch.cat([old_generate_ids, next_tokens], dim=-1)

            # new_token update
            next_tokens = input_dict['generate_ids'][0:die_batch, -(next_n + 1):]

        input_dict['input_ids'] = next_tokens

        return input_dict

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


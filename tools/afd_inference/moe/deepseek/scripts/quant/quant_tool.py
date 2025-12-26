import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch_npu

from models.global_setting import _TODO_REQUIRE_API, PREFETCH_SIZE
from models.common import quant_dtype, model_dtype, apply_quant, exe_mode, NpuLinear, fix_rand_seed, exe_mode

dynamic_quant_mode = int(os.getenv("QUANT_MODE", "3"))


def quantizable_linear(origin_linear: NpuLinear):
    if origin_linear is None:
        return None
    if dynamic_quant_mode > 0:
        linear_cls = DynamicA8W8Linear(origin_linear)
    else:
        linear_cls = origin_linear
    return linear_cls

def quantizable_pertensor_linear(origin_linear: NpuLinear):
    if origin_linear is None:
        return None
    linear_cls = PertensorA8W8Linear(origin_linear)
    return linear_cls

def not_quant(origin_linear):
    return origin_linear


def quantizable_gmm(origin_gmm):
    if origin_gmm is None:
        return None
    gmm_cls = origin_gmm
    if dynamic_quant_mode:
        gmm_cls = DeepseekV2DynamicA8W8GMM(origin_gmm)
    else:
        gmm_cls = origin_gmm
    return gmm_cls


def replace_linear_deepseek(model):
    enable_fa_quant = int(os.getenv("ENABLE_FA_QUANT", "0")) and exe_mode == "dynamo"
    if dynamic_quant_mode != 5 or enable_fa_quant:
        return replace_linear_general(model)
    else:
        return replace_quant_layer_x(model)


def replace_linear_general(model):
    use_merge = int(os.getenv("USE_MERGE", "0"))

    # replace linear matmul within MLA layers
    if dynamic_quant_mode in [3, 5]:
        attn_quant = quantizable_linear
    elif dynamic_quant_mode == 4:
        attn_quant = not_quant
    elif dynamic_quant_mode == 2:
        attn_quant = quantizable_pertensor_linear
    else:
        raise ValueError("dynamic_quant_mode only supports 2, 3, 4 or 5!")

    for layer_idx, layer in enumerate(model.model.layers):
        if use_merge:
            layer.self_attn.merged_qkv_a_proj = attn_quant(layer.self_attn.merged_qkv_a_proj)
        else:
            layer.self_attn.q_a_proj = attn_quant(layer.self_attn.q_a_proj)
            layer.self_attn.kv_a_proj_with_mqa = attn_quant(layer.self_attn.kv_a_proj_with_mqa)

        layer.self_attn.q_b_proj = attn_quant(layer.self_attn.q_b_proj)
        layer.self_attn.kv_b_proj = attn_quant(layer.self_attn.kv_b_proj)
        layer.self_attn.o_proj = attn_quant(layer.self_attn.o_proj)

        # replace linear matmul for gmm and mlp in moe layers
        if layer_idx >= model.config.first_k_dense_replace \
            and layer_idx % model.config.moe_layer_freq == 0:
            layer.mlp.experts = quantizable_gmm(layer.mlp.experts)
            if model.config.n_shared_experts is not None and layer.mlp.shared_experts is not None:
                if int(os.getenv("ENABLE_GMM_IN_SHARED", "0")):
                    layer.mlp.shared_experts = quantizable_gmm(layer.mlp.shared_experts)
                else:
                    layer.mlp.shared_experts.merge_up_gate_proj = quantizable_linear(layer.mlp.shared_experts.merge_up_gate_proj)
                    layer.mlp.shared_experts.down_proj = quantizable_linear(layer.mlp.shared_experts.down_proj)
        else:  # dense layers
            layer.mlp.merge_up_gate_proj = quantizable_linear(layer.mlp.merge_up_gate_proj)
            layer.mlp.down_proj = quantizable_linear(layer.mlp.down_proj)
    return model


def replace_quant_layer_x(model):
    assert int(os.getenv("USE_MERGE", "0")) == 0
    assert model.config.n_shared_experts is not None
    assert model.config.moe_layer_freq == 1

    for layer_idx, layer in enumerate(model.model.layers):
        # MLA: quantize up_proj and o_proj
        layer.self_attn.q_b_proj = quantizable_linear(layer.self_attn.q_b_proj)
        if layer_idx < model.config.first_k_dense_replace:
            continue
        layer.self_attn.o_proj = quantizable_linear(layer.self_attn.o_proj)

    return model


class DynamicA8W8Linear(nn.Module):
    def __init__(self, origin_linear: NpuLinear, offset=False, out_dtype=model_dtype):
        super().__init__()
        # prefill(eager)场景下的qmm不走weightNz
        self.out_features, self.in_features = origin_linear.weight.size()
        self.bias = origin_linear.bias
        self.weight = origin_linear.weight
        self.weight.requires_grad = False
        self.weight.data = apply_quant(self.weight.data)
        epsilon = 1e-2
        fix_rand_seed()
        self.in_scale = Parameter(1 / (torch.rand(self.in_features, dtype=model_dtype) * (1 - epsilon) + epsilon), requires_grad=False)
        self.out_scale = Parameter(torch.rand(self.out_features, dtype=quant_dtype) * (1 - epsilon) + epsilon, requires_grad=False)
        if offset:
            self.offset = Parameter(torch.rand(self.out_features, dtype=quant_dtype), requires_grad=False)
        else:
            self.offset = None
        self.out_dtype = out_dtype

    def forward(self, x, is_quant=False, dynamic_scale=None, throw_dequant=False, throw_quant=False, out_shape=None):
        if throw_quant:
            x = torch_npu.npu_quant_matmul(x, self.weight, self.out_scale, pertoken_scale=dynamic_scale, bias=self.bias,
                                    output_dtype=self.out_dtype)
            x = x.view(*out_shape[:-1], self.out_features)
            return x

        if is_quant and dynamic_scale is not None:
            x_scale = dynamic_scale
        else:
            # generate dynamic quantization scale
            x, x_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=self.in_scale)

        out_shape = x.size()[:-1] + (self.out_features, )
        x = x.view(-1, x.size(-1))
        x_scale = x_scale.view(-1)

        if throw_dequant:
            x = torch_npu.npu_quant_matmul(x, self.weight, self.out_scale, bias=self.bias,
                                           output_dtype=torch.int32)
            return x, self.out_scale, x_scale, out_shape
        else:
            x = torch_npu.npu_quant_matmul(x, self.weight, self.out_scale, pertoken_scale=x_scale, bias=self.bias,
                                           output_dtype=self.out_dtype)
            x = x.view(out_shape)

            return x


class PertensorA8W8Linear(nn.Module):
    def __init__(self, origin_linear: NpuLinear, offset=False, out_dtype=model_dtype):
        super().__init__()
        self.out_features, self.in_features = origin_linear.weight.size()
        self.weight = origin_linear.weight
        self.bias = origin_linear.bias
        self.weight.requires_grad = False
        self.weight.data = apply_quant(self.weight.data)
        self.out_dtype = out_dtype
        self.div_mode = int(os.getenv("DIV_MODE", "1"))

        epsilon = 1e-2
        fix_rand_seed()
        self.in_scale = Parameter(1 / (torch.rand(self.in_features, dtype=quant_dtype) * (1 - epsilon) + epsilon), requires_grad=False)
        out_scale = Parameter(torch.rand(self.out_features, dtype=quant_dtype) * (1 - epsilon) + epsilon, requires_grad=False)
        if self.out_dtype == torch.float16:
            self.out_scale = torch_npu.npu_trans_quant_param(out_scale.npu(), None)  # to uint64
        else:
            self.out_scale = out_scale

    def forward(self, x):
        quant_x = torch_npu.npu_quantize(x, self.in_scale, None, torch.qint8, -1, self.div_mode)

        x = torch_npu.npu_quant_matmul(quant_x, self.weight, self.out_scale, bias=self.bias,
                                       output_dtype=self.out_dtype)  # when out is float16, scale is uint64; when out is bf16, scale is bf16
        return x


# replacing npu_DeepseekV2MLP
class DeepseekV2DynamicA8W8GMM(nn.Module):
    def __init__(self, model, out_dtype=model_dtype):
        super().__init__()
        self.enable_prefetch = int(os.getenv("ENABLE_PREFETCH", "0"))
        self.enable_combine_dequant = int(os.getenv("ENABLE_COMBINE_DEQUANT", "0"))
        self.enable_low_latency = int(os.getenv("ENABLE_LOW_LATENCY", "0"))
        self.route_share_on_same_card = int(os.getenv("ROUTE_SHARE_ON_SAME_CARD", "0"))
        self.enable_weight_nz = model.enable_weight_nz
        self.out_dtype = out_dtype
        self.config = model.config
        self.layer_flag = model.layer_flag
        self.world_size = model.world_size
        self.n_routed_experts_per_rank = model.n_routed_experts_per_rank
        self.n_routed_experts = model.n_routed_experts
        self.top_k = self.config.num_experts_per_tok

        self.act_fn = model.act_fn
        self.group_w1_w3 = model.group_w1_w3
        self.group_w1_w3.requires_grad = False
        self.group_w1_w3.data = apply_quant(self.group_w1_w3.data)
        _, self.out_features_1, self.in_features_1 = self.group_w1_w3.size()

        self.group_w2 = model.group_w2
        self.group_w2.requires_grad = False
        self.group_w2.data = apply_quant(self.group_w2.data)
        _, self.out_features_2, self.in_features_2 = self.group_w2.size()

        epsilon = 1e-2
        fix_rand_seed()
        if self.in_features_1 == self.config.hidden_size:
            if self.enable_low_latency:
                all_experts_scale_1 = torch.rand((self.n_routed_experts_per_rank, self.in_features_1), dtype=torch.float32) * (1 - epsilon) + epsilon
            else:
                all_experts_scale_1 = torch.rand((self.n_routed_experts, self.in_features_1), dtype=torch.float32) * (1 - epsilon) + epsilon
            self.all_experts_scale_1 = Parameter(all_experts_scale_1, requires_grad=False)
        else:
            self.all_experts_scale_1 = None
        fix_rand_seed()
        scale_1 = torch.rand((self.in_features_1), dtype=model_dtype) * (1 - epsilon) + epsilon
        self.in_scale_1 = Parameter(scale_1, requires_grad=False)
        self.out_scale_1 = Parameter(
            torch.rand(
                size=(self.n_routed_experts_per_rank, self.out_features_1), 
                dtype=quant_dtype) * (1 - epsilon) + epsilon, requires_grad=False)
        fix_rand_seed()
        self.in_scale_2 = Parameter(
            torch.rand(
                self.in_features_2, 
                dtype=quant_dtype) * (1 - epsilon) + epsilon, requires_grad=False)
        if self.enable_combine_dequant:
            # use f32 as scaleDtype
            self.out_scale_2 = Parameter(
                torch.rand(
                    size=(self.n_routed_experts_per_rank, self.out_features_2),
                    dtype=quant_dtype) * (1 - epsilon) + epsilon, requires_grad=False)
        else:
            # use mode_dtype as scaleDtype
            self.out_scale_2 = Parameter(
                torch.rand(
                    size=(self.n_routed_experts_per_rank, self.out_features_2),
                    dtype=model_dtype) * (1 - epsilon) + epsilon, requires_grad=False)

    def forward(self, x, expert_tokens, is_quant=False, dynamic_scale=None, avg_tokens_per_expert=None):
        # no need to transpose weight here if weight_nz enabled
        hidden_size = x.size(-1)
        if is_quant:
            h = x
            pertoken_scale = dynamic_scale
        else:
            # 1. in_scale = None
            # 2. in_scale = all(1) === None
            # 3. in_scale = other
            h, pertoken_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=self.in_scale_1)

        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            h = h.view(-1, hidden_size)
        # gmm1: gate_up
        mm1_mm3 = torch_npu.npu_grouped_matmul([h], [self.group_w1_w3],
                                                group_list=expert_tokens, split_item=3,
                                                output_dtype=torch.int32, group_type=0,
                                                group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        if self.enable_prefetch:
            torch_npu.npu_prefetch(self.group_w2, mm1_mm3, PREFETCH_SIZE, 0)
        # dequant_swiglu_quant
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            mm1_mm3, self.out_scale_1,
            pertoken_scale.squeeze(0), None, self.in_scale_2, None,
            expert_tokens, activate_left=False)

        if pertoken_scale.dim() > 1:
            inter_size = intermediate_h.size(-1)
            pertoken_scale = pertoken_scale.reshape(-1)
            intermediate_h = intermediate_h.view(-1, inter_size)
        # gmm2: down
        if self.enable_combine_dequant and (not self.route_share_on_same_card or (self.route_share_on_same_card and self.layer_flag == "moe_route")):
            out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [self.group_w2], bias=None,
                                                    group_list=expert_tokens, split_item=3,
                                                    output_dtype=torch.int32, group_type=0,
                                                    group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        else:
            out_hidden = torch_npu.npu_grouped_matmul([intermediate_h], [self.group_w2], bias=None,
                                                    scale=[self.out_scale_2], per_token_scale=[pertoken_scale],
                                                    group_list=expert_tokens, split_item=3,
                                                    output_dtype=self.out_dtype, group_type=0,
                                                    group_list_type=1, tuning_config=avg_tokens_per_expert)[0]
        return out_hidden, pertoken_scale, self.out_scale_2

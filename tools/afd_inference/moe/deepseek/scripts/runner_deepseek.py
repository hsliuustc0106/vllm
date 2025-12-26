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
import math
import copy
import logging
from functools import wraps
from operator import attrgetter
import torch
import torch_npu
from engine.model_runner import ModelRunner
from models.global_setting import _TODO_REQUIRE_API
from quant.quant_tool import replace_linear_deepseek
from models.common import model_dtype, sync_and_get_time, process_run_time, exe_mode, get_actual_seq_len_list, to_transpose_nz

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

def override(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

class DeepSeekRunner(ModelRunner):
    def __init__(self, model_path, execute_mode, **kwargs):
        super().__init__(model_path, execute_mode, **kwargs)
        self.attn_dp_size = int(os.getenv("ATTN_DP_SIZE", "1"))
        self.enable_micro_batch = int(os.getenv("ENABLE_MICRO_BATCH", "0"))
        self.use_fa_tensor = int(os.getenv("USE_FA_TENSOR", "0"))
        self.use_real_actual_seq_len = int(os.getenv("USE_REAL_ACTUAL_SEQ_LEN", "0"))
        self.token_file_path = os.getenv("TOKEN_FILE_PATH", "tokens.xlsx") if self.use_real_actual_seq_len else None

        self.next_n = kwargs.get("next_n", 0)
        self.spec_len = self.next_n + 1

    def init_model(self, is_spec=False):
        if self.only_prefill:
            from models.modeling_deepseek_prefill import DeepseekV2ForCausalLM
        else:
            from models.modeling_deepseek import DeepseekV2ForCausalLM
            from ffn import FFNForCausalLM

        if self.with_ckpt:
            self.use_pretrained_model = True
            config = None
        else:
            self.use_pretrained_model = False
            from models.configuration_deepseek import DeepseekV2Config as config
            print(f"init_ffn_model")

            config.num_hidden_layers = int(os.getenv("LAYER_NUM", "61"))
            config.first_k_dense_replace = 3
            self.is_spec = is_spec
            if self.next_n > 0:
                if is_spec:
                    config.num_hidden_layers = 1
                    config.first_k_dense_replace = 0
                    self.layer_num = config.num_hidden_layers

        DSKModel = DeepseekV2ForCausalLM if self.is_attn_die else FFNForCausalLM
        super().init_model(DSKModel, config)

    @override
    def quant_model(self):
        if self.dynamic_quant_mode:
            self.model = replace_linear_deepseek(self.model)

    @override
    def cast_format(self):
        def for_each_to_tranpose_nz(weights, enable_nz: bool, parent, layer_idx=""):
            prefix = "NZ" if enable_nz else "transpose"
            for name in weights:
                try:
                    getter = attrgetter(name)
                    tensor = getter(parent)
                except:
                    logging.info(f"[WARN] weightName {name} not exist in parent layer.{layer_idx}")
                    continue
                logging.info("Before %s; size: %s, dtype: %s", f"{prefix} layer.{layer_idx}.{name}", tensor.size(),
                             tensor.dtype)
                if enable_nz:
                    setattr(tensor, "data", to_transpose_nz(tensor, True))
                else:
                    setattr(tensor, "data", tensor.data.transpose(-2, -1))
                logging.info("After %s; size: %s, dtype: %s", f"{prefix} layer.{layer_idx}.{name}",
                             getter(parent).size(), getter(parent).dtype)

        enable_weight_nz = int(os.getenv("ENABLE_WEIGHT_NZ", "0")) > 0 and exe_mode == "dynamo"
        enable_weight_nz_decode = enable_weight_nz and self.only_decode

        weights = ["lm_head.weight"] + (["model.mtp_proj.weight"] if self.model.model.mtp_proj is not None else [])
        for_each_to_tranpose_nz(weights, enable_weight_nz_decode, self.model)

        for layer_idx, layer in enumerate(self.model.model.layers):
            if self.only_prefill and layer_idx >= self.model.config.first_k_dense_replace:
                weights = [
                    "self_attn.q_a_proj.weight",
                    "self_attn.kv_a_proj_with_mqa.weight",
                    "self_attn.kv_b_proj.weight",
                    "self_attn.k_b_proj.weight",
                    "self_attn.v_b_proj.weight",
                    "mlp.merge_up_gate_proj.weight", "mlp.down_proj.weight",
                    "mlp.shared_experts.merge_up_gate_proj.weight", "mlp.shared_experts.down_proj.weight"
                ]
                for_each_to_tranpose_nz(weights, False, layer, layer_idx)

                gmm_weights = [
                    "mlp.shared_experts.group_w1_w3", "mlp.shared_experts.group_w2",
                    "mlp.experts.group_w1_w3", "mlp.experts.group_w2"
                ]
                for_each_to_tranpose_nz(gmm_weights, enable_weight_nz, layer, layer_idx)

                weights = [
                    "self_attn.q_b_proj.weight",
                    "self_attn.q_b_nope_proj.weight", "self_attn.q_b_rope_proj.weight",
                    "self_attn.o_proj.weight"
                ]
                for_each_to_tranpose_nz(weights, enable_weight_nz, layer, layer_idx)
            else:
                weights = [
                    "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
                    "self_attn.kv_a_proj_with_mqa.weight",
                    "self_attn.kv_b_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.merge_up_gate_proj.weight", "mlp.down_proj.weight"
                ]
                for_each_to_tranpose_nz(weights, enable_weight_nz_decode, layer, layer_idx)


    @override
    def mark_inputs(self, model_inputs):
        self.mark_detail(model_inputs, "input_ids")
        self.mark_detail(model_inputs, "kv_len")
        self.mark_detail(model_inputs, "attention_mask")
        self.mark_detail(model_inputs, "position_ids")
        self.mark_detail(model_inputs, "generate_ids")
        self.mark_detail(model_inputs, "input_ids_tsfm")
        self.mark_detail(model_inputs, "cur_position_id")
        self.mark_detail(model_inputs, "input_lens")
        self.mark_detail(model_inputs, "past_key_values", is_cache=True)

        if self.use_fa_tensor:
            self.mark_detail(model_inputs, "actual_seq_lengths_kv")

    @override
    def model_input_prepare(self, input_dict):
        input_ids = input_dict.get("input_ids")
        attention_mask = input_dict.get("attention_mask")
        past_key_values = input_dict.get("past_key_values")
        is_prefill = input_dict.get("is_prefill")
        kv_len = input_dict.get("kv_len")
        share_mask_tril = input_dict.get("share_mask_tril")
        cur_position_id = input_dict.get("cur_position_id", None)
        input_ids_tsfm = input_dict.get("input_ids_tsfm", None)
        attention_mask_bk = input_dict.get("attention_mask_bk", None)
        generate_ids = input_dict.get("generate_ids", None)
        actual_seq_lengths_kv = input_dict.get("actual_seq_lengths_kv", None)

        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            is_prefill=is_prefill,
            kv_len=kv_len,
            input_lens=input_dict.get("input_lens"),
            share_mask_tril=share_mask_tril,
            world_size=self.world_size,
            cur_position_id=cur_position_id,
            input_ids_tsfm=input_ids_tsfm,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            generate_ids=generate_ids)
        if attention_mask_bk is not None:
            model_inputs["attention_mask_bk"] = attention_mask_bk

        return model_inputs

    def prompt_modifier(self, raw_prompts):
        prompts = copy.deepcopy(raw_prompts)
        if self.only_prefill:
            prompts = prompts * self.batch_size
        elif self.enable_low_latency:
            prompts = prompts * (self.batch_size // self.attn_dp_size)
        else:
            prompts = prompts * (self.batch_size // self.attn_dies)
        return prompts

    def model_init_dict_prepare(self, prompts, main_input_dict=None, **kwargs):
        inputs = self.get_tokenizer_output(prompts, **kwargs)

        if main_input_dict is None:
            input_ids = inputs.input_ids.to(torch.int32)
            attention_mask = inputs.attention_mask
            if self.next_n > 0:
                input_ids_tsfm = torch.randn(
                    (input_ids.size(0), input_ids.size(1), self.hidden_size), 
                    dtype=model_dtype, device=input_ids.device)
            else:
                input_ids_tsfm = None # for common inputs
            if int(os.getenv("ENABLE_PROFILE", "0")):
                attention_mask = attention_mask * 0 + 1
        else: # MTP prefill shifts left main prefill output
            input_ids = main_input_dict["input_ids_spec"]
            attention_mask = main_input_dict["attention_mask_bk"]
            input_ids_tsfm = main_input_dict["input_ids_tsfm"][0, :].unsqueeze(0)

        # get init input_dict, if use npu pfa, attention mask is set to default 2048*2048
        share_mask_tril = self.get_init_attn_mask(2048, self.device)
        if self.next_n > 0:
            attention_mask = share_mask_tril
        bsz = self.batch_size * self.attn_tp_size // self.attn_dies
        kv_len = torch.ones(
            (bsz, self.spec_len), 
            dtype=torch.int64, device=input_ids.device)

        input_lens = torch.tensor(input_ids.size(1), dtype=torch.int32).npu()
        generate_ids = torch.randn((input_ids.size(0), 1024), dtype=torch.int32, device=input_ids.device).clip(0, 100)

        if not self.use_real_actual_seq_len:
            if self.use_fa_tensor:
                actual_seq_lengths_kv = torch.ones(
                    (bsz, self.spec_len),
                    dtype=torch.int64, device=input_ids.device) * _TODO_REQUIRE_API["actual_seq_len"]
                print(f"jcz not use_real_actual_seq_len actual_seq_lengths_kv:{actual_seq_lengths_kv.shape}")
            else:
                actual_seq_lengths_kv = [_TODO_REQUIRE_API["actual_seq_len"],] * bsz
                print(f"jcz not use_real_actual_seq_len actual_seq_lengths_kv:{len(actual_seq_lengths_kv)} {actual_seq_lengths_kv[0]}")
        else:
            assert os.path.exists(self.token_file_path)
            # TODO:适配use_fa_tensor场景
            row_offset = (self.global_rank - self.ffn_dies) * bsz + 1
            actual_seq_lengths_kv = get_actual_seq_len_list(self.token_file_path, bsz, row_offset)
            logging.info(f"actual_seq_lengths_kv: {actual_seq_lengths_kv}, max_len: {max(actual_seq_lengths_kv)}, min_len: {min(actual_seq_lengths_kv)}")

        logging.info(f"Prompt lens: {input_lens}, input_shape: {input_ids.shape}, is_prefill: {self.only_prefill}")
        input_dict = {
            "input_ids": input_ids,
            "input_lens": input_lens,
            "attention_mask": attention_mask,
            "past_key_values": self.model.kv_cache,
            "is_prefill": self.only_prefill,
            "kv_len": kv_len,
            "share_mask_tril": share_mask_tril,
            "generate_ids": generate_ids,
            "input_ids_tsfm": input_ids_tsfm,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
        }
        return input_dict

    @override
    def model_generate(self, prompts, warm_up=False, **kwargs):
        assert self.input_max_len > 0
        next_n = kwargs.get("next_n", 0)
        prompts = self.prompt_modifier(prompts)
        input_dict = self.model_init_dict_prepare(prompts, **kwargs)

        generate_tokens = 0
        cnt = 0
        profile_switch, profile_save_path = self._get_running_profiling_config(0 if self.only_prefill else 2, warm_up)
        profiler = self._define_profiling(profile_switch, profile_save_path)

        run_time_list = []
        with profiler as prof:
            while True:
                jump_flag = self._get_running_config(cnt, warm_up, generate_tokens)
                if jump_flag:
                    break
                # obtain model start time
                start_time = sync_and_get_time()
                model_inputs = self.model_input_prepare(input_dict)
                output_dict, logits = self.model_inference(model_inputs, warm_up=warm_up)

                # sync device to obtain end time and mark profiling step
                run_time = sync_and_get_time(start_time, self.model_name)
                run_time_list.append(run_time * 1000)
                prof.step()
                generate_tokens += 1
                cnt += 1

                next_tokens = output_dict["input_ids"]
                print(f"next tokens={next_tokens}", flush=True)

        generate_ids = input_dict["generate_ids"][0:1, input_dict["input_lens"]:].clip(0, self.model.config.vocab_size-1)
        res = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        if isinstance(res, list):
            for answer in res:
                logging.info("Inference decode result: \n%s", answer)
        else:
            logging.info("Inference decode result: \n%s", res)
        # record time
        avg_run_time, flag = process_run_time(run_time_list)
        logging.info("Inference Decode Average Run Time: {:.2f} ms {}".format(avg_run_time, flag))
        return res

    def ffn_server(self):
        micro_batch_number = int(os.getenv("MICRO_BATCH_NUMBER", "3")) if exe_mode == "dynamo" else 1
        dumy_x = torch.ones([self.batch_size * self.spec_len // self.attn_dies // micro_batch_number, self.hidden_size], dtype=model_dtype).npu()
        logging.info(f"ffn_server dumy_x:{dumy_x.shape}")
        cnt = 0
        expert_layer_num = (self.layer_num - 3 + 1) if self.next_n != 0 else (self.layer_num - 3) # 稀疏层的层数，MTP场景下需要加上MTP的moe层
        max_step = math.ceil(self.max_new_tokens/(1 + 0.7 * self.next_n)) if self.next_n != 0 else self.max_new_tokens  # MTP场景接受率固定0.7
        ffn_max_step = (max_step + 2) #* expert_layer_num * micro_batch_number  # 2: ATTN侧的warm_up会推2个Token,为了保证能够同时退出
        logging.info(f"Begin to run ffn server, mtp{self.next_n}, max_step:{ffn_max_step}")
        profile_switch, profile_save_path = self._get_running_profiling_config(0 if self.only_prefill else 2, False)
        profiler = self._define_profiling(profile_switch, profile_save_path)
        with profiler as prof:
            while cnt < ffn_max_step if not self.use_real_actual_seq_len else True:
                with torch.no_grad():
                    y = self.model(dumy_x)
                prof.step()
                cnt += 1

    def generate(self, prompts, enable_profile=False, **kwargs):
        self.enable_profile = enable_profile
        if self.is_attn_die:
            self.model_generate(prompts, True, **kwargs)
            self.model_generate(prompts, False, **kwargs)
        else:
            self.ffn_server()

    def set_enable_profile(self, flag):
        self.enable_profile = flag
        logging.info(">>>>> Runner set_enable_profile as: %d", flag)

    def _get_running_config(self, cnt, warm_up, generate_tokens):
        default_decode_dump = 2

        # warm_up only perform for 5 times(decode)
        jump_flag_warm = warm_up and cnt >= default_decode_dump
        # dont generate after max_token
        jump_flag_oversize = generate_tokens >= self.max_new_tokens or cnt >= self.max_new_tokens
        jump_flag = jump_flag_warm or jump_flag_oversize

        return jump_flag

    def _get_running_profiling_config(self, cnt, warm_up):
        # profile settings
        profile_switch = self.enable_profile and (not warm_up)

        path_prefill = f"{self.profiling_path}/prefill"
        path_incre = f"{self.profiling_path}/incre"
        profile_save_path_dict = {0: path_prefill, 2: path_incre}
        profile_save_path = profile_save_path_dict.get(cnt, path_incre)
        return profile_switch, profile_save_path

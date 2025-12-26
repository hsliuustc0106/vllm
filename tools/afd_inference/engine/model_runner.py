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
import torch
import torch_npu
from transformers import AutoTokenizer
from torch.distributed.distributed_c10d import _world
import numpy as np

torch._logging.set_logs(recompiles=True)
root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.npu.config.allow_internal_format = True


class InferenceContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def step(self):
        return

def show_model_states(origin_model, model_name="src_model"):
    src_param_size = 0
    for name, params in origin_model.named_parameters():
        size_per_param = np.prod(params.size())
        src_param_size += size_per_param
        logging.info("Paramgrep: %s, %s, %s, %s",
                     name, params.size(), params.dtype, params.device)
    logging.info("Total param size of %s tensor parallel: %s", model_name, src_param_size)

class ModelRunner:
    def __init__(self, model_path, execute_mode, **kwargs):
        self.model_name = kwargs.get("model_name", "default_model_name")
        self.dtype = kwargs.get("dtype", torch.bfloat16)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 131072)
        self.input_max_len = kwargs.get("input_max_len", 1024)
        self.max_new_tokens = kwargs.get("max_new_tokens", 32)
        self.batch_size = kwargs.get("batch_size", 72)
        self.tokenizer = None
        self.model = None
        self.device = None
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.rank_id = int(os.getenv("RANK_ID", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        assert self.rank_id == self.global_rank
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.attn_tp_size = int(os.getenv("ATTN_TP_SIZE", "1"))
        self.attn_dies = int(os.getenv("ATTN_DIES", "1"))
        self.ffn_dies = int(os.getenv("FFN_DIES", "1"))
        self.layer_num = int(os.getenv("LAYER_NUM", "61"))
        self.dynamic_quant_mode = int(os.getenv("QUANT_MODE", "3"))
        self.next_n = kwargs.get("next_n", 0)
        self.is_spec = False
        self.enable_cache_compile = int(os.getenv("ENABLE_CACHE_COMPILE", "0"))
        self.save_model_switch = int(os.getenv("SAVE_MODEL", "0"))
        if self.world_size == 1:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(model_path, f"rank_{self.local_rank}")

        self.use_pretrained_model = True
        self.execute_mode = execute_mode
        self.tokenizer_mode = kwargs.get("tokenizer_mode", "default")
        self.profiling_path = kwargs.get("profiling_path", "")
        self.dump_precision_path = kwargs.get("dump_precision_path", "")
        self.enable_profile = False

        self.with_ckpt = int(os.getenv("WITH_CKPT", "0"))
        self.only_prefill = os.getenv("PREFILL_OR_DECODE", "decode") == "prefill"
        self.only_decode = not self.only_prefill
        self.next_n = kwargs.get("next_n", 0)
        self.spec_len = self.next_n + 1
        self.kvcache = None
        self.enable_low_latency = int(os.getenv("ENABLE_LOW_LATENCY", "0"))
        self.hidden_size = None
        layer_out = os.getenv("LAYER_OUT", "FA")
        if layer_out.upper() == "AF":
            self.is_attn_die = True if self.global_rank < self.attn_dies else False
        else:
            self.is_attn_die = True if self.global_rank >= self.ffn_dies else False
        self.ffn_mode = os.getenv("FFN_MODE")

        self.init_device()

    def init_device(self):
        logging.info("Set execution using npu index: %s, global: %s", self.local_rank, self.global_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        if torch.npu.is_available() and self.world_size > 1:
            if _world._default_pg is None:
                print(f"rank:{self.global_rank} begin to init_process_group", flush=True)
                torch.distributed.init_process_group(
                    backend="hccl",
                    world_size=self.world_size, rank=self.global_rank)
                print(f"rank:{self.global_rank} world_size:{self.world_size} success to init_process_group", flush=True)

    def init_model(self, model, config=None):
        print("begin load_model")
        if self.use_pretrained_model:
            self.load_model(model)
        else:
            self.init_model_from_config(model, config=config)
        print("end load_model")
        self.quant_model() if self.is_attn_die else None # FFN已经初始化时已经是量化和NZ后的
        print("end quant_model")
        self.to_device()
        print("end to_device")
        self.cast_format() if self.is_attn_die else None
        print("end cast_format")
        if self.save_model_switch:
            self.save_model() if self.global_rank in [0, 1, 15] else None
        print("begin compile_model")
        self.compile_model()
        print("end compile_model")
        self.init_tokenizer()
        print("end init_tokenizer")

    def init_model_from_config(self, model, config):
        assert config is not None
        logging.info("Try to init_model_from_config!!!")
        current_file_path = os.path.dirname(__file__)
        config_file = os.path.join(current_file_path,  '../moe/deepseek/config/config.json')
        model_config = config.from_pretrained(config_file, torch_dtype=self.dtype,\
                        max_position_embeddings=self.max_position_embeddings)
        model_config.num_hidden_layers = config.num_hidden_layers
        model_config.first_k_dense_replace = config.first_k_dense_replace
        model_config.is_spec = self.is_spec
        self.model = model(model_config)
        self.ffn2attn = self.model.ffn_to_attn if not self.is_attn_die else None
        self.hidden_size = model_config.hidden_size

    def load_model(self, model):
        logging.info("Try to load pretrained model in path: [%s]", self.model_path)
        self.model = model.from_pretrained(self.model_path,
                                                low_cpu_mem_usage=True,
                                                ignore_mismatched_sizes=True,
                                                torch_dtype=self.dtype,
                                                max_position_embeddings=self.max_position_embeddings,
                                            )
        self.model.config.num_hidden_layers = self.layer_num
        print(f"self.model.config.num_hidden_layers>>>  {self.model.config.num_hidden_layers}", flush=True)
        self.hidden_size = self.model.config.hidden_size

    def save_model(self):
        save_model_path_prefix = os.getenv("SAVE_MODEL_PATH", "/home/l00653936/weight/bbit")
        save_model_path = f"{save_model_path_prefix}/ckpt_ds_layer{self.layer_num}_atp{self.attn_tp_size}_{self.world_size}p/rank_{self.local_rank}/"

        self.model.save_pretrained(save_model_path)
        show_model_states(self.model)
        logging.info(f"Model saved to {save_model_path}")
        exit()

    def quant_model(self):
        pass

    def to_device(self):
        # show_model_states(self.model)
        self.model.to(self.device)

    def cast_format(self):
        pass

    def compile_model(self):
        logging.info("The final model structure is: \n %s", self.model)
        if self.is_attn_die:
            if self.execute_mode == "dynamo":
                logging.info("Try to compile attn model")
                self.graph_compile()
        else:
            if self.ffn_mode == "dynamo":
                logging.info("Try to compile ffn model")
                self.graph_compile()
    
    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="right", truncation_side='right')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    
    def graph_compile(self):
        # FFN用编译缓存后性能更差，因此过滤掉它
        if not self.is_attn_die or self.enable_cache_compile == 0:
            logging.info("graph_compile 1 begin")
            import torchair as tng
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
            from torchair.configs.compiler_config import CompilerConfig
            compiler_config = CompilerConfig()
            compiler_config.experimental_config.frozen_parameter = True
            compiler_config.experimental_config.tiling_schedule_optimize = True
            # compiler_config.debug.graph_dump.type = 'pbtxt'
            # compiler_config.experimental_config.enable_view_optimize=False
            compiler_config.experimental_config.topology_sorting_strategy = "StableRDFS"
            npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
            self.model = torch.compile(self.model, dynamic=self.is_attn_die, fullgraph=True, backend=npu_backend)
            logging.info("graph_compile 1 end")
        else:
            logging.info("graph_compile 2")
            pass

    def mark_detail(self, model_inputs, item_key, is_cache=False):
        if self.execute_mode == "dynamo":
            item = model_inputs.get(item_key, None)
            if item is None:
                return
            if is_cache:
                for item_sub in item:
                    for sub_item in item_sub:
                        torch._dynamo.mark_static(sub_item)
            else:
                torch._dynamo.mark_static(item)

    def mark_inputs(self, model_inputs):
        if self.execute_mode == "dynamo":
            pass

    def model_input_prepare(self, input_dict):
        return None

    def repeat_batch(self, tensor, N):
        if N == 1:
            return tensor
        return tensor.repeat(N, *[1]*(tensor.dim() - 1))

    def _define_profiling(self, profile_switch=False, profile_save_path="prof"):
        if profile_switch:
            os.makedirs(profile_save_path, exist_ok=True)
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
            if self.only_decode:
                active = 7 if self.is_attn_die else 5
                repeat = 1
                skip_first = 1
            else:
                active=1
                repeat=1
                skip_first=0
            profiler = torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.NPU,
                        torch_npu.profiler.ProfilerActivity.CPU],
                    with_stack=False,
                    record_shapes=False,
                    profile_memory=False,
                    experimental_config=experimental_config,
                    schedule=torch_npu.profiler.schedule(wait=15, warmup=0, active=active, repeat=repeat, skip_first=skip_first),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_save_path))
        else:
            profiler = InferenceContextManager()
        return profiler

    def model_inference(self, model_inputs, warm_up=False):
        if warm_up:
            self.mark_inputs(model_inputs)
        with torch.no_grad():
            logits = self.model(**model_inputs)
        return logits

    def model_generate(self, prompts, warm_up=False, **kwargs):
        pass

    def get_tokenizer_output(self, prompts, **kwargs):
        assert self.input_max_len > 0
        calling_func = {
            "default": self.tokenizer,
            "chat": self.tokenizer.apply_chat_template,
        }
        if self.only_prefill:
            max_length = self.input_max_len
        else:
            max_length = self.spec_len
        kwargs = {
            "return_tensors": "pt",
            "truncation": True,
            "padding": "max_length",
            "max_length": max_length
        }

        if self.tokenizer_mode == "chat":
            chat_kwargs = {
                "add_generation_prompt": True, "return_dict": True,
            } 
            kwargs.update(chat_kwargs)

        tokenizer = calling_func[self.tokenizer_mode]
        inputs = tokenizer(prompts, **kwargs).to(self.device)
        return inputs

    @classmethod
    def get_init_attn_mask(cls, mask_length, device, valid_len=None):
        share_mask_tril = ~torch.tril(
            torch.ones((mask_length, mask_length), 
                    dtype=torch.bool, device=device))
        if valid_len is not None:
            share_mask_tril[-valid_len:, :] = torch.zeros(valid_len, mask_length)
        return share_mask_tril

    @classmethod
    def get_decode_mask(cls, mask_length, device, position):
        decode_mask = torch.zeros((1, mask_length), dtype=torch.bool, device=device)
        decode_mask[0, :position] = True
        return decode_mask

    @classmethod
    def to_nz(cls, tensor, name=""):
        # logging.info("NZ parameter %s; size: %s, dtype: %s", name, tensor.size(), tensor.dtype)
        return torch_npu.npu_format_cast(tensor.data, 29)
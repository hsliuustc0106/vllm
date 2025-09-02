# **************************************************
# vllm_config: model='/data2/models/deepseek-v2-lite', speculative_config=None, tokenizer='/data2/models/deepseek-v2-lite', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=163840, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/data2/models/deepseek-v2-lite, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=True, pooler_config=None, compilation_config={"level":3,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output","vllm.mamba_mixer2"],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":1,"cudagraph_capture_sizes":[512,504,496,488,480,472,464,456,448,440,432,424,416,408,400,392,384,376,368,360,352,344,336,328,320,312,304,296,288,280,272,264,256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"pass_config":{},"max_capture_size":512,"local_cache_dir":null}
# local_rank: 0
# rank: 0
# distributed_init_method: tcp://10.90.67.86:37193
# **************************************************

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

from vllm.utils import get_distributed_init_method, get_open_port, get_ip
from typing import Any, Optional, Union
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.worker.gpu_worker import Worker
from vllm.forward_context import set_forward_context
from vllm.distributed import init_afd_process_group

from datetime import timedelta

# def creat_process_group(rank, world_size, attn_size, ffn_size):
#     import torch

#     torch.npu.set_device(rank)
#     new_default_group = init_process_group(
#         init_method="tcp://127.0.0.1:29500",
#         backend="nccl",
#         rank=rank,
#         world_size=world_size,
#         group_name="ffn",
#     )
#     return new_default_group


def create_worker(
    model_name: str,
    block_size: int,
    seed: int,
    is_driver_worker: bool = True,
    enforce_eager: bool = True,
    dtype: Optional[str] = "auto",
):
    engine_args = EngineArgs(
        model=model_name,
        seed=seed,
        block_size=block_size,
        enforce_eager=enforce_eager,
        dtype=dtype,
        additional_config={"role": "ffn"},
    )
    engine_config = engine_args.create_engine_config()

    # new_default_group = init_afd_process_group(
    #     init_method="tcp://127.0.0.1:29500",
    #     backend="nccl",
    #     rank=1,
    #     world_size=2,
    #     group_name="ffn",
    #     timeout=timedelta(minutes=2),
    # )

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port()
    )

    worker = Worker(
        vllm_config=engine_config,
        local_rank=0,
        rank=0,
        distributed_init_method=distributed_init_method,
        is_driver_worker=is_driver_worker,
    )

    worker.init_device()
    worker.load_model()
    print("ffn worker instantiated")
    with set_forward_context(
        None,
        engine_config,
        num_tokens=1,
        num_tokens_across_dp=1,
        skip_cuda_graphs=enforce_eager,
    ):
        while(True):
            worker.model_runner.model.forward_ffn()


model_name = "/data2/models/deepseek-v2-lite"
block_size = 128
seed = 0
is_driver_worker = True
enforce_eager = True
additional_configs = {}
create_worker(model_name, block_size, seed, is_driver_worker, enforce_eager)

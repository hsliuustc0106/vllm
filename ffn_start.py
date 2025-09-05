# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

from vllm.utils import get_distributed_init_method, get_open_port, get_ip
from typing import Optional
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.worker.gpu_worker import AFDWorker
from vllm.forward_context import set_forward_context


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

    distributed_init_method = get_distributed_init_method(
        get_ip(), get_open_port()
    )

    worker = AFDWorker(
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
        while True:
            worker.model_runner.model.forward_ffn()


# destroy_process_group()

model_name = "/data2/models/deepseek-v2-lite"
block_size = 128
seed = 0
is_driver_worker = True
enforce_eager = True
additional_configs = {}
create_worker(model_name, block_size, seed, is_driver_worker, enforce_eager)

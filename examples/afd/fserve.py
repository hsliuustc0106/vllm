# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import re

import torch.multiprocessing as mp

from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.utils import cli_env_setup
from vllm.utils import (
    FlexibleArgumentParser,
    get_distributed_init_method,
    get_ip,
    get_open_port,
)
from vllm.v1.worker.gpu_worker import AFDWorker


def create_worker(
    vllm_config,
    rank,
    distributed_init_method,
    is_driver_worker: bool = True,
):
    worker = AFDWorker(
        vllm_config=vllm_config,
        local_rank=rank,
        rank=rank,
        distributed_init_method=distributed_init_method,
        is_driver_worker=is_driver_worker,
    )

    worker.init_device()
    worker.load_model()
    print("ffn worker instantiated")
    worker.model_runner.execute_model()


if __name__ == "__main__":
    cli_env_setup()
    mp.set_start_method("spawn")
    parser = FlexibleArgumentParser(description="vLLM AFD FFN server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    engine_args = EngineArgs.from_cli_args(args)
    vllm_config = engine_args.create_engine_config()
    afd_size = vllm_config.additional_config.get("afd_size")
    if afd_size is None or afd_size == "":
        raise ValueError("Afd size must be specified in additional_config")

    attn_size, ffn_size = map(int, re.match(r"(\d+)\D+(\d+)", afd_size).groups())
    distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())

    processes = []
    for rank in range(ffn_size):
        p = mp.Process(
            target=create_worker, args=(vllm_config, rank, distributed_init_method)
        )
        processes.append(p)
        p.start()

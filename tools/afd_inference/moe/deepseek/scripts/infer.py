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
import sys
import json
import copy
import time
import logging
import argparse
import torch
# import torch_npu
# import torchair
# from torchair.core.utils import logger
# import logging
# logger.setLevel(logging.DEBUG)

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(CUR_DIR, "../../.."))
sys.path.append(ROOT_DIR)
RUN_DIR = os.path.realpath(os.path.join(CUR_DIR, ".."))
sys.path.append(RUN_DIR)

from engine.utils import generate_prompt
from runner_deepseek import DeepSeekRunner
from models.common import model_dtype, sync_and_get_time, process_run_time

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


import subprocess
def bind_process_to_core(pid):
    try:
        # 构建 taskset 命令
        command = f"taskset -pc {pid}"
        # 执行命令
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        tgt_core = int(result.stdout.strip().split(':')[-1].split('-')[0]) + 4
        print("bind ======================== 1", tgt_core, flush=True)
        command = f"taskset -pc {tgt_core} {pid}"
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("bind ======================== 2", result, flush=True)
        command = f"renice -n -20 -p {pid}"
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("bind ======================== 3", result, flush=True)
    except subprocess.CalledProcessError as e:
        print(f"bind core error: {e.stderr.strip()}")


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--model_path', type=str, help="Path of model weights")
    parser.add_argument('--model_name', type=str, help="Model Name")
    parser.add_argument('--execute_mode', type=str, default="eager", choices=["dynamo", "eager"],
                        help="eager or dynamo")
    parser.add_argument('--tokenizer_mode', type=str, default="default", choices=["default", "chat"],
                        help="tokenizer_mode should be default or chat")
    parser.add_argument('--profiling_path', type=str, help="Path of profiling, not set means not dump")
    parser.add_argument('--dump_precision_path', type=str, help="Path of dump data")
    parser.add_argument('--local-rank', type=int, default=0, help="Local rank id for torch distributed launch")
    parser.add_argument('--input_max_len', type=int, default=1024, help="Max number of input")
    parser.add_argument('--max_new_tokens', type=int, default=32, help="Max number of new tokens")
    parser.add_argument('--batch_size', type=int, default=2, help="Batch size for testing")
    parser.add_argument('--json_path', type=str, help="Path of settings")
    parser.add_argument('--enable_mla', type=int, help="Enable MLA for deepseek_v2, default is GQA")
    parser.add_argument('--next_n', type=int, default=0, help="next n token prediction, default is 0 (not apply mtp)")
    parser_args = parser.parse_args()
    return parser_args

def run_deepseek(model_path, execute_mode, **kwargs):
    # default use 1Batch for prefill and nBatch for decode
    _PROMPTS = generate_prompt(1, args.tokenizer_mode)
    model_runner = DeepSeekRunner(model_path, execute_mode, **kwargs)

    # 表示在图模式下开启算子二进制复用，提高图模式下编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()

    if not int(os.getenv("ON_CLOUD", "0")):
        current_pid = os.getpid()
        bind_process_to_core(current_pid)
    enable_prof = int(os.getenv("ENABLE_PROFILE", "0"))
    model_runner.generate(_PROMPTS, enable_profile=enable_prof, **kwargs)


def repeat_batch(tensor, N):
        if N == 1:
            return tensor
        return tensor.repeat(N, *[1]*(tensor.dim() - 1))


def model_generate_spec(_PROMPTS, main_model, mtp_model, warm_up=False, **kwargs):
    generated_tokens_list = []
    next_n = kwargs.get("next_n", 0)
    prompts = main_model.prompt_modifier(_PROMPTS)

    main_input_dict = main_model.model_init_dict_prepare(prompts, **kwargs)
    mtp_input_dict = mtp_model.model_init_dict_prepare(prompts, **kwargs)

    generate_tokens = 0
    cnt = 0
    cur_position_id = main_input_dict['input_lens']
    cur_position_id_spec = 0
    accepted_lens = 0
    is_first_loop = True

    profile_switch, profile_save_path = main_model._get_running_profiling_config(2, warm_up)
    profiler_mtp = mtp_model._define_profiling(profile_switch, profile_save_path)

    run_time_list = []
    with profiler_mtp as prof:
        while True:
            jump_flag = main_model._get_running_config(cnt, warm_up, generate_tokens)
            if jump_flag:
                break

            # obtain model start time
            start_time_mtp = sync_and_get_time(use_syn=False)

            # update mtp input_dict
            mtp_input_dict['generate_ids'] = main_input_dict['generate_ids'] # spec_generate_ids 
            mtp_input_dict['input_ids_tsfm'] = main_input_dict['input_ids_tsfm'] # spec_input_ids_tsfm 
            # mtp_input_dict['cur_position_id'] = cur_position_id_spec

            # sequentially spec decode next n tokens
            for i in range(next_n):
                model_inputs = mtp_model.model_input_prepare(mtp_input_dict)
                mtp_input_dict, _ = mtp_model.model_inference(model_inputs, warm_up=warm_up)

            # update_main inputs from mtp_input_dict
            main_input_dict['input_ids'] = mtp_input_dict['input_ids']
            main_input_dict['input_ids_tsfm'] = mtp_input_dict['input_ids_tsfm']
            model_inputs = main_model.model_input_prepare(main_input_dict)
            main_input_dict, main_logits = main_model.model_inference(model_inputs, warm_up=warm_up)

            # sync device and mark profiler step
            run_time_mtp = sync_and_get_time(start_time_mtp, model_name=f"{main_model.model_name}")
            run_time_list.append(run_time_mtp * 1000)
            prof.step()

            # cur_position_id_spec = cur_position_id + 1 + next_n # TODO for speed_test only
            cnt += 1
            generated_tokens_list.append(main_input_dict["input_ids"])
            generate_tokens = generate_tokens + 0.7 * next_n  + 1
            # logging.info("pass MTP verify and updated, accept_rate: %d, num token accepted: %d, generate_tokens: %d", accepted_rate, accepted_lens, generate_tokens)

            if int(os.getenv("DUMP_DATA", "0")):
                local_rank = int(os.getenv("LOCAL_RANK", "999"))
                if local_rank == 15 and cnt >= 3 and cnt % 25 == 0:
                    dump_output_path = os.path.join(main_model.dump_precision_path, "rank_{}_output_{}.pt".format(os.getenv("RANK_ID"), cnt))
                    torch.save(main_logits.detach().cpu(), dump_output_path)

    if len(generated_tokens_list) > 0:
        generate_ids = torch.cat(generated_tokens_list, dim=1)[0:1, :].clip(0, main_model.model.config.vocab_size - 1)
        res = main_model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
    else:
        res = None

    if isinstance(res, list):
        for answer in res:
            logging.info("Inference decode result: \n%s", answer)
    else:
        logging.info("Inference decode result: \n%s", res)

    avg_run_time, flag = process_run_time(run_time_list)
    logging.info("Inference Decode Average Run Time: {:.2f} ms {}".format(avg_run_time / (1 + 0.7 * next_n), flag))

    return res


def run_deepseek_next_n(model_path, execute_mode, **kwargs):
    # default use 1Batch for prefill and nBatch for decode
    torch.npu.set_compile_mode(jit_compile=False)
    enable_prof = int(os.getenv("ENABLE_PROFILE", "0"))

    model_runner = DeepSeekRunner(model_path, execute_mode, **kwargs)
    logging.info("run_deepseek_next_n begin init_model")
    model_runner.init_model(is_spec=False)
    logging.info("run_deepseek_next_n end init_model")
    model_runner.set_enable_profile(enable_prof)
    if model_runner.is_attn_die:
        model_runner_mtp = DeepSeekRunner(model_path, execute_mode, **kwargs)
        model_runner_mtp.init_model(is_spec=True)
        model_runner_mtp.set_enable_profile(enable_prof)

        _PROMPTS = generate_prompt(1, args.tokenizer_mode)
        # warmup
        logging.info("run_deepseek_next_n is_attn_die begin warm_up")
        model_generate_spec(_PROMPTS, model_runner, model_runner_mtp, warm_up=True, **kwargs)
        logging.info("run_deepseek_next_n is_attn_die end warm_up")
        # generate perf data
        model_generate_spec(_PROMPTS, model_runner, model_runner_mtp, **kwargs)
    else:
        logging.info("run_deepseek_next_n begin ffn_server")
        model_runner.ffn_server()


if __name__ == "__main__":
    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)

    args = parse_args()
    input_max_len = args.input_max_len  # 输入padding的长度
    max_new_tokens = args.max_new_tokens  # 最大输出token个数
    max_position_embeddings = (input_max_len + max_new_tokens + args.next_n)  # 用于申请kv_cache时指定seq_len长度
    model_config = {
        "dtype": model_dtype,  # 和模型权重目录下config.json中torch_dtype一致
        "input_max_len": input_max_len,
        "max_new_tokens": max_new_tokens,
        "max_position_embeddings": max_position_embeddings,
        "enable_mla": args.enable_mla
    }
    run_config = {
        "tokenizer_mode": args.tokenizer_mode,
        "profiling_path": args.profiling_path,
        "dump_precision_path": args.dump_precision_path,
        "batch_size": args.batch_size,
        "model_name": args.model_name,
        "next_n": args.next_n
    }
    config = {**model_config, **run_config}
    if args.next_n > 0:
        run_deepseek_next_n(args.model_path, args.execute_mode, **config)
    else:
        run_deepseek(args.model_path, args.execute_mode, **config)

    logging.info("model run success")

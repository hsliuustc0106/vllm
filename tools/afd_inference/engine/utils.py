# Copyright 2020 Huawei Technologies Co., Ltd
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
import math
import os
import json
import logging
import time
import torch
import random
import copy
import shutil
import torch_npu
import numpy as np
from difflib import SequenceMatcher

SUCCESS = 0
FAILED = 1


class Dataset:
    def __init__(self, dataset_name=None, batch_size=1, infer_iters=0, model=None, params=None, device_list=None):
        self.dataset_name = dataset_name
        if self.dataset_name is None:
            self.dataset_name = "gsm8k"
        self.batch_size = batch_size
        self.infer_iters = infer_iters
        self.model = model
        self.params = params
        self.device_list = device_list
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if (len(self.device_list) > 1 and (local_rank == 0 or local_rank == self.device_list[0])) or (
                len(self.device_list) == 1):
            logging.info("Select dataset name: %s" % self.dataset_name)

    def generate_format_dataset(self):
        dataset_file = f"config/dataset/{self.dataset_name}.json"
        batch_prompts = []
        batch_answers = []
        with open(dataset_file, 'r') as d:
            dataset = json.load(d)
        for idx in range(int(
            len(dataset) / self.batch_size)):
            prompts = []
            answers = []
            prompts_dict = dataset[self.batch_size * idx: self.batch_size * (idx + 1)]
            for prompt in prompts_dict:
                if self.model in ["llama2"] and self.params in ["13b"] and self.dataset_name in ["sft"]:
                    prompts.append("<s>Human:" + prompt["input"] + " </s><s>Assistant: ")
                else:
                    prompts.append(prompt["input"])
                answers.append(prompt["answer"])
            batch_prompts.append(prompts)
            batch_answers.append(answers)
        last_batch = dataset[int(len(dataset) / self.batch_size) * self.batch_size:]
        if last_batch:
            prompts = []
            answers = []
            prompts_dict = last_batch + (self.batch_size - len(last_batch)) * [last_batch[-1]]
            for prompt in prompts_dict:
                prompts.append(prompt["input"])
                answers.append(prompt["answer"])
            batch_prompts.append(prompts)
            batch_answers.append(answers)
        if self.infer_iters != 0:
            loop = math.ceil(self.infer_iters / len(batch_prompts))
            batch_prompts = (batch_prompts * loop)[:self.infer_iters]
            batch_answers = (batch_answers * loop)[:self.infer_iters]
        batch_answers_json = json.dumps(batch_answers, ensure_ascii=False, indent=4)
        precision_golden_path = f'./json/precision/golden/{self.model}/{self.params}/'

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        if (len(self.device_list) > 1 and (local_rank == 0 or local_rank == self.device_list[0])) or (len(self.device_list) == 1):
            os.makedirs(precision_golden_path, exist_ok=True)
            f_answer = open(
                os.path.join(precision_golden_path, f'{self.model}_{self.params}_{self.dataset_name}.json'), 'w')
            f_answer.write(batch_answers_json)
            f_answer.close()
        return batch_prompts, batch_answers

    def generate_cos_e(self):
        dataset_file = "dataset/cos_e.json"
        batch_prompts = []
        with open(dataset_file, 'r') as d:
            dataset = json.load(d)
        for idx in range(int(len(dataset) / self.batch_size)):
            prompts = []
            prompts_dict = dataset[self.batch_size * idx: self.batch_size * (idx + 1)]
            for prompt in prompts_dict:
                prompts.append(prompt["question"]["stem"])
            batch_prompts.append(prompts)
        last_batch = dataset[int(len(dataset) / self.batch_size) * self.batch_size:]
        if last_batch:
            prompts = []
            prompts_dict = last_batch + (self.batch_size - len(last_batch)) * [last_batch[-1]]
            for prompt in prompts_dict:
                prompts.append(prompt["question"]["stem"])
            batch_prompts.append(prompts)
        if self.infer_iters != 0:
            batch_prompts = batch_prompts[:self.infer_iters]
        return batch_prompts

    def generate_gsm8k(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_sft(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_mixtral_test(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_firefly(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_mmlu_zeroshot(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_ceval_zeroshot(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_cmmlu_zeroshot(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_bbh_zeroshot(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_mmlu_fewshot(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_ceval_fewshot(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_cmmlu_fewshot(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_bbh_fewshot(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts

    def generate_sophon(self):
        batch_prompts = Dataset.generate_format_dataset(self)
        return batch_prompts


def generate_prompts(model_config):
    params = model_config["params"]
    batch_size = model_config["batch_size"]
    infer_iters = model_config["infer_iters"]
    dataset = "gsm8k" if "dataset" not in model_config.keys() else model_config["dataset"]
    device_list = model_config["device_list"]
    model = os.getenv("MODEL")
    if dataset in ["mmlu", "cmmlu", "ceval", "bbh"]:
        infer_iters = batch_size * infer_iters
        batch_size = 1

    if isinstance(dataset, list):
        logging.info(f"Select dataset type: Customizing Datasets")
        batch_prompts = []
        golden_answers = []
        for idx in range(int(len(dataset) / batch_size)):
            prompts = []
            prompts_dict = dataset[batch_size * idx: batch_size * (idx + 1)]
            for prompt in prompts_dict:
                prompts.append(prompt)
            batch_prompts.append(prompts)
        last_batch = dataset[int(len(dataset) / batch_size) * batch_size:]
        if last_batch:
            prompts = []
            prompts_dict = last_batch + (batch_size - len(last_batch)) * [last_batch[-1]]
            for prompt in prompts_dict:
                prompts.append(prompt)
            batch_prompts.append(prompts)
        if infer_iters != 0:
            loop = math.ceil(infer_iters / len(batch_prompts))
            batch_prompts = (batch_prompts * loop)[:infer_iters]
    else:
        batch_prompts, golden_answers = getattr(Dataset(dataset, batch_size, infer_iters, model, params, device_list), "generate_%s" % dataset)()
    model_config["batch_prompts"] = batch_prompts
    model_config["golden_answers"] = golden_answers


def generate_params(inputs, model_config):
    seq_len_out = model_config["seq_len_out"]
    device = model_config["device"]
    kwagrs_params = {"max_new_tokens": seq_len_out}
    for key in inputs.keys():
        kwagrs_params.update({
            key: inputs[key].to(device)
        })
    return kwagrs_params


def generate_answer(model_config):
    batch_prompts = model_config["batch_prompts"]
    model_name = os.getenv("MODEL")
    params = model_config["params"]
    precision_infer_path = f'./json/precision/infer/{model_name}/{params}/'
    batch_answers = []
    dataset = "gsm8k" if "dataset" not in model_config.keys() else model_config["dataset"]

    model_runner = model_config["runner"]
    kwargs = model_config["kwargs"]
    for prompts in batch_prompts:
        res = model_runner.model_generate(prompts, **kwargs)
        infer_answers = []

        for prompt in res:
            if dataset in ["cmmlu_zeroshot", "mmlu_zeroshot", "bbh_zeroshot", "ceval_zeroshot",
                            "cmmlu_fewshot", "mmlu_fewshot", "bbh_fewshot", "ceval_fewshot"]:
                answer = prompt.replace(' ', '').replace('.', '').replace('\n', '').replace('\r', '')
                answer = answer[0]
            elif dataset in ["arithmetic_zeroshot"]:
                answer = prompt.split('is ')[-1].split('.')[0].replace(',', '').replace(' ', '')
            else:
                answer = prompt
            infer_answers.append(answer)
        batch_answers.append(copy.deepcopy(infer_answers))
    model_config["infer_answers"] = batch_answers
    batch_answers_json = json.dumps(batch_answers, ensure_ascii=False, indent=4)

    os.makedirs(precision_infer_path, exist_ok=True)
    f_answer = open(os.path.join(precision_infer_path, f'{model_name}_{params}_{dataset}.json'), 'w')
    f_answer.write(batch_answers_json)
    f_answer.close()


def analysis_memory(model_config):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device_num = int(os.getenv("DEVICE_NUM", "1"))
    device_list = model_config["device_list"]
    max_memory = 0
    if (len(device_list) > 1 and (local_rank == 0 or local_rank == device_list[0])) or (len(device_list) == 1):
        logging.info("max_memory_allocated: %s" % torch.npu.max_memory_allocated())
        logging.info("memory_allocated: %s" % torch.npu.memory_allocated())
        logging.info("max_memory_reserved: %s" % torch.npu.max_memory_reserved())
        logging.info("memory_reserved: %s" % torch.npu.memory_reserved())
        max_memory = torch.npu.max_memory_reserved()
    try:
        if (device_num > 1 and local_rank == 0) or (device_num == 1):
            ckpt_log_path = os.getenv("CKPT_LOG_PATH")
            with open(ckpt_log_path, 'r') as d:
                ckpt_context_list = json.load(d)
            if ckpt_context_list[0].get('Max memory') == None:
                ckpt_context_list[0].update({
                    "Max memory": max_memory,
                    "Max memory flag": 1
                })
            else:
                max_memory_flag = 0
                max_memory_golden = ckpt_context_list[0]["Max memory"]
                if max_memory / max_memory_golden < 1.05:
                    max_memory_flag = 1
                ckpt_context_list[len(ckpt_context_list) -1].update({
                    "Max memory": max_memory,
                    "Max memory flag": max_memory_flag
                })
            ckpt_context_list_dump = json.dumps(ckpt_context_list, ensure_ascii=False, indent=4)
            f_answer = open(ckpt_log_path, 'w')
            f_answer.write(ckpt_context_list_dump)
            f_answer.close()
    except Exception:
        pass



def analysis_precision(model_config):
    dataset = "gsm8k" if "dataset" not in model_config.keys() else model_config["dataset"]
    golden_answer = list(np.array(model_config["golden_answers"]).flatten())
    infer_answer = list(np.array(model_config["infer_answers"]).flatten())
    model_name = os.getenv("MODEL")
    params = model_config["params"]
    device_list = model_config["device_list"]
    sim_list = []
    match_score_list = []
    sim_total = 0
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if len(golden_answer) == len(infer_answer):
        answer_len = len(golden_answer)
    else:
        if (len(device_list) > 1 and (local_rank == 0 or local_rank == device_list[0])) or (len(device_list) == 1):
            logging.error("dataset : %s, golden len is not equal infer, golden: %s, infer: %s", dataset,
                          len(golden_answer), len(infer_answer))
        answer_len = min(len(golden_answer), len(infer_answer))
    for idx in range(answer_len):
        if dataset in ["cmmlu_zeroshot", "mmlu_zeroshot", "bbh_zeroshot", "ceval_zeroshot",
                               "cmmlu_fewshot", "mmlu_fewshot", "bbh_fewshot", "ceval_fewshot"]:
            try:
                if golden_answer[idx][0].lower() == infer_answer[idx][0].lower():
                    similarity = 1
                else:
                    similarity = 0
            except Exception:
                similarity = 0
            match_score_list.append(similarity)
            sim_list.append(similarity)
        elif dataset in ["arithmetic_zeroshot"]:
            try:
                if golden_answer[idx] == infer_answer[idx]:
                    similarity = 1
                else:
                    similarity = 0
            except Exception:
                similarity = 0
            match_score_list.append(similarity)
            sim_list.append(similarity)
        else:
            try:
                match_score = SequenceMatcher(None, golden_answer[idx], infer_answer[idx]).quick_ratio()
                if match_score >= 0.45:
                    similarity = 1
                else:
                    similarity = 0
            except Exception:
                similarity = 0
            match_score_list.append(match_score)
            sim_list.append(similarity)
    if (len(device_list) > 1 and (local_rank == 0 or local_rank == device_list[0])) or (len(device_list) == 1):
        logging.info("match score: %s", match_score_list)
        logging.info("similarity list  : %s", sim_list)
        sim_total = sum(sim_list) / len(sim_list)
        logging.info("model: %s, params: %s, dataset: %s, total question: %s, precision total score: %s",
                     model_name, params, dataset, answer_len, sim_total)
    device_num = int(os.getenv("DEVICE_NUM", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    try:
        if (device_num > 1 and local_rank == 0) or (device_num == 1):
            ckpt_log_path = os.getenv("CKPT_LOG_PATH")
            with open(ckpt_log_path, 'r') as d:
                ckpt_context_list = json.load(d)
            if ckpt_context_list[0].get('Precision flag') == None:
                ckpt_context_list[0].update({
                    "Precision list": match_score_list,
                    "Precision list flag": 1,
                    "Precision": sim_total,
                    "Precision flag": 1
                })
            else:
                precision_flag = 0
                percision_list_flag = 1
                precision_golden = ckpt_context_list[0]["Precision"]
                precision_list_golden = ckpt_context_list[0]["Precision list"]
                if sim_total == precision_golden:
                    precision_flag = 1
                try:
                    for idx, sim_score in enumerate(match_score_list):
                        if sim_score != precision_list_golden[idx]:
                            percision_list_flag = 0
                except Exception:
                    percision_list_flag = 0
                ckpt_context_list[len(ckpt_context_list) -1].update({
                    "Precision list": match_score_list,
                    "Precision list flag": percision_list_flag,
                    "Precision": sim_total,
                    "Precision flag": precision_flag
                })
            ckpt_context_list_dump = json.dumps(ckpt_context_list, ensure_ascii=False, indent=4)
            f_answer = open(ckpt_log_path, 'w')
            f_answer.write(ckpt_context_list_dump)
            f_answer.close()
    except Exception:
        pass


# basic token generater
def generate_chat_prompt(bs):
    _PROMPTS = [
        {"role": "user", "content": "Write a piece of quicksort code in C++"},
    ]

    _PROMPTS = [_PROMPTS] * (bs // len(_PROMPTS) + 1)
    _PROMPTS = _PROMPTS[:bs]
    logging.info("prompt batch size: %d", len(_PROMPTS))
    return _PROMPTS


def generate_default_prompts():
    # prompts的size大小决定了模型执行时的batch size大小
    _PROMPTS = [
        "给出一段对话，使用合适的语气和回答方式继续对话。\n对话：\nA：你今天看起来很高兴，发生了什么好事？\nB：是的，我刚刚得到一份来自"
        "梅西银行的工作通知书。\nA：哇，恭喜你！你打算什么时候开始工作？\nB：下个月开始，所以我现在正为这份工作做准备。",
        # "用一句话描述地球为什么是独一无二的。",
        # "Let x = 1. What is x << 3 in Python 3? the answer is",
        # "In Python 3, what is ['a', 'Chemistry', 0, 1][-3]?",
        # "The study of older adults and aging is reffered to as",
        # "Why is the sky blue?",
        # "What's your name?",
        # "Hello my name is",
    ]
    return _PROMPTS


def generate_prompt(bs, tokenizer_mode):
    if tokenizer_mode == "default":
        return generate_default_prompts()
    else:
        return generate_chat_prompt(bs)

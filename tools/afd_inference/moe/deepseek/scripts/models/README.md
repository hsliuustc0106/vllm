---
license: other
license_name: deepseek
license_link: https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL
---

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V2" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Homepage" src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://chat.deepseek.com/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V2-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://huggingface.co/deepseek-ai" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank" style="margin: 2px;">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true" target="_blank" style="margin: 2px;">
    <img alt="Wechat" src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank" style="margin: 2px;">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-CODE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL" style="margin: 2px;">
    <img alt="Model License" src="https://img.shields.io/badge/Model_License-Model_Agreement-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  <a href="#2-model-downloads">Model Download</a> |
  <a href="#3-evaluation-results">Evaluation Results</a> |
  <a href="#4-model-architecture">Model Architecture</a> |
  <a href="#6-api-platform">API Platform</a> |
  <a href="#8-license">License</a> |
  <a href="#9-citation">Citation</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2405.04434"><b>Paper Link</b>👁️</a>
</p>

# DeepSeek-V2-Chat-0628

## 1. Introduction

DeepSeek-V2-Chat-0628 is an improved version of DeepSeek-V2-Chat. For model details, please visit [DeepSeek-V2 page](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat) for more information. 

DeepSeek-V2-Chat-0628 has achieved remarkable performance on the LMSYS Chatbot Arena Leaderboard:

Overall Ranking: #11, outperforming all other open-source models.

<p align="center">
  <img width="90%" src="figures/arena1.jpeg" />
</p>

Coding Arena Ranking: #3, showcasing exceptional capabilities in coding tasks.

<p align="center">
  <img width="90%" src="figures/arena2.png" />
</p>

Hard Prompts Arena Ranking: #3, demonstrating strong performance on challenging prompts.

<p align="center">
  <img width="90%" src="figures/arena3.png" />
</p>

## 2. Improvement

Compared to the previous version DeepSeek-V2-Chat, the new version has made the following improvements:

| **Benchmark** | **DeepSeek-V2-Chat** | **DeepSeek-V2-Chat-0628** | **Improvement** |
|:-----------:|:------------:|:---------------:|:-------------------------:|
| **HumanEval** | 81.1 | 84.8 | +3.7 |
| **MATH** | 53.9 | 71.0 | +17.1 |
| **BBH** | 79.7 | 83.4 | +3.7 |
| **IFEval** | 63.8 | 77.6 | +13.8 |
| **Arena-Hard** | 41.6 | 68.3 | +26.7 |
| **JSON Output (Internal)** | 78 | 85 | +7 |

Furthermore, the instruction following capability in the "system" area has been optimized, significantly enhancing the user experience for immersive translation, RAG, and other tasks.

## 3. How to run locally (on 71.20.45.105)
### Inference with torch_npu

**To utilize DeepSeek-V3 in FLOAT16 format for inference, 16 NPUs at least are required.**

There are servel config files in the scripts directory:
- set_daily_cfg.bash: used for setting dp/tp number, profiling switch.
- set_file_path.bash: used for setting result saving path
- set_model_config.bash: used for specifying eager/dynamo mode
- set_cann_env.bash: used for setting cann package path and log level.


The launch script is `infer_daily_dev.sh`.
The input parameters are:
- MODEL_DIR: path to weight .pts and model configuration file
- RUNNING_MODE: eager mode or dynamo mode
- ENABLE_MLA: always keep it as 1
- NEW_BATCH: prompts batch size
- NEW_LAYER: model decode layer numbers, 5 or 61
- NEW_WORLD: 16 when one node
- NEW_QUANT: 3
- RANK_OFFSET: rank offset start from 0. 0 for first node, 16 for the next node.


To run the model in dynamo mode, execute the following command:
```shell
conda activate zyj_py38_main
cd inference/moe/deepseek/scripts/
bash infer_daily_dev.sh /home/l00595113/deepseekv3-lite-base-latest_bugtest/ 1 1 128 5 16 3 0  # /home/l00595113/deepseekv3-lite-base-latest_bugtest/ is the  path to weight files and config files
```

Some fused vector ops are not allowed to run in the eager mode currently, such as `npu_interleave_rope`. As a result, if you want to run the model in the eager mode, disable `ENABLE_ROPE_DIM64,
ENABLE_KV_VECTOR,ENABLE_GATE_VECTOR,ENABLE_FFN_VECTOR` these four env variables in `models/global_setting.py`.


## 4. License
This code repository is licensed under [the MIT License](LICENSE-CODE). The use of DeepSeek-V2 Base/Chat models is subject to [the Model License](LICENSE-MODEL). DeepSeek-V2 series (including Base and Chat) supports commercial use.

## 5. Citation
```
@misc{deepseekv2,
      title={DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model}, 
      author={DeepSeek-AI},
      year={2024},
      eprint={2405.04434},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 6. Contact
If you have any questions, please raise an issue or contact us at [service@deepseek.com](service@deepseek.com).

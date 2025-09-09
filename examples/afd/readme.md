## AFD Demo Readme

本 Demo 展示了如何将 Transformer 模型中的 Attention 层与 FFN（MoE）层解耦，分别部署在不同进程甚至不同机器上，实现分布式推理。

---

###  环境准备

#### 1. 克隆并切换到对应分支
```bash
git clone https://github.com/hsliuustc0106/vllm.git
cd vllm
git fetch origin pull/12/head:afd-demo
git checkout afd-demo
```

#### 2. 安装依赖
```bash
pip install -r requirements.txt
pip install -e .
```

### 启动步骤

####  Step 1：启动 FFN 服务（MoE 层）

运行以下命令启动 FFN 推理服务（负责 MoE 层计算）：

```bash
python ffn_start.py 
```


####  Step 2 离线模式运行 Attention（offline_attn.py）

如果你只想在本地调试 Attention 层，不连接 FFN 服务，可使用离线模式：

```python
from vllm import LLM, SamplingParams

prompts = [
    "1 3 5 7 9",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
#llm = LLM(model="/data2/models/Qwen3-0.6B")
llm = LLM(model="/data2/models/deepseek-v2-lite", enforce_eager=True, additional_config={"role": "attn"})

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt{prompt!r}, generated text: {generated_text!r}")

```

>  说明：
- `--role attn`：表示当前进程仅运行 Attention 层。

---

#### Step 2 在线模式启动 Attention （online_attn.sh）

若要与 FFN 服务通信，需启动在线 Attention 服务：

```bash
chmod +x online_attn.sh
./online_attn.sh
```

`online_attn.sh` 示例内容如下（可按需修改）：

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
vllm serve /data2/models/deepseek-v2-lite --enforce_eager --additional-config='{"role":"attn"}'

```

>  说明：
- 该服务会将 Attention 输出通过 `_AFD_CONNECTOR` 发送给 FFN 服务，并接收其返回结果。
- 确保 `ffn_start.py` 已启动。

---

### 流程概览

```text
Input Prompt
     ↓
online_attn.sh (Attention服务)
     ↓
Attention Layer Output
     ↓
AFD_CONNECTOR.send_attn_output()
     ↓
ffn_start.py（FFN服务）
     ↓
MoE Layer Output
     ↓
AFD_CONNECTOR.recv_ffn_output()
     ↓
Final Output (online_attn.sh)
```

---

###  验证是否成功

####  检查日志输出
- `ffn_start.py` 日志中应出现：
```
INFO ffn decoder layer X forwarding
```

- `online_attn.sh` 日志中应出现：
```
INFO attn decoder X forwarding
```

####  测试请求（在线模式）

使用 curl 或浏览器访问：
```bash
curl -v http://0.0.0.0:8000/v1/chat/completions \
-H 'Content-Type: application/json' \
-d \
'{ "model": "/data2/models/deepseek-v2-lite",
"messages": [
{"role": "user", "content": "1 3 5 7 9"} ],
"temperature": 0.6,
"repetition_penalty": 1.0,
"top_p": 0.95,
"top_k": 40,
"max_tokens": 20,
"stream": false}'
```

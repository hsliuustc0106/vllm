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

以2A2F配置为例，运行以下命令启动 FFN 服务（负责 MoE 层计算）：

```bash
export NCCL_SOCKET_IFNAME=eno1 # 在跨机执行时需要配置NCCL和GLOO使用的网卡
export GLOO_SOCKET_IFNAME=eno1

export MASTER_IP=<master_ip> # 在跨机执行时需要配置master节点的ip和端口信息
export MASTER_PORT=<master_port>

export CUDA_VISIBLE_DEVICES=0,1
python fserve.py --model="/home/models/DeepSeek-V2-Lite" --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager --additional-config='{"role":"ffn", "afd_size":"2A2F"}'
```

>  说明：
- 通过role来指定进程角色。
- afd_size指的是attn和ffn分别使用的卡数。符合xAyF的格式。



---

#### Step 2 启动 Attention （online_attn.sh）

若要与 FFN 服务通信，需启动在线 Attention 服务：
```bash
#!/bin/bash
export NCCL_SOCKET_IFNAME=eno1 # 在跨机执行时需要配置NCCL和GLOO使用的网卡
export GLOO_SOCKET_IFNAME=eno1

export MASTER_IP=<master_ip> # 在跨机执行时需要配置master节点的ip和端口信息
export MASTER_PORT=<master_port>

export CUDA_VISIBLE_DEVICES=0,1
vllm serve /data2/models/deepseek-v2-lite --enforce_eager --additional-config='{"role":"attn", "afd_size":"2A2F"}'

```
>  说明：
- 通过role来指定进程角色。
- 该服务会将 Attention 输出通过 `afd_connector` 发送给 FFN 服务，并接收其返回结果。
- 确保 `fserve.py` 已启动。


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
日志中出现以下内容说明成功拉起服务：
```plain
(APIServer pid=73628) INFO:     Started server process [73628]
(APIServer pid=73628) INFO:     Waiting for application startup.
(APIServer pid=73628) INFO:     Application startup complete.

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

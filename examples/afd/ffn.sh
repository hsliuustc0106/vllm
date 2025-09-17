#export NCCL_SOCKET_IFNAME=eno1
#export GLOO_SOCKET_IFNAME=eno1

python fserve.py --model="/data2/models/deepseek-v2-lite" --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager --additional-config='{"role":"ffn", "afd_size":"2a2f"}'

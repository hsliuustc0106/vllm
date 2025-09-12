vllm serve /data2/models/deepseek-v2-lite  --tensor_parallel_size=2 --enable_expert_parallel --enforce_eager --additional-config='{"role":"attn", "ffn_size":2, "attn_size":2}'


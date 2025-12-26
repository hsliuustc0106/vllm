import os
exe_mode = os.getenv("EXE_MODE")

# "weight_qb", "weight_kva" and "weight_qa" must be weight_nz, kv_b_proj_w_k must be ND, dataType must be bfloat16
ENABLE_MLA_PROLOG = True and (exe_mode == "dynamo")

# global settings
ENABLE_STREAM = True and (exe_mode == "dynamo")

# mla kv cahce sequence length setting (also set as kv_len when FA)
ACTUAL_SEQ_LEN = int(os.getenv("ACTUAL_SEQ_LEN", "256"))  # it should be no more than INPUT_MAX_LEN+MAX_NEW_TOKENS

# paged attention settings in mla 
ENABLE_PAGE_ATTENTION = True
_PAGE_ATTENTION_SETTING = {
    "block_size": 128,
    "max_length": ACTUAL_SEQ_LEN,
}

# setting-dict
_TODO_REQUIRE_API = {
    "actual_seq_len": ACTUAL_SEQ_LEN,
    "enable_stream": ENABLE_STREAM,
    "enable_pa": ENABLE_PAGE_ATTENTION,
    "enable_mla_prolog": ENABLE_MLA_PROLOG,
}

PREFETCH_SIZE = 141557760 # 135M data
FFN1_PREFETCH_SIZE = 56*1024*1024
FFN2_PREFETCH_SIZE = 14*1024*1024
LM_HEAD_PREFETCH_SIZE = 56*1024*1024
OPROJ_PREFETCH_SIZE = 28*1024*1024

#!/bin/bash

rm -rf /ascend_log
rm -rf /root/ascend

bash infer.sh --MODEL_DIR=/workspace/j00586476/deepseekv3-lite-base-latest_bugtest \
              --EXE_MODE="dynamo" \
              --FFN_MODE="dynamo" \
              --BATCH_SIZE=4320 \
              --LAYER_NUM=10 \
              --WORLD_SIZE=16 \
              --ATTN_DIES=12 \
              --NEXT_N=1 \
              --N_ROUTED_EXPERTS_PER_RANK=3 \
              --REMAINDER_ROUTER_EXPERT=0 \
              --EXPERTS_SHARE_NUM_COPY=1 \
              --DENSE_TP_SIZE=4 \
              --ON_CLOUD=0 \
              --ENABLE_CACHE_COMPILE=0 \
              --ENABLE_SUPERKERNEL=0 \
              --ENABLE_PREFETCH=1 \
              --LAYER_OUT="FA" \
              --ACTUAL_SEQ_LEN=4096 \
              --ENABLE_BATCH_WITH_RECV=1 \
              --USE_REAL_ACTUAL_SEQ_LEN=0 \
              --TOKEN_FILE_PATH=/home/j00586476/tokens.xlsx \
              --ATTN_FFN_START_SYNC=0 \
              --WITH_CKPT=0 \
              --ATTN_TP_SIZE=1 \
              --OPROJ_TP_SIZE=1 \
              --EXPERTS_TP_SIZE=1 \
              --ENABLE_PROFILE=1 \
              --MAX_NEW_TOKENS=120

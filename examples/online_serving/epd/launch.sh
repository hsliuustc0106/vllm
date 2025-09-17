#!/usr/bin/env bash
set -euo pipefail

export VLLM_VERSION=0.10.0

wait_for_server() {
    local port=$1
    timeout 12000 bash -c '
        until curl -s "http://localhost:'"$port"'/v1/chat/completions" > /dev/null; do
            sleep 1
        done
    ' && return 0 || return 1
}

MODEL="${MODEL:-Qwen2.5-VL-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-vl-3b-instruct}"

LOG_PATH=${LOG_PATH:-./logs}
mkdir -p "$LOG_PATH"

ENCODE_PORT=19534
PREFILL_DECODE_PORT=19535
PROXY_PORT=10001

GPU_E=0
GPU_PD=1

START_TIME=$(date +"%Y%m%d_%H%M%S")
ENC_LOG="$LOG_PATH/encoder.log"
PD_LOG="$LOG_PATH/pd.log"
PROXY_LOG="$LOG_PATH/proxy.log"
PID_FILE="./pid.txt"

SHARED_STORAGE_PATH="/dev/shm/epd"

###############################################################################
# Encoder worker
###############################################################################
start_encoder_worker() {
    ASCEND_RT_VISIBLE_DEVICES="$GPU_E" vllm serve "$MODEL" \
        --gpu-memory-utilization 0.0 \
        --port "$ENCODE_PORT" \
        --enable-request-id-headers \
        --no-enable-prefix-caching \
        --max-num-seqs 128 \
        --max-model-len 3072 \
        --max-num-batched-tokens 40000 \
        --served-model-name "$MODEL_NAME" \
        --enforce-eager \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_producer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$SHARED_STORAGE_PATH"'",
                "ec_max_num_scheduled_tokens": "4096"
            }
        }' \
        >"$ENC_LOG" 2>&1 &

    echo $! >> "$PID_FILE"
}

start_encoder_worker

###############################################################################
# Prefill / decode worker
###############################################################################
start_prefill_decode_worker() {
    ASCEND_RT_VISIBLE_DEVICES="$GPU_PD" vllm serve "$MODEL" \
        --gpu-memory-utilization 0.95 \
        --port "$PREFILL_DECODE_PORT" \
        --no-enable-prefix-caching \
        --enable-request-id-headers \
        --max-num-seqs 128 \
        --max-model-len 3072 \
        --max-num-batched-tokens 40000 \
        --served-model-name "$MODEL_NAME" \
        --enforce-eager \
        --ec-transfer-config '{
            "ec_connector": "ECSharedStorageConnector",
            "ec_role": "ec_consumer",
            "ec_connector_extra_config": {
                "shared_storage_path": "'"$SHARED_STORAGE_PATH"'"
            }
        }' \
        >"$PD_LOG" 2>&1 &

    echo $! >> "$PID_FILE"
}

start_prefill_decode_worker

# Wait until both workers are ready
# wait_for_server "$ENCODE_PORT"
# wait_for_server "$PREFILL_DECODE_PORT"

###############################################################################
# Proxy
###############################################################################
start_proxy() {
    python proxy.py \
        --host "127.0.0.1" \
        --port "$PROXY_PORT" \
        --encode-servers-urls "http://localhost:$ENCODE_PORT" \
        --prefill-decode-servers-urls "http://localhost:$PREFILL_DECODE_PORT" \
        >"$PROXY_LOG" 2>&1 &

    echo $! >> "$PID_FILE"
}

start_proxy

# wait_for_server "$PROXY_PORT"
echo "All services are up!"
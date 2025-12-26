#!/bin/bash
custom_dir=/usr/local/Ascend/ascend-toolkit/latest
if [ -d ${custom_dir} ]; then
    cann_path=${custom_dir}
else
    cann_path=/usr/local/Ascend/latest
fi
echo "cann_path >>> " ${cann_path}
export ASCEND_HOME_PATH=${cann_path}
source ${cann_path}/bin/setenv.bash
export LD_LIBRARY_PATH=${cann_path}/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}

if [ ${ON_CLOUD} -eq 0 ]; then
    rm -rf cache
    rm -rf kernel_meta*
    rm -rf dynamo_*
    rm -rf /root/atc_data
    rm -rf /root/.cache/*
    rm -rf .torchair_cache
fi

# set log level: DEBUG:0, INFO:1, WARNNING:2, ERROR:3;
# export ASCEND_GLOBAL_LOG_LEVEL=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1
# export ASCEND_GLOBAL_EVENT_ENABLE=1
# export ASCEND_MODULE_LOG_LEVEL=FE=0
export ASCEND_MC2_DEBUG_MODE=1  # moe dispatch in eager mode: 1
export ASCEND_PROCESS_LOG_PATH="${WORK_DIR}/${RES_PATH}/ascend_log"
export TNG_HOST_COPY=1  # enable torchair optimize

# dump ge graph
# export DUMP_GE_GRAPH=2 # 全量
# export DUMP_GRAPH_LEVEL=2 # 阶段
# export DUMP_GRAPH_PATH="${RES_PATH}/ge_graph"
# export TNG_LOG_LEVEL=0
export PYTHONPATH=$PYTHONPATH:./

# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1
# export ASCEND_LAUNCH_BLOCKING=1

# 以下两个不支持设置为0
export ASCEND_ATTN_TO_FFN_WIN_TYPE=1
export ASCEND_FFN_TO_ATTN_WIN_TYPE=1

# export ASCEND_OP_COMPILE_SAVE_KERNEL_META=1

#export ASCEND_FFN_TO_ATTN_AIV_NUM=24
#export FFN_WORKER_BATCHING_CORE_NUM=24

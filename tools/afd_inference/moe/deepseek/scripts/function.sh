#!/bin/bash
function check_env_vars()
{
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
#    LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`       # 获取本节点IP
#    LOCAL_HOST="90.90.97.36"       # 获取本节点IP
    if [ ${ON_CLOUD} -eq 0 ]; then
        export HCCL_SOCKET_IFNAME=enp194s0f0
#        IPs=('71.20.45.104') # '71.20.45.116' )           # 所有节点的IP，确保第1个IP是master
        IPs=('90.90.97.34' '90.90.97.36' '90.90.97.40' '90.90.97.42') # '71.20.45.116' )           # 所有节点的IP，确保第1个IP是master
        # IPs=('90.90.97.40') # '71.20.45.116' )           # 所有节点的IP，确保第1个IP是master
        # IPs=('90.90.97.36') # '71.20.45.116' )           # 所有节点的IP，确保第1个IP是master
        MA_NUM_HOSTS=${#IPs[@]}                           # 节点数量
        export MASTER_ADDR=${IPs[0]}                      # 主节点IP
        export MASTER_PORT=6039                           # 主节点port
        VC_TASK_INDEX=0                                   # 本节点index
        # 获取每个节点的index
        for i in "${!IPs[@]}";
        do
            echo "LOCAL_HOST=${LOCAL_HOST}, IPs[${i}]=${IPs[$i]}"
            if [ "$LOCAL_HOST" == "${IPs[$i]}" ]; then
                echo "Node Rank : ${i}"
                VC_TASK_INDEX=$i
                break
            fi
        done
    else
        . /home/ma-user/anaconda3/etc/profile.d/conda.sh
        conda activate lh_py311
        echo "Python version >>>" `python3 -V`
        export HCCL_SOCKET_IFNAME=eth0
        export MASTER_ADDR=`echo $VC_WORKER_HOSTS|awk -F "," '{print $1}'`
        export MASTER_PORT=6138                            # 主节点port
    fi
    echo "VC_TASK_INDEX >>>" $VC_TASK_INDEX

    export MA_NUM_GPUS=16                       # 一个节点上的卡数，每个节点应该一样
    # export MA_NUM_GPUS=12                       # 一个节点上的卡数，每个节点应该一样
    export RANK_OFFSET=`expr $VC_TASK_INDEX \* ${MA_NUM_GPUS}`

    # source model_config
    if [ ${ENABLE_MLA} -eq 1 ]; then
        MODEL_NAME="deepseek_v2"
        echo '>>>> MODEL_NAME is '${MODEL_NAME}', RunningMode is '${EXE_MODE}
        export TOKENIZER_MODE="chat"

    else
        echo "Not Supported ENABLE_MLA: "${ENABLE_MLA}
        exit 0
    fi

    if [ ${ENABLE_CACHE_COMPILE} -eq 1 ] && [ ${EXE_MODE} != "dynamo" ]; then
        echo "[ERROR]: ONLY IN DYNAMO MODE, CAN CACHE_COMPILE BE ENABLED !"
        exit
    fi

    if [ ${PREFILL_OR_DECODE} != "prefill" ] &&  [ ${PREFILL_OR_DECODE} != "decode" ]; then
        echo "[ERROR]: ONLY SUPPORT PREFILL OR DECODE RESPECTIVELY !"
        exit
    fi

    # check world size
    if [ $MA_NUM_HOSTS ]; then
        export DEVICE_SIZE=$(($MA_NUM_GPUS*$MA_NUM_HOSTS))  # 所有节点总卡数
        if [ ${DEVICE_SIZE} -ge  ${WORLD_SIZE} ]; then
            echo "[INFO] total ranks is ${DEVICE_SIZE}, and use ${WORLD_SIZE} ranks in actual!"
        else
            echo "[ERROR] total ranks is ${DEVICE_SIZE}, but use ${WORLD_SIZE} ranks in actual!"
            exit 0
        fi
    fi

    # calc FFN_DIES. Each die corresponds to a attn/ffn worker process
    export FFN_DIES=$(expr ${WORLD_SIZE} - ${ATTN_DIES})
    export ASCEND_MC2_FFN_NUM=${FFN_DIES}

    export TASK_QUEUE_ENABLE=2 # eager mode:opt host perf

    # attn setting
    export VOCAB_RANKS=1  # 1: vocab embed use DP strategy; WORLD_SIZE: use TP strategy
    export ATTN_DP_SIZE=`expr ${ATTN_DIES} \/ ${ATTN_TP_SIZE}`
    [[  `expr $ATTN_TP_SIZE \* $ATTN_DP_SIZE`  -ne  $ATTN_DIES  ]] &&  echo  "ERROR: dp * tp != world_size"  &&  exit  1

    # moe setting
    if [ ${ROUTE_SHARE_ON_SAME_CARD} -eq 1 ] && [ ${EXPERTS_TP_SIZE} -eq 2 ] ; then
        echo  "ERROR: not support ROUTE_SHARE_ON_SAME_CARD=1 and EXPERTS_TP_SIZE=2 !";
        exit  1;
    fi;

    if [ ${WORLD_SIZE} -eq 16 ]; then
        export ENABLE_EXPERT_ADPT=1
        # export EXPERTS_SHARE_NUM_COPY=1
        if [ ${ROUTE_SHARE_ON_SAME_CARD} -eq 1 ]; then
            EXPERTS_SHARE_NUM_COPY=${WORLD_SIZE}
            # use user prefined N_ROUTED_EXPERTS_PER_RANK
#        else
#            export N_ROUTED_EXPERTS_PER_RANK=1
        fi
    else
        export ENABLE_EXPERT_ADPT=0 # 0 for full-version
        if [ ${ROUTE_SHARE_ON_SAME_CARD} -eq 1 ]; then
            EXPERTS_SHARE_NUM_COPY=${WORLD_SIZE}
            # use user prefined N_ROUTED_EXPERTS_PER_RANK
        else
#            export N_ROUTED_EXPERTS_PER_RANK=1
            if [ ${EXPERTS_SHARE_NUM_COPY} -ge `expr ${WORLD_SIZE} \/ ${EXPERTS_TP_SIZE}` ] ; then
                echo  "ERROR: EXPERTS_SHARE_NUM_COPY(${EXPERTS_SHARE_NUM_COPY}) is too large !";
                exit  1;
            fi;
        fi
    fi

    NAME_PREFIX="test"
    ATTN_MODE="Attn${ATTN_DP_SIZE}d${ATTN_TP_SIZE}t"
    MOE_MODE="MoeAdpt${ENABLE_EXPERT_ADPT}_${EXPERTS_TP_SIZE}die"
    MODEL_RUN_MODE="L${LAYER_NUM}_In${INPUT_MAX_LEN}_${DTYPE}"
    MTP_NUM="MTP${NEXT_N}"
    MODEL_NAME="${NAME_PREFIX}_${ATTN_MODE}_${MOE_MODE}_${MODEL_RUN_MODE}_${MTP_NUM}"
    echo "========================================>"
    if [ $NEXT_N -ne 0 ];then
        echo "Enable MTP with next_n ="${NEXT_N}
    else
        echo "MTP not applied"
    fi
    echo "==============set_file_path==============>"
    DATE=`date +%Y%m%d` 
    # set result path
    DIR_PREFIX="res"
    NAME=${MODEL_NAME}_${WORLD_SIZE}p_a${ATTN_DIES}f${FFN_DIES}_${EXE_MODE}_BS${BATCH_SIZE}_Q${QUANT_MODE}_${PREFILL_OR_DECODE}
    
    if [ ${ON_CLOUD} -eq 0 ]; then
        export RES_PATH="${DIR_PREFIX}/${DATE}/${NAME}"
        WORK_DIR=`pwd`
        DUMP_PRECISION_PATH=${WORK_DIR}'/'${RES_PATH}'/dump_data'
        mkdir -p ${WORK_DIR}'/'${RES_PATH}
        mkdir -p ${DUMP_PRECISION_PATH}
    else
        export RES_PATH="${DIR_PREFIX}/${DATE}/${NAME}/${VC_TASK_INDEX}"
        WORK_DIR='/home/ma-user/modelarts/outputs/train_url_0'
        DUMP_PRECISION_PATH=${WORK_DIR}'/'${RES_PATH}'/dump_data'
        mkdir -p ${DUMP_PRECISION_PATH}
    fi

    # set profiling path
    if [ ${ENABLE_PROFILE} -eq 0 ]; then
        PROFILING_PATH=""
    else
        if [ ${ON_CLOUD} -eq 0 ]; then
            PROFILING_PATH="${RES_PATH}/prof" # profiling_dump_path
        else
            PROFILING_PATH=${WORK_DIR}'/'${RES_PATH}'/prof'
            mkdir -p ${PROFILING_PATH}
        fi
    fi
    echo 'result save to' ${WORK_DIR}'/'${RES_PATH}', profiling saved to '${PROFILING_PATH}

    source set_cann_env.bash
    echo "==================================>"

    export HCCL_IF_IP=$LOCAL_HOST
    export HCCL_IF_BASE_PORT=27544
    if [ ${PREFILL_OR_DECODE} == "decode" ]; then
        export HCCL_BUFFSIZE=200
        SPEC_LEN=`expr ${NEXT_N} \+ 1`
        # 计算每die上的batch size，考虑MTP
        BS_PER_DIE=$(( BATCH_SIZE / ATTN_DIES * SPEC_LEN))
        # 计算DISPATH_SIZE, 应该用EP_WORLD_SIZE，假定二者一致
        DISPATH_SIZE=$(( BS_PER_DIE * WORLD_SIZE * 7168 * N_ROUTED_EXPERTS_PER_RANK * 2 * 2 ))
        # 计算COMBINE_SIZE, 9是top8+1得到
        COMBINE_SIZE=$(( BS_PER_DIE * 7168 * 9 * 2 * 2 ))
        # 计算新的HCCL缓冲区大小
        NEW_HCCL_BUFFSIZE=$(( (DISPATH_SIZE + COMBINE_SIZE) / 1048576 ))
        # 输出结果
        if [ ${NEW_HCCL_BUFFSIZE} -gt ${HCCL_BUFFSIZE} ]; then
            HCCL_BUFFSIZE=`expr ${NEW_HCCL_BUFFSIZE} \+ 8`
        fi
        export HCCL_BUFFSIZE=$HCCL_BUFFSIZE
        echo "HCCL_BUFFSIZE = " ${HCCL_BUFFSIZE}
    fi
    # 910c needs enable HCCL aiv
    export HCCL_OP_EXPANSION_MODE=AIV
    # export HCCL_DETERMINISTIC=true  # 开启确定性计算
    # unset HCCL_OP_EXPANSION_MODE
    export HCCL_CONNECT_TIMEOUT=1200
    export HCCL_EXEC_TIMEOUT=600
    export HCCL_OP_COUNTER_ENABLE=0

    echo "MODEL_NAME >>>" $MODEL_NAME
    echo "MODEL_DIR >>>" $MODEL_DIR
    echo "BATCH_SIZE >>>" $BATCH_SIZE
    echo "INPUT_MAX_LEN >>>" $INPUT_MAX_LEN
    echo "MAX_NEW_TOKENS >>>" $MAX_NEW_TOKENS
    echo "LAYER_NUM >>>" $LAYER_NUM
    echo "WORLD_SIZE >>>" $WORLD_SIZE
    echo "QUANT_MODE >>>" $QUANT_MODE
    echo "NEXT_N >>>" $NEXT_N
    echo "DTYPE >>>" $DTYPE
    echo "ATTN_TP_SIZE >>>" $ATTN_TP_SIZE
    echo "ROUTE_SHARE_ON_SAME_CARD >>>" $ROUTE_SHARE_ON_SAME_CARD
    echo "ENABLE_EXPERT_ADPT >>>" $ENABLE_EXPERT_ADPT
    echo "EXPERTS_SHARE_NUM_COPY >>>" $EXPERTS_SHARE_NUM_COPY
    echo "N_ROUTED_EXPERTS_PER_RANK >>>" $N_ROUTED_EXPERTS_PER_RANK
    echo "TOKENIZER_MODE >>>" $TOKENIZER_MODE
    echo "EXE_MODE >>>" $EXE_MODE
    echo "ENABLE_PROFILE >>>" $ENABLE_PROFILE
    echo "EXPERTS_TP_SIZE >>>" $EXPERTS_TP_SIZE
    echo "ACTUAL_SEQ_LEN >>>" $ACTUAL_SEQ_LEN
    echo "ENABLE_CACHE_COMPILE >>>" $ENABLE_CACHE_COMPILE
    echo "ENABLE_SUPERKERNEL >>>" $ENABLE_SUPERKERNEL
    echo "MOE_USE_ALL_TO_ALL >>>" $MOE_USE_ALL_TO_ALL
    echo "ENABLE_COMBINE_DEQUANT >>>" $ENABLE_COMBINE_DEQUANT
    echo "PREFILL_OR_DECODE >>>" $PREFILL_OR_DECODE
    echo "EMBED_TP_SIZE >>>" $EMBED_TP_SIZE
    echo "ENABLE_MICRO_BATCH >>>" $ENABLE_MICRO_BATCH
    echo "ENABLE_GMM_TUNE_CONFIG >>>" $ENABLE_GMM_TUNE_CONFIG
}

function launch_python_task()
{
    cores=`cat /proc/cpuinfo|grep "processor" |wc -l`
    avg_core_per_rank=`expr $cores \/ $MA_NUM_GPUS`
    core_gap=`expr $avg_core_per_rank \- 1`
    for((i=0; i<${MA_NUM_GPUS}; i++))
    do
        start=`expr $i \* $avg_core_per_rank`
        end=`expr $start \+ $core_gap`
        cmdopt=$start"-"$((start+3))","$((start+5))"-"$end
        export LOCAL_RANK=$i
        export RANK=$(expr $i + $RANK_OFFSET)
        export RANK_ID=$RANK
        echo $i $cmdopt $RANK_ID
        if [ $i -eq 0 ];then
        taskset -c $cmdopt python3 infer.py \
                    --model_name=${MODEL_NAME} --model_path $MODEL_DIR \
                    --input_max_len=${INPUT_MAX_LEN} --max_new_tokens=${MAX_NEW_TOKENS} --batch_size=${BATCH_SIZE} \
                    --tokenizer_mode=${TOKENIZER_MODE} --execute_mode=${EXE_MODE} \
                    --profiling_path=${PROFILING_PATH} \
                    --dump_precision_path=${DUMP_PRECISION_PATH} \
                    --enable_mla=${ENABLE_MLA} \
                    --next_n=${NEXT_N} 2>&1 | tee ${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log &
        else
        taskset -c $cmdopt python3 infer.py \
                    --model_name=${MODEL_NAME} --model_path $MODEL_DIR \
                    --input_max_len=${INPUT_MAX_LEN} --max_new_tokens=${MAX_NEW_TOKENS} --batch_size=${BATCH_SIZE} \
                    --tokenizer_mode=${TOKENIZER_MODE} --execute_mode=${EXE_MODE} \
                    --profiling_path=${PROFILING_PATH} \
                    --dump_precision_path=${DUMP_PRECISION_PATH} \
                    --enable_mla=${ENABLE_MLA} \
                    --next_n=${NEXT_N} &> ${WORK_DIR}/${RES_PATH}/log_${LOCAL_RANK}.log &
        fi
    done
}

function save_key_info()
{
    wait
    if [ ${ON_CLOUD} -eq 1 ]; then
        mkdir -p ${WORK_DIR}/log_${VC_TASK_INDEX}
        cp -r /home/ma-user/modelarts/log ${WORK_DIR}/log_${VC_TASK_INDEX}
    fi
    last_worker_index=`expr $MA_NUM_HOSTS \- 1`
    if [ ${ON_CLOUD} -eq 1 ] && [ ${VC_TASK_INDEX} -eq ${last_worker_index} ]; then
        echo "===================start to save key infos"
        cur_dir=`pwd`
        key_info_dir=${WORK_DIR}/info/

        cann_info_dir=${key_info_dir}/cann/
        log_info_dir=${key_info_dir}/log/
        prof_info_dir=${key_info_dir}/prof/
        dump_info_dir=${key_info_dir}/dump/
        code_info_dir=${key_info_dir}/code/

        mkdir -p ${cann_info_dir}
        mkdir -p ${log_info_dir}
        mkdir -p ${prof_info_dir}
        mkdir -p ${dump_info_dir}
        mkdir -p ${code_info_dir}

        cp -r ${cur_dir}/../../../../ma-pre-start.sh ${cann_info_dir}/
        cat /usr/local/Ascend/CANN*/*/version.info |grep timestamp > ${cann_info_dir}/timestamp.txt
        cp ${cur_dir}/../config/output.yaml ${key_info_dir}/
        cp ${WORK_DIR}/${RES_PATH}/log_*.log ${log_info_dir}/
        cp -r ${PROFILING_PATH} ${prof_info_dir}/
        cp -r ${DUMP_PRECISION_PATH} ${dump_info_dir}/
        cp -r ${cur_dir}/../../../../inference/ ${code_info_dir}/
    fi
}

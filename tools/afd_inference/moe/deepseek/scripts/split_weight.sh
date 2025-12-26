cann_path=/usr/local/Ascend/latest
source ${cann_path}/bin/setenv.bash # 昇腾cann包安装目录
export ASCEND_HOME_PATH=${cann_path}
source set_model_config.bash 0 1
source set_daily_cfg.bash 128 5 16 3 0
export WORLD_SIZE=16
export LAYER_NUM=5
export MA_NUM_GPUS=16

quantize_model=0  # 0: save model in the original dtype; 1: save model in int8
path_model_origin=$1
path_model_after_tp=$2

python split_weight.py --model-path ${path_model_origin} --output-path ${path_model_after_tp}\
                        --quantize-model ${quantize_model} --ep_size 16 --world-size ${WORLD_SIZE}

#!/bin/bash

# 同步的源目录
SOURCE_DIR="/workspace/j00586476/sources/attn-ffn-separate-dev"

# 目标节点。注意为了避免执行中输入password，先将目标节点与源节点建立互信
TARGET_SERVERS=(
              "root@90.90.97.34:/workspace/j00586476/sources/attn-ffn-separate-dev"
              "root@90.90.97.36:/workspace/j00586476/sources/attn-ffn-separate-dev"
              "root@90.90.97.40:/workspace/j00586476/sources/attn-ffn-separate-dev"
              "root@90.90.97.42:/workspace/j00586476/sources/attn-ffn-separate-dev"
)
EXCLUDES="/workspace/j00586476/sources/attn-ffn-separate-dev/attn-ffn-separate-dev_0902_eager/inference/moe/deepseek/scripts/res/*"

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "源目录 $SOURCE_DIR 不存在，请检查。"
    exit 1
fi

for target in "${TARGET_SERVERS[@]}"; do
    rsync -avrz  --exclude $EXCLUDES --exclude "/workspace/j00586476/sources/attn-ffn-separate-dev/attn-ffn-separate-dev_0902_eager/inference/moe/deepseek/scripts/extra-info/*" --delete "$SOURCE_DIR/" "$target"
    if [ $? -ne 0 ]; then
        echo "同步到 $target 失败，请检查网络连接或目标服务器状态。"
    else
        echo "成功同步到 $target"
    fi
done
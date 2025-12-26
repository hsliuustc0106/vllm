#!/bin/bash
kill -9 $(ps -ef| grep 'infer.py'| grep python3 | grep -v grep| awk '{print $2}')  #需手动杀死进程
kill -9 $(ps -ef| grep 'ffn.py'| grep python3 | grep -v grep| awk '{print $2}')  #需手动杀死进程
if pgrep -f "python.*infer.py" > /dev/null; then
    echo "检测到有执行infer.py的Python进程正在运行,脚本中断退出。"
    exit 1
else
    echo "未检测到执行infer.py的Python进程。"
fi

source function.sh

if [ $# -gt 0 ]; then
    echo ">>>>>>> try to set yaml"
    python parse_shell_and_dump.py "$@"
    eval $(python load_param_and_set.py --yaml_file=output.yaml)
else
    echo ">>>>>>> perform default yaml"
    eval $(python load_param_and_set.py --yaml_file=default.yaml)
fi

check_env_vars
launch_python_task
save_key_info
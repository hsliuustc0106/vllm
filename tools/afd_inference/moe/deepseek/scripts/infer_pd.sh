#!/bin/bash
kill -9 $(ps -ef| grep 'infer.py'| grep python | grep -v grep| awk '{print $2}')

source function.sh

echo "begin to run prefill model"
python parse_shell_and_dump.py "$@" --PREFILL_OR_DECODE=prefill
eval $(python load_param_and_set.py --yaml_file=output.yaml)
check_env_vars
launch_python_task

wait
sleep 5
echo "begin to run decode model"
python parse_shell_and_dump.py "$@" --PREFILL_OR_DECODE=decode
eval $(python load_param_and_set.py --yaml_file=output.yaml)
check_env_vars
launch_python_task
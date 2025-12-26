#!/bin/bash
if pgrep -f "python.*infer.py" > /dev/null; then
    echo "检测到有执行infer.py的Python进程正在运行,脚本中断退出。"
    exit 1
else
    echo "未检测到执行infer.py的Python进程。"
fi
#kill -9 $(ps -ef| grep 'infer.py'| grep python | grep -v grep| awk '{print $2}')  #需手动杀死进程

source function.sh

function check_result()
{
    file=${WORK_DIR}/${RES_PATH}/log_0.log
    echo "check" $file

    if [ ! -f "$file" ]; then
        echo "ERROR: log" $file "not exist."
        exit 1
    fi
    error_str=`grep "ERROR" $file`
    if [ -n "$error_str" ]; then
        echo "CASE" ${CASE_N} "found ERROR, plz check"
        exit 1
    fi
}

for CASE_N in {1..10};do
    echo '----------------start test case' $CASE_N '----------------'
    eval $(python load_param_and_set.py --yaml_file=test_case/case${CASE_N}.yaml)
    check_env_vars
    launch_python_task
    wait
    check_result
    echo '----------------finish test case' $CASE_N '----------------'
done
echo '----------------all case pass----------------'
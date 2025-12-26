#!/bin/bash
PROFILING_DIR=$1

array=("FRAMEWORK" "PROF*" "logs")

WORK_DIR=`pwd`
cd ${WORK_DIR}/${PROFILING_DIR}/prof/incre
ALL_PROF=$(ls)

for x in ${ALL_PROF}
do
    for i in "${array[@]}"
    do
        result=`find ${x} -name "${i}"`
        # echo "find result:${result}"
        rm -rf ${result}
    done
done
echo "Done..."
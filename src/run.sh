#! /bin/bash
############################################
#
# Author: liuanan
# Create time: 2023-04-29 15:17:38
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

log_dir=../log/
mkdir -p ${log_dir}
rm ${log_dir}/*
sh kill.sh
if [ $# -eq 0 ]
then
    echo "Usage: --mode"
    exit 1
fi
while [[ $# -gt 0 ]]
do
    case "$1" in
      --mode)
        mode=$2
        shift
        ;;
      *)
        echo "error: unknow options -> $1"
        echo "Usage: --mode"
        ;;
    esac
    shift
done

if [[ "$mode" == "train" ]]
then
    nohup python3 run.py $mode > ${log_dir}/run.log 2> ${log_dir}/run.err &
elif [[ "$mode" == "game" ]]
then
    python3 run.py $mode
else
    echo "unknown mode: ${mode}"
fi

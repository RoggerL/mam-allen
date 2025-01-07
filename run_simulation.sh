#!/bin/bash

# 设置日志文件存储的目录
log_dir="/home/liugangqiang/log"

# 确保日志目录存在
mkdir -p "$log_dir"

# 获取当前日期和时间，精确到秒
current_datetime=$(date +%Y%m%d-%H_%M_%S)

# 构建日志文件的完整路径
log_file="$log_dir/${current_datetime}.log"

# 打印日志文件路径
echo "日志文件将被保存到: $log_file"

# 运行Python程序，并将输出重定向到日志文件
nohup /home/liugangqiang/miniconda3/envs/pygenn/bin/python /home/liugangqiang/mam-allen-genn20240404/run_example_downscaled-analysis.py >$log_file 2>&1 &
# nohup /home/liugangqiang/miniconda3/envs/pygenn/bin/python /home/liugangqiang/mam-allen-genn20240404/run_example_downscaled-analysis-V1.py >$log_file 2>&1 &
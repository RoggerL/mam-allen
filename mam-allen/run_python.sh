#!/bin/bash

# 获取python路径
echo "python路径为: $1"

# 定义要检查的用户名
USER="liugangqiang"

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

# 要求输入备注
echo "Please enter a remark before starting the monitor:"
read remark

# 如果没有输入备注，则不运行程序
if [ -z "$remark" ]; then
    echo "No remark provided. Exiting."
    exit 1
else
    # 将备注记录信息输出到日志文件
    echo "Remark recorded: $remark" >> $log_file
fi

# 定义一个函数来检查Python进程是否存在
check_python_process() {
    ps -u $USER | grep -q python
}

# 如果Python进程存在，则脚本持续运行，直到所有Python进程结束
monitor() {
    while check_python_process; do
        echo "Python process is running..."
        # echo "monitor PID: $$"
        sleep 10  # 每10秒检查一次
    done

    echo "No more Python process found for user $USER. Exiting."
}


#防止重复运行
monitor 




# 打印当前脚本的进程ID
# echo "Current script PID: $$" >> $log_file 2>&1 &

# 运行Python程序，并将输出重定向到日志文件
# nohup /home/liugangqiang/miniconda3/envs/pygenn/bin/python /home/liugangqiang/mam-allen-genn20240404/run_example_downscaled-analysis.py >>$log_file 2>&1 &
# nohup /home/liugangqiang/miniconda3/envs/genn-5/bin/python /home/liugangqiang/mam-allen-genn20240404/run_example_downscaled-analysis-V1.py >>$log_file 2>&1 &
# nohup /home/liugangqiang/miniconda3/envs/genn-5/bin/python /home/liugangqiang/mam-allen-genn20240404/run_example_downscaled-analysis-V1-input-output.py >>$log_file 2>&1 &
# nohup /home/liugangqiang/miniconda3/envs/genn-5/bin/python /home/liugangqiang/mam-allen-genn20240404/plot.py >$log_file 2>&1 &
# nohup /home/liugangqiang/miniconda3/envs/genn-5/bin/python /home/liugangqiang/mam-allen-genn20240404/run_example_downscaled-analysis-V1-current.py >>$log_file 2>&1 &
# nohup /home/liugangqiang/miniconda3/envs/genn-5/bin/python /home/liugangqiang/mam-allen-genn20240404/run_example_downscaled-analysis-V1-change-weight.py >>$log_file 2>&1 &
# nohup /home/liugangqiang/miniconda3/envs/genn-5/bin/python /home/liugangqiang/mam-allen-genn20240404/run_example_downscaled-analysis-V1-change-weight-byhand.py >>$log_file 2>&1 & echo "python脚本进程号:$!" >> $log_file
# nohup /home/liugangqiang/miniconda3/envs/genn-5/bin/python $1 >>$log_file 2>&1 &
nohup /home/liugangqiang/miniconda3/envs/genn-5/bin/python $1 >>$log_file 2>&1 & echo "python脚本进程号:$!" >> $log_file

# 检测程序是否在运行
monitor >> $log_file 2>&1 &
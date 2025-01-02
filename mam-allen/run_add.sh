#!/bin/bash

# 定义要检查的用户名
USER="liugangqiang"

# 定义一个函数来检查Python进程是否存在
check_python_process() {
    ps -u $USER | grep -q python
}

# 如果Python进程存在，则脚本持续运行，直到所有Python进程结束
monitor() {
    while check_python_process; do
        echo "Python process is running..."
        sleep 10  # 每10秒检查一次
    done

    echo "No more Python process found for user $USER. Exiting."
}

# 在后台运行监控函数，并将输出重定向到日志文件
monitor > monitor_script.log 2>&1 &

# 打印当前脚本的进程ID
echo "Current script PID: $$"

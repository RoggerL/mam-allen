#!/bin/bash

# 获取python路径
# echo "python路径为: $1"

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
while check_python_process; do
    echo "已经有python程序正在运行"
    echo "请选择一个选项:"
    echo "1. 等待前面程序完成"
    echo "2. 直接运行"
    echo "3. 退出程序"
    read -p "请输入选项 (1/2/3): " choice

    case $choice in
        1)
            echo "等待前面的程序完成..."
            sleep 10  # 每10秒检查一次
            ;;
        2)
            echo "直接运行新程序"
            break
            ;;
        3)
            echo "退出程序"
            exit 0
            ;;
        *)
            echo "无效的选项，请重新输入"
            ;;
    esac
done

echo "程序开始运行"

nohup /home/liugangqiang/miniconda3/envs/genn-4/bin/python /home/liugangqiang/mam-allen-20241222/mam-allen/run_multiarea.py >>$log_file 2>&1 & echo "python脚本进程号:$!" >> $log_file

# 检测程序是否在运行
monitor >> $log_file 2>&1 & 
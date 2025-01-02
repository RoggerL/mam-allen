import os
import subprocess
import time

# 定义要检查的用户名
USER = "liugangqiang"

def check_python_process():
    try:
        # 执行 ps 命令并检查是否有 Python 进程
        result = subprocess.run(['ps', '-u', USER], stdout=subprocess.PIPE)
        return 'python' in result.stdout.decode('utf-8')
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# 如果Python进程存在，则脚本持续运行，直到所有Python进程结束
while check_python_process():
    print("Python process is running...")
    time.sleep(10)  # 每10秒检查一次

print(f"No more Python process found for user {USER}. Exiting.")

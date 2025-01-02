import requests
import tarfile
import os

def upload_file():
        #文件路径信息
    dir_path = 'mam-allen-genn/simulations/9f7371ae3b82db95d2f9f89ffddfc316'
    file_path = dir_path+'.tar'
    upload_path = 'home'
    file_name = '9f7371ae3b82db95d2f9f89ffddfc31.tar'
    # DSM的基本信息
    base_url = 'http://210.31.77.116:5000'  # DSM的地址和端口
    account = 'liugangqiang'  # DSM的账户名
    password = 'lgq199610'  # DSM的密码

    # 1.登录并获取SID
    auth_params = {
        'api': 'SYNO.API.Auth',
        'version': '3',  # 使用最新版本
        'method': 'login',
        'account': account,
        'passwd': password,
        'session': 'FileStation',
        'format': 'sid',  # 使用sid格式获取会话ID
    }
    auth_response = requests.get(f'{base_url}/webapi/auth.cgi', params=auth_params)
    auth_response_json = auth_response.json()
    print(auth_response_json)

    # 检查是否成功获取sid
    if 'sid' in auth_response_json['data']:
        sid = auth_response_json['data']['sid']
    else:
        # 如果响应中没有sid，则打印错误信息并退出
        print("登录失败，响应内容：", auth_response_json)
        exit()


    #  打包文件夹
    with tarfile.open(file_path, "w") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))

    # 2.上传文件
    files = {
        'api': (None, 'SYNO.FileStation.Upload'),
        'version': (None, '2'),
        'method': (None, 'upload'),
        'path': (None, upload_path),
        'create_parents': (None, 'true'),
        'overwrite': (None, 'true'),  # 或者 'skip' 根据需要
        'file': (file_name, open(file_path, 'rb'), 'application/octet-stream'),
    }
    
        # 执行文件上传
    upload_url = f'{base_url}/webapi/entry.cgi?_sid={sid}'
    response = requests.post(upload_url, files=files)
    
    # 检查响应
    return response.json()

print(upload_file())
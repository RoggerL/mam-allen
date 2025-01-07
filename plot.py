import json
import os
from config import base_path
from config import data_path
import matplotlib.pyplot as plt
import numpy as np

current_time = "20240718_113428"
# with open(os.path.join(data_path,f"rate_list_{current_time}.json"), 'w') as file:
#     json.dump(rate_list, file)  # Saves the dictionary as a JSON file
#     print("File saved as:", f"rate_list_{current_time}.json")

factor = np.arange(0,2,1)
with open(os.path.join(data_path,f"rate_list_{current_time}.json"), 'r') as file:
    rate_list = json.load(file)
    
layer_types = ["23","4","5","6"]
neuron_types = ["E","S","P","H"]
for layer_type in layer_types:
    print("make figure")
    plt.figure()
    for j,neuron_type in enumerate(neuron_types):
        print("plot figure {}".format(j))
        plt.subplot(2, 2, j+1)  # 创建2x2的子图
        pop = neuron_type+layer_type
        plt.plot(factor, np.array(rate_list[pop]),label = f"rate_{pop}")
        plt.legend()
        # plt.title('Relationship between input and rate')
        # plt.xlabel('Factor')
        # plt.ylabel('rate')
    
    plt.figtext(0.5, 0.04, 'Factor', ha='center')
    plt.figtext(0.04, 0.5, 'Rate', va='center', rotation='vertical')  
    plt.savefig(os.path.join(data_path, f"input_factor-rate_{layer_type}.png"))
    plt.close()  # 关闭图像，避免重叠绘图    


import dataset
from config import base_path,data_path
import matplotlib.pyplot as plt

def get_population_list():
    '''
    Create a list of neuron population names, 
    where each name represents a combination of a neuron type and a layer type.
    '''
    
    pop_list = []
    neuron_types = ["E","S","P","V"]
    layer_types = ["1","23","4","5","6"]
    for layer_type in layer_types:
        if layer_type == "1":
            pop_list.append("H1")
        else:
            for neuron_type in neuron_types:
                pop_list.append(neuron_type+layer_type)
    return pop_list

def get_population_list_TH():
    '''
    Create a list of neuron population names of TH, 
    where each name represents a combination of a neuron type and a layer type.
    '''    
    
    pop_list = []
    neuron_types = ["E","S","P","V"]
    layer_types = ["1","23","5","6"]
    for layer_type in layer_types:
        if layer_type == "1":
            pop_list.append("H1")
        else:
            for neuron_type in neuron_types:
                pop_list.append(neuron_type+layer_type)
    return pop_list

pop_list_norm = get_population_list()
pop_list_TH = get_population_list_TH()

# colors = {
#     "1":"#FFF49C",
#     "23":"#FFA367",
#     "4":"#FF5A5F",
#     "5":"#3BC14A",
#     "6":"#0047AB"
#           }

colors = {'E': '#5555df',
            'P': '#048006',
            'V': '#a6a123',
            'S': '#c82528',
            # 'H': '#94b54f',
            'H': '#606a2b',
            }

shapes = {
    "1":'|',
    "23":'^',
    "4":'d',
    "5":"p",
    "6":'*'
          }

# 连接数据库并提取数据
param_label = 'conprecent'
with dataset.connect(f'sqlite:///{data_path}/dataset.db') as db:
    # for input_pop in pop_list_norm:
    #     print("input_pop=",input_pop)
    for param_name in ['delay_e','delay_i']:

        for pop_name in pop_list_norm:
            neu = pop_name[0]
            layer = pop_name[1:]
                        
            value_k_list = []
            rate_k_list = []

            # 遍历 k 值
            for k in range(10):
                if len(list(db[param_name].find(param_value=k))) > 0:
                    # value_k_list.append(k)
                    value_k = db[param_name].find(param_value=k)
                    for row in value_k:
                        print("row=",row)
                        # if row[pop_name] < 100:
                        if True:
                            value_k_list.append(k*0.1)
                            rate_k_list.append(row[pop_name])

            # print("rate_k=",rate_k_list)
        
            plt.plot(value_k_list, rate_k_list,color=colors[neu], linestyle='-', linewidth=1,marker=shapes[layer],markerfacecolor='none',label=f'{pop_name}')
            # 可视化图像
            # plt.figure(figsize=(10, 6))
        # plt.show()
        # plt.title("Variation of discharge rate of neural populations in V1 with {}'s extra current input".format(input_pop))
        plt.title("Variation of discharge rate of neural populations \n in V1 with {}".format(param_name))
        plt.xlabel("change of {}".format(param_name))
        plt.ylabel("Discharge rate of neural populations in V1: r/Hz")
        plt.legend(loc='upper left',fontsize='small',bbox_to_anchor=(0.99, 1))
        plt.savefig("/home/liugangqiang/fig3/rate with {}".format(param_name))
        plt.close()
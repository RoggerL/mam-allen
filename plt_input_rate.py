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
with dataset.connect(f'sqlite:///{data_path}/dataset.db') as db:
    for input_pop in pop_list_norm:
        print("input_pop=",input_pop)
        
        for param_name in pop_list_norm:
            neu = param_name[0]
            layer = param_name[1:]
                        
            value_k_list = []
            rate_k_list = []

            # 遍历 k 值
            for k in range(10):
                if len(list(db[input_pop].find(input_value=k))) > 0:
                    # value_k_list.append(k)
                    value_k = db[input_pop].find(input_value=k)
                    for row in value_k:
                        if row[param_name] < 100:
                        # if True:
                            value_k_list.append(k*10)
                            rate_k_list.append(row[param_name])

            # print("rate_k=",rate_k_list)
        
            plt.plot(value_k_list, rate_k_list,color=colors[neu], linestyle='-', linewidth=1,marker=shapes[layer],markerfacecolor='none',label=f'{param_name}')
            # 可视化图像
            # plt.figure(figsize=(10, 6))
        # plt.show()
        # plt.title("Variation of discharge rate of neural populations in V1 with {}'s extra current input".format(input_pop))
        plt.title("Variation of discharge rate of neural populations \n in V1 with {}'s extra current input".format(input_pop))
        plt.xlabel("{}'s extra current input: $\Delta$I/nA".format(input_pop))
        plt.ylabel("discharge rate of neural populations in V1: r/Hz")
        plt.legend(loc='upper left',fontsize='small',bbox_to_anchor=(0.99, 1))
        plt.savefig('/home/liugangqiang/fig/variation of rate with {}'.format(input_pop))
        plt.close()
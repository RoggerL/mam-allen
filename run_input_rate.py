import os
import time 
import random
import pygenn
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool, Manager
from config import base_path,data_path
from multiarea_model import MultiAreaModel
from scipy.signal import correlate
import dataset
import re

#get complete area list 
complete_area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                      'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                      'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                      'STPa', '46', 'AITd', 'TH']

# Print the current directory to prevent incorrect operation.
current_directory = os.path.dirname(os.path.abspath(__file__))
print("abstract path of current directory is:", current_directory)

#set single neuron parameters
# population_list = get_population_list()

# single_neuron_dict = {
#     "E":{
#     # Leak potential of the neurons .
#     'E_L': -70.0, # mV
#     # Threshold potential of the neurons .
#     'V_th': -50.0, # mV
#     # Membrane potential after a spike .
#     'V_reset': -60.0, # mV
#     # Membrane capacitance .
#     'C_m': 500.0, # pF
#     # Membrane time constant .
#     'tau_m': 20.0, # ms
#     # Time constant of postsynaptic currents .
#     'tau_syn': 0.5, # ms
#     # Refractory period of the neurons after a spike .
#     't_ref': 2.0 # ms
#     },
#     "S":{
#     # Leak potential of the neurons .
#     'E_L': -76.0, # mV
#     # Threshold potential of the neurons .
#     'V_th': -50.0, # mV
#     # Membrane potential after a spike .
#     'V_reset': -60.0, # mV
#     # Membrane capacitance .
#     'C_m': 800.0, # pF
#     # Membrane time constant .
#     'tau_m': 50.0, # ms
#     # Time constant of postsynaptic currents .
#     'tau_syn': 0.5, # ms
#     # Refractory period of the neurons after a spike .
#     't_ref': 1.0 # ms
#     },
#     "P":{
#     # Leak potential of the neurons .
#     'E_L': -86.0, # mV
#     # Threshold potential of the neurons .
#     'V_th': -50.0, # mV
#     # Membrane potential after a spike .
#     'V_reset': -60.0, # mV
#     # Membrane capacitance .
#     'C_m': 200.0, # pF
#     # Membrane time constant .
#     'tau_m': 10.0, # ms
#     # Time constant of postsynaptic currents .
#     'tau_syn': 0.5, # ms
#     # Refractory period of the neurons after a spike .
#     't_ref': 1.0 # ms
#     },
#     'V':{
#     # Leak potential of the neurons .
#     'E_L': -70.0, # mV
#     # Threshold potential of the neurons .
#     'V_th': -50.0, # mV
#     # Membrane potential after a spike .
#     'V_reset': -65.0, # mV
#     # Membrane capacitance .
#     'C_m': 100.0, # pF
#     # Membrane time constant .
#     'tau_m': 20.0, # ms
#     # Time constant of postsynaptic currents .
#     'tau_syn': 0.5, # ms
#     # Refractory period of the neurons after a spike .
#     't_ref': 1.0 # ms
#     },
#     'H':{
#     # Leak potential of the neurons .
#     'E_L': -70.0, # mV
#     # Threshold potential of the neurons .
#     'V_th': -50.0, # mV
#     # Membrane potential after a spike .
#     'V_reset': -65.0, # mV
#     # Membrane capacitance .
#     'C_m': 100.0, # pF
#     # Membrane time constant .
#     'tau_m': 20.0, # ms
#     # Time constant of postsynaptic currents .
#     'tau_syn': 0.5, # ms
#     # Refractory period of the neurons after a spike .
#     't_ref': 1.0 # ms
#     }
#     } 

single_neuron_dict = {
    "E":{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 500.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0 # ms
    },
    "V":{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 500.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0 # ms
    },    
    "P":{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 500.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0 # ms
    },       
    "H":{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 500.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0 # ms
    },      
    "S":{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 500.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 2.0 # ms
    },      
}

def extract_area_dict(d, structure, target_area, source_area):
    """
    Extract the dictionary containing only information
    specific to a given pair of areas from a nested dictionary
    describing the entire network.

    Parameters
    ----------
    d : dict
        Dictionary to be converted.
    structure : dict
        Structure of the network. Define the populations for each single area.
    target_area : str
        Target area of the projection
    source_area : str
        Source area of the projection
    """
    area_dict = {}
    for pop in structure[target_area]:
        area_dict[pop] = {}
        for pop2 in structure[source_area]:
            area_dict[pop][pop2] = d[target_area][pop][source_area][pop2]
    return area_dict

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
  
def plt_matrix(matrix,area,path,type="currents",label_max=True,show_filename=False):
    matrix_df = pd.DataFrame(matrix)
    matrix_df = matrix_df.round(3)    
    plt.figure(figsize=(8, 4))
    plt.axis('off')
    table = plt.table(cellText=matrix_df.values, colLabels=matrix_df.columns, rowLabels=matrix_df.index, cellLoc='center', loc='center')
    # Adjust the font size and border of the table
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    
    # Find the maximum and minimum values of each column.
    max_values = matrix_df.max()
    min_values = matrix_df.min()
    
    for (i, j), cell in table.get_celld().items():
        cell.set_linewidth(0.5)  # Thinned border
        # Mark the maximum value in yellow and the minimum value in red.
        if label_max == True:
            if i > 0 and j >= 0:
                if matrix_df.iloc[i - 1, j] == max_values[j]:
                    cell.set_facecolor('yellow')  
            if i > 0 and j >= 0:
                if matrix_df.iloc[i - 1, j] == min_values[j]:
                    cell.set_facecolor('red')  

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(f'{path}/matrix_{type}'):
        os.makedirs(f'{path}/matrix_{type}')
    filename = f'{path}/matrix_{type}/{area}_{current_time}.png'  
    if show_filename:
        print("filename=",filename)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def compute_lag(X, Y, dt, plt_show=False, min_lag=None, max_lag=None):
    # 计算互相关
    correlation = correlate(X, Y, mode='full')
    lags = np.arange(-len(X) + 1, len(X)) * dt  # lag 轴（转换为时间）

    # 如果指定了 min_lag 和 max_lag，则裁剪 lags 和 correlation
    if min_lag is not None and max_lag is not None:
        valid_indices = (lags >= min_lag) & (lags <= max_lag)
        lags = lags[valid_indices]
        correlation = correlation[valid_indices]

    # 找到最大互相关值对应的 lag
    max_corr_index = np.argmax(correlation)
    time_lag = lags[max_corr_index]
    max_corr_value = correlation[max_corr_index]

    # 计算最高匹配的皮尔逊相关值
    shifted_Y = np.roll(Y, int(time_lag / dt))  # 根据 time_lag 对 Y 进行位移
    pearson_corr = np.corrcoef(X, shifted_Y)[0, 1]  # 计算皮尔逊相关系数



    if plt_show:
        # 输出结果
        print(f"Estimated Time-Lag: {time_lag:.4f} seconds")
        print(f"Maximum Cross-Correlation Value: {max_corr_value:.4f}")
        print(f"Pearson Correlation at Maximum Lag: {pearson_corr:.4f}")        
        
        # 绘制互相关结果
        plt.figure(figsize=(10, 4))
        plt.plot(lags, correlation, label="Cross-Correlation")
        plt.axvline(x=time_lag, color='r', linestyle='--', label=f"Time-Lag: {time_lag:.4f}s")
        plt.title("Cross-Correlation between X and Y")
        plt.xlabel("Lag (s)")
        plt.ylabel("Correlation")
        plt.legend()
        plt.grid()
        plt.show()
    
    return time_lag, pearson_corr
#Set input current for each pop in each area

# input_current = {
#                 "H1" : 551,
#                 "V23" : 501.+20.,
#                 "S23" : 501-50.,
#                 "E23" : 501.+20.,
#                 "P23" : 501.,
#                 "V4" : 501-10.,
#                 "S4" : 501-10.,
#                 "E4" : 501+100.*1.,
#                 "P4" : 501+10.*0,
#                 "V5" : 501-10.,
#                 "S5" : 501.-20.,
#                 "E5" : 501+50.,
#                 "P5" : 501.-20.,
#                 "V6" : 501+10.*0,
#                 "S6" : 501+10.*0,
#                 "E6" : 501+50.,
#                 "P6" : 501+10.*0,              
#             }

# # V1
# input_current = {
#                 "H1" : 501,
#                 "V23" : 501.,
#                 "S23" : 501,
#                 "E23" : 501.+ 50.,
#                 "P23" : 501.,
#                 "V4" : 501 + 10.,
#                 "S4" : 501,
#                 "E4" : 501.+ 50.,
#                 "P4" : 501 + 10.,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.+ 10.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

#V2
# input_current = {
#                 "H1" : 501,
#                 "V23" : 501.,
#                 "S23" : 501,
#                 "E23" : 501.+ 50.,
#                 "P23" : 501.,
#                 "V4" : 501 + 10.,
#                 "S4" : 501,
#                 "E4" : 501.+ 50.,
#                 "P4" : 501 + 10.,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.+ 10.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

#V3
# input_current = {
#                 "H1" : 501,
#                 "V23" : 501.,
#                 "S23" : 501,
#                 "E23" : 501.+ 60.,
#                 "P23" : 501.,
#                 "V4" : 501 + 10.,
#                 "S4" : 501,
#                 "E4" : 501.+ 60.,
#                 "P4" : 501 + 10.,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.+ 10.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

# V3A
# input_current = {
#                 "H1" : 501.,
#                 "V23" : 501.,
#                 "S23" : 501,
#                 "E23" : 501.+ 90.,
#                 "P23" : 501.,
#                 "V4" : 501 + 10.,
#                 "S4" : 501,
#                 "E4" : 501.+ 90.,
#                 "P4" : 501 + 10.,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.+ 20.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.+10,
#                 "P6" : 501.,              
#             }

#MT
# input_current = {
#                 "H1" : 501.,
#                 "V23" : 501.,
#                 "S23" : 501,
#                 "E23" : 501.+ 90.,
#                 "P23" : 501.,
#                 "V4" : 501 + 10.,
#                 "S4" : 501,
#                 "E4" : 501.+ 90.,
#                 "P4" : 501 + 10.,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.+ 20.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

#V4t
# input_current = {
#                 "H1" : 501.,
#                 "V23" : 501.,
#                 "S23" : 501,
#                 "E23" : 501.+ 90.,
#                 "P23" : 501.,
#                 "V4" : 501 + 10.,
#                 "S4" : 501,
#                 "E4" : 501.+ 90.,
#                 "P4" : 501 + 10.,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.+ 20.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

#V4
# input_current = {
#                 "H1" : 501,
#                 "V23" : 501.,
#                 "S23" : 501,
#                 "E23" : 501.+ 80.,
#                 "P23" : 501.,
#                 "V4" : 501 + 10.,
#                 "S4" : 501,
#                 "E4" : 501.+ 80.,
#                 "P4" : 501 + 10.,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.+ 20.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.+10,
#                 "P6" : 501.,              
#             }

#VOT
# input_current = {
#                 "H1" : 501.,
#                 "V23" : 501.,
#                 "S23" : 501,
#                 "E23" : 501.+ 90.,
#                 "P23" : 501.,
#                 "V4" : 501 + 10.,
#                 "S4" : 501,
#                 "E4" : 501.+ 90.,
#                 "P4" : 501 + 10.,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.+ 20.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

# input_current = {
#                 "H1" : 501.,
#                 "V23": 501.,
#                 "S23": 501,
#                 "E23": 501.+80.,
#                 "P23": 501.,
#                 "V4" : 501.+10.,
#                 "S4" : 501,
#                 "E4" : 501.+80.,
#                 "P4" : 501.+10.,
#                 "V5" : 501.,
#                 "S5" : 501.+10.,
#                 "E5" : 501.+20.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

# input_current = {
#                 "H1" : 501.,
#                 "V23": 501.,
#                 "S23": 501.-20.,
#                 "E23": 501.,
#                 "P23": 501.+10.,
#                 "V4" : 501.,
#                 "S4" : 501.,
#                 "E4" : 501.,
#                 "P4" : 501.,
#                 "V5" : 501.-20.,
#                 "S5" : 501.,
#                 "E5" : 501.,
#                 "P5" : 501.-20.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }


# input_current = {
#                 "H1" : 501,
#                 "V23" : 501.,
#                 "S23" : 501. + 20.,
#                 "E23" : 501. + 20.,
#                 "P23" : 501. + 10.,
#                 "V4" : 501,
#                 "S4" : 501,
#                 "E4" : 501 + 10.,
#                 "P4" : 501,
#                 "V5" : 501,
#                 "S5" : 501.,
#                 "E5" : 501.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

#try 20250210
# input_current = {
#                 "H1" : 501.,
#                 "V23": 501.,
#                 "S23": 501.,
#                 "E23": 501.+80.,
#                 "P23": 501.,
#                 "V4" : 501.+10.,
#                 "S4" : 501,
#                 "E4" : 501.+80.,
#                 "P4" : 501.+10.,
#                 "V5" : 501.,
#                 "S5" : 501.+10.,
#                 "E5" : 501.+40.,
#                 "P5" : 501.,
#                 "V6" : 501.,
#                 "S6" : 501.,
#                 "E6" : 501.,
#                 "P6" : 501.,              
#             }

# input_update = deepcopy(input_current)

def simulation_block(block_id, device_id,param_name = 'input_value'):
    '''
    single simulation block。simulate neural network model in a single device.

    Parameter:
    i: index of block
    device: index of device
    '''
    
    print("param_name=",param_name)
    
    with dataset.connect(f'sqlite:///{data_path}/dataset.db') as db:  
        if len(list(db[param_name].find(input_value=block_id)))>0:
            print("{} of {} has been simulated".format(param_name,block_id))
            return None  
    
    #Set device for simulation
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    #Evaluation finish time
    start_time = time.time()
    print(time.strftime("start_time:%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    pred_time = start_time + 10252.23166012764
    print(time.strftime("pred_time:%Y-%m-%d %H:%M:%S", time.localtime(pred_time)))


    #Set alpha value
    alpha_dict_norm= {'H1': 0.1, 'E23': 1., 'S23': 0.6, 'V23': 1., 'P23': 1.9, 'E4': 0.8, 'S4': 0.8, 'V4': 1., 'P4': 1., 'E5': 1.6, 'S5': 0.5, 'V5': 3., 'P5': 1.1, 'E6': 0.7, 'S6': 1., 'V6': 2., 'P6': 1.}
    alpha_dict_TH= {'H1': 0.1, 'E23': 0.5, 'S23': 0.6, 'V23': 0.5, 'P23': 1., 'E4': 1., 'S4': 0.8, 'V4': 1., 'P4': 1., 'E5': 1.6, 'S5': 0.5 , 'V5': 1., 'P5': 1., 'E6': 0.5, 'S6': 1., 'V6': 1., 'P6': 1.}
    beta_dict_norm = {"H1" : 3.9+0.01,
                 "E23" : 0.71+0.01, "P23" : 0.48+0.01, "S23" : 1.+0.01, "V23" :0.9+0.01,
                 "E4" : 1.66+0.01, "S4" : 0.24+0.01, "V4" : 0.46+0.01, "P4" : 0.8+0.01,
                 "E5" : 0.95+0.01, "S5" : 0.48+0.01, "V5" : 1.2+0.01, "P5" :1.09+0.01,
                 "E6" : 1.12+0.01, "S6" : 0.63+0.01, "V6" : 0.5+0.01, "P6" : 0.42+0.01}
    if 'connection' in param_name:
        pattern = r'^(.+?)connection$'
        match = re.match(pattern, param_name)
        alpha_dict_norm[match.group(1)] += 0.1*block_id

    if 'conprecent' in param_name:
        pattern = r'^(.+?)conprecent$'
        match = re.match(pattern, param_name)
        alpha_dict_norm[match.group(1)] *= (1+0.1*block_id)
        
    if 'conmultiple' in param_name:
        pattern = r'^(.+?)conmultiple$'
        match = re.match(pattern, param_name)
        alpha_dict_norm[match.group(1)] *= (1+block_id)
        
    
    conn_params = {'replace_non_simulated_areas': 'hom_poisson_stat',
                   'g': -4.,
                   'g_H' : -4.,
                   'g_V' : -4.,
                   'g_S' : -4.,
                   'g_P' : -4.,
                   'alpha_norm': alpha_dict_norm,
                   'alpha_TH': alpha_dict_TH,         
                   'beta_norm': beta_dict_norm,        
                   
                #    'K_stable': os.path.join(base_path, "K_stable.npy"),
                    # 'K_stable':None,           
                   'PSP_e_23_4': 0.4,
                   'PSP_e_5_h1': 0.15,
                   'PSP_e': 0.15,
                   'av_indegree_V1': 3950.}
    
    # V1
    input_current = {
                    "H1" : 501,
                    "V23" : 501.,
                    "S23" : 501,
                    "E23" : 501.+ 50.,
                    "P23" : 501.,
                    "V4" : 501 + 10.,
                    "S4" : 501,
                    "E4" : 501.+ 50.,
                    "P4" : 501 + 10.,
                    "V5" : 501,
                    "S5" : 501.,
                    "E5" : 501.+ 10.,
                    "P5" : 501.,
                    "V6" : 501.,
                    "S6" : 501.,
                    "E6" : 501.,
                    "P6" : 501.,              
                }
    
    if param_name in input_current:
        print("param_name=",param_name)
        input_current[param_name] += block_id*10.
    
    input_params = {'rate_ext': 5800, 
                    'input_factor_E' : 0.5,
                    "input": input_current,
                    'poisson_input': False,}
    neuron_params = {'V0_mean': -150.,
                     'V0_sd': 50.,
                     'single_neuron_dict': single_neuron_dict
                     }
    delay_params = {
    # Local dendritic delay for excitatory transmission 
    'delay_e': 1.5, #ms
    # Local dendritic delay for inhibitory transmission 
    'delay_i': 0.75, #ms
    # Relative standard deviation for both local and inter-area delays
    'delay_rel': 0.5, # ms
    # Axonal transmission speed to compute interareal delays 
    'interarea_speed': 3.5 #mm/ms
    }
    
    if param_name == 'delay_e':
        delay_params['delay_e'] += 0.1*block_id
    
    if param_name == 'delay_i':
        delay_params['delay_i'] += 0.1*block_id
    
    network_params = {'N_scaling': 1.,
                      'K_scaling': 1.,
                      'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates2.json'),
                      'input_params': input_params,
                      'connection_params': conn_params,
                      'neuron_params': neuron_params,
                      'delay_params': delay_params}

    sim_params = {'t_sim': 1000.,
                  'master_seed': 15,
                  'num_processes': 1,
                  'local_num_threads': 1,
                  'recording_dict': {'record_vm': False},
                #   'areas_simulated': [complete_area_list[i%32]],
                  'areas_simulated': ['V1'],
                  "cut_connect" : False}

    theory_params = {'dt': 0.1}

    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=False,
                       theory_spec=theory_params,
                       analysis= True,
                       code_path = 'path_{}'.format(device_id))
    # p, r = M.theory.integrate_siegert()
    # print("Mean-field theory predicts an average "
    #       "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
    
    rates = {
            "H1" : 10.,
            "V23" : 10.,
            "S23" : 10.,
            "E23" : 10.,
            "P23" : 10.,
            "V4" : 10.,
            "S4" : 10.,
            "E4" : 10.,
            "P4" : 10.,
            "V5" : 10.,
            "S5" : 10.,
            "E5" : 10.,
            "P5" : 10.,
            "V6" : 10.,
            "S6" : 10.,
            "E6" : 10.,
            "P6" : 10.,        
        }    
    
    M.create_current()
    M.plt_matrix_weight('V1')
    
    if True:
        print("start simulator")
        M.simulation.simulate()
        
        M.analysis.load_data()
        M.analysis.create_pop_rates(t_min=500.)
        M.analysis.create_pop_rate_dists()
        M.analysis.create_rate_time_series()
        M.analysis.create_synaptic_input()
        
        for area in M.analysis.areas_loaded:
            if area != 'TH':
                pop_list = pop_list_norm
            else :
                pop_list = pop_list_TH
            
            lag_and_values = [[compute_lag(M.analysis.rate_time_series_pops[area][pop_1][500:],M.analysis.rate_time_series_pops[area][pop_2][500:],dt=0.2, min_lag=-10., max_lag=10.) for pop_1 in ['E23','E4','E5','E6']] for pop_2 in ['E23','E4','E5','E6']]
            lags = [[lag_and_value[0] for lag_and_value in line] for line in lag_and_values]
            values = [[lag_and_value[1] for lag_and_value in line] for line in lag_and_values]
            print("lags=",lags)
            print("values=",values)
                        
            if True:
                M.analysis.multi_rate_display(area,pops = pop_list,output = "png")
                if pygenn.__version__ == "5.0.0":
                    M.analysis.multi_voltage_display(area,pops = pop_list,output = "png")
                    M.analysis.multi_current_display(area,pops = pop_list,output = "png")
                    M.analysis.avg_current_display(area,pops = pop_list,t_min=500,output = "png")
                M.analysis.multi_input_display(area=area,pops = pop_list,t_min=500,output = "png")
                M.analysis.multi_power_display(area=area,pops = pop_list,t_min=500,output = "png",resolution=0.2)
                current_dict = M.analysis.theory_current_display(area,pops = pop_list,t_min=500,output = "png")
                    # M.analysis.synaptic_current_display(area,pops = pop_list_norm,t_min=1000,output = "png")
                # print("current_dict",current_dict)
            # else:
                # M.analysis.multi_rate_display(area,pops = pop_list_TH,output = "png")
        
        
        a = dict(label=M.simulation.label,param_value=block_id)

        for area in M.analysis.areas_loaded:
            # if area == 'V1':
            if area != "TH":
                pop_list = pop_list_norm
            else:
                pop_list = pop_list_TH
            
            for pop in pop_list:
                if True:
                    M.analysis.single_rate_display(area=area,pop=pop,output = "png")
                    # frac_neurons : float, [0,1]
                    frac_neurons = 0.01
                    M.analysis.single_dot_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                    if pygenn.__version__ == "5.0.0":
                        M.analysis.single_voltage_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                        M.analysis.single_current_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                    M.analysis.single_input_display(area=area,pop=pop,t_min=500,frac_neurons=frac_neurons,output = "png")
                    M.analysis.single_power_display(area=area,pop=pop,t_min=500,output = "png",resolution=0.2)
        
            K_in = extract_area_dict(M.K, M.structure, area,area)
            W_in = extract_area_dict(M.W, M.structure, area,area)
            if True:
                rates = {pop+"_"+area :M.analysis.pop_rates[area][pop][0] for pop in pop_list}
                print("rates=",rates)
                # a = dict(label=M.label,input_value=block_id)
                a.update(rates)
                    
                # M.create_current(rates=rates)
                # M.plt_matrix_weight(area)

        with dataset.connect(f'sqlite:///{data_path}/dataset.db') as db:
            db[param_name].insert(a)
            print("insert {} {}".format(param_name,block_id))

    # # M.analysis.save()
    # # M.analysis.show_rates(output = "png")

    # # for area in M.analysis.pop_rates['Parameters']['areas']:
    # #     for pop in M.analysis.pop_rates[area]:
    # #         if pop != 'total':
    # #             # print(M.analysis.pop_rates[area][pop][0])
    # #             pass

    # # structure_similarity_dict = dict()
    # # activivity_similarity_dict = dict()

    # # print(type(M.analysis.rate_time_series_pops['V1']['H1']))

    # # for area1 in M.K:
    # #     structure_similarity_dict[area1] = dict()
    # #     activivity_similarity_dict[area1] = dict()
    # #     for area2 in M.K:
    # #         # print("area1=",area1)
    # #         # print("area2=",area2)
    # #         rate_sumsquared_1 = 0
    # #         rate_sumsquared_2 = 0
    # #         rate_sumconvolution = 0

    # #         indegree_sumsquared_1 = 0
    # #         indegree_sumsquared_2 = 0
    # #         indegree_sumconvolution = 0

    # #         if area1 == 'TH' or area2  == 'TH':
    # #             pop_list = pop_list_TH
    # #         else:
    # #             pop_list = pop_list_norm

    # #         for pop in pop_list:
    # #             # rate_sumsquared_1 = rate_sumsquared_1 + M.analysis.pop_rates[area1][pop][0]*M.analysis.pop_rates[area1][pop][0]
    # #             rate_sumsquared_1 = rate_sumsquared_1 + np.sum(M.analysis.rate_time_series_pops[area1][pop]*M.analysis.rate_time_series_pops[area1][pop])
    # #             # rate_sumsquared_2 = rate_sumsquared_2 + M.analysis.pop_rates[area2][pop][0]*M.analysis.pop_rates[area2][pop][0]
    # #             rate_sumsquared_2 = rate_sumsquared_2 + np.sum(M.analysis.rate_time_series_pops[area2][pop]*M.analysis.rate_time_series_pops[area2][pop])
    # #             # rate_sumconvolution = rate_sumconvolution + M.analysis.pop_rates[area1][pop][0]*M.analysis.pop_rates[area2][pop][0]
    # #             rate_sumconvolution = rate_sumconvolution + np.sum(M.analysis.rate_time_series_pops[area1][pop]*M.analysis.rate_time_series_pops[area2][pop])

    # #             # print(M.K[k][pop])
    # #             indegree_list_1 = []
    # #             indegree_list_2 = []
    # #             for a in M.K[area1][pop]:
    # #                 for b in M.K[area1][pop][a]:
    # #                     indegree_list_1.append(M.K[area1][pop][a][b])
    # #                     indegree_list_2.append(M.K[area2][pop][a][b])
    # #             # print("indegree_list=",np.array(indegree_list_1))        
    # #             # print("indegree_list=",np.sum(np.array(indegree_list_1)*np.array(indegree_list_1)))
    # #             indegree_sumsquared_1 = indegree_sumsquared_1 + np.sum(np.array(indegree_list_1)*np.array(indegree_list_1))
    # #             # print("indegree_list=",np.sum(np.array(indegree_list_1)*np.array(indegree_list_2)))
    # #             indegree_sumconvolution = indegree_sumconvolution + np.sum(np.array(indegree_list_1)*np.array(indegree_list_2))
    # #             # print("indegree_list=",np.sum(np.array(indegree_list_2)*np.array(indegree_list_2)))
    # #             indegree_sumsquared_2 = indegree_sumsquared_2 + np.sum(np.array(indegree_list_2)*np.array(indegree_list_2))

    # #         activivity_similarity = rate_sumconvolution / (np.sqrt(rate_sumsquared_1) * np.sqrt(rate_sumsquared_2))
    # #         activivity_similarity_dict[area1][area2] = activivity_similarity
    # #         # print("activivity_similarity=",activivity_similarity)
    # #         structure_similarity = indegree_sumconvolution / (np.sqrt(indegree_sumsquared_1) * np.sqrt(indegree_sumsquared_2))
    # #         structure_similarity_dict[area1][area2] = structure_similarity  
    # #         # print("structure_similarity=",structure_similarity)



    # # # # 创建柱状图
    # # # plt.bar(list(structure_similarity.keys()), list(structure_similarity.values()))

    # # # # 添加标题和轴标签
    # # # plt.title('Fruit Quantity')
    # # # plt.xlabel('Fruit')
    # # # plt.ylabel('Quantity')

    # # # # 显示图表
    # # # plt.show()

    # # # Extracting keys and values for direct plotting
    # # # print(structure_similarity_dict['V1'])
    # # keys = structure_similarity_dict['V1'].keys()
    # # values1 = structure_similarity_dict['V1'].values()
    # # values2 = activivity_similarity_dict['V1'].values()

    # # # Creating figure and axis object
    # # fig, ax = plt.subplots()

    # # # Set positions for each bar
    # # ind = range(len(structure_similarity_dict['V1']))  # The x locations for the groups
    # # width = 0.5  # The width of the bars

    # # # Plotting both sets of data
    # # rects1 = ax.bar(ind, [v for v in values1], width, label='structure')
    # # rects2 = ax.bar([p + width for p in ind],[v for v in values2], width, label='activivity')

    # # # Add some text for labels, title, and custom x-axis tick labels, etc.
    # # ax.set_ylabel('similarity')
    # # ax.set_title('structure and activity similarity across different area')
    # # ax.set_xticks([p + width / 2 for p in ind])
    # # ax.set_xticklabels(keys)
    # # ax.legend()

    # # # Show the plot
    # # plt.savefig("test.png")

    # M.upload_file()

    # # end_time = time.time()
    # # print(time.strftime("end_time:%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    # # print("real_time=",end_time - start_time)
    
    del M

def run_simulation(args):
    task_id, device,param_name,available_devices = args
    print("param_name=",param_name)
    try:
        simulation_block(task_id, device, param_name)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 任务完成后释放设备
        available_devices.put(device)

def multi_program(device_list, num_tasks,param_name = 'input_value'):
    
    # 使用 Manager 列表管理设备的空闲状态
    # with Manager() as manager:
    #     available_devices = manager.list(device_list)  # 动态设备列表
    with Manager() as manager:
        available_devices = manager.Queue()  # 使用 Queue 代替 list

        # 将所有设备加入队列
        for device in device_list:
            available_devices.put(device)


        # 创建任务参数生成器
        def generate_task_args(param_name = 'input_value'):
            for task_id in range(num_tasks):
                while available_devices.empty():
                    time.sleep(0.1)  
                device = available_devices.get()  
                yield (task_id, device, param_name, available_devices)

        start_time = time.time()

        # 创建一个进程池
        with Pool(processes=len(device_list)) as pool:
            results = []

            for task_args in generate_task_args(param_name):
                result = pool.apply_async(run_simulation, args=(task_args,))
                results.append(result)

            # for result in results:
            #     result.get()
            
            timeout = 8*3600
            # 处理所有任务的结果
            for result, task_args in results:
                try:
                    result.get(timeout=timeout)  # 等待任务完成，最多等待 timeout 秒
                except multiprocessing.TimeoutError:
                    task_id, device, _, _ = task_args
                    print(f"Task {task_id} exceeded the time limit and is being terminated.")
                    # 手动释放设备
                    available_devices.put(device)
                except Exception as e:
                    task_id, device, _, _ = task_args
                    print(f"Task {task_id} raised an exception: {e}")
                    # 手动释放设备
                    available_devices.put(device)

        end_time = time.time()
        print(f"\nAll tasks completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # 测试配置
    device_list = [4]  
    num_tasks = 1

    # V1各群神经元放电率随每群神经元额外电流输入变化
    # for param_name in pop_list_norm:
    #     multi_program(device_list, num_tasks,param_name=param_name)
        
    # V1各群神经元放电率随每群神经元投射的连接强度的变化
    # for param_name in pop_list_norm:
    #     multi_program(device_list, num_tasks,param_name="{}connection".format(param_name))
    # simulation_block(block_id= 5,device_id=5,param_name = 'S5connection')

    # for param_name in pop_list_norm:
    #     multi_program(device_list, num_tasks,param_name="{}conprecent".format(param_name))

    # for param_name in pop_list_norm:
    #     multi_program(device_list, num_tasks,param_name="{}conmultiple".format(param_name))

    # for param_name in pop_list_norm:
    #     multi_program(device_list, num_tasks,param_name="{}multiarea".format(param_name))
    # multi_program(device_list, num_tasks,param_name='delay_e')
    # multi_program(device_list, num_tasks,param_name='delay_i')
    
    simulation_block(0, 5, param_name="test")
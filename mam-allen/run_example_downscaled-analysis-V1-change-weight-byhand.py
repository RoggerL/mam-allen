import numpy as np
import os
import time
from multiarea_model import MultiAreaModel
from config import base_path
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from datetime import datetime
import random

# population_list = get_population_list()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

# 将数值保留三位有效数字
def round_to_three(x):
    return round(x, 1)

def solve_homogeneous(A, tol=1e-1):
    """
    求解齐次线性方程组 Ax = 0 的非平凡解.
    
    参数:
    A: 系数矩阵 (m, n)
    tol: 容差，判断奇异值为0的阈值
    
    返回:
    零空间的基础解向量列表
    """
    
    # 奇异值分解
    U, S, Vt = np.linalg.svd(A)
    
    # 识别奇异值接近于零的位置
    null_mask = (S <= tol)
    
    # 提取零空间
    null_space = Vt.T[:, null_mask]
    
    return null_space

def solve_homogeneous_system(A, tol=1e-1):

    # 计算 A^T A
    ATA = np.dot(A.T, A)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(ATA)

    # 选择最小的特征值对应的特征向量
    min_eigenvalue_index = np.argmin(eigenvalues)
    x_opt = eigenvectors[:, min_eigenvalue_index]

    # 确保 x_opt 是单位向量
    x_opt = x_opt / np.linalg.norm(x_opt)

    print("最优的 x 向量是：", x_opt)
    print("Ax=",np.dot(A,x_opt))
    
    print("A=",A)
    
    # # 计算特征值和特征向量
    # eigenvalues, eigenvectors = np.linalg.eig(A)

    # # 打印所有特征值
    # print("原始特征值: ", eigenvalues)

    # # 找到最小特征值的索引，并将其设置为0
    # min_eigenvalue_index = np.argmin(eigenvalues)
    # eigenvalues[min_eigenvalue_index] = 0

    # # 打印修改后的特征值
    # print("修改后的特征值: ", eigenvalues)

    # # 使用修改后的特征值和特征向量重构矩阵
    # reconstructed_matrix = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)

    # # 打印重构后的矩阵
    # print("重构后的矩阵: ", reconstructed_matrix)
    
    # x_opt = solve_homogeneous(reconstructed_matrix)

    return x_opt

# def computer_syn_current(A,x,rate=10):

# g_dict ={
#             'H1':  1.,
#             'E23': 1.,
#             'S23': 1.,
#             'V23': 1.,
#             'P23': 1.,
#             'E4':  1.,
#             'S4':  1.,
#             'V4':  1.,
#             'P4':  1.,
#             'E5':  1.,
#             'S5':  1.,
#             'V5':  1.,
#             'P5':  1.,
#             'E6':  1.,
#             'S6':  1.,
#             'V6':  1.,
#             'P6':  1.,
# }



def computer_current(gd,showmatrix=False):
    # print("in_g=",g_dict)
    g_dict = deepcopy(gd)
    
    start_time = time.time()
    # 将新的时间戳转换为本地时间的struct_time对象
    # local_time = time.localtime(start_time)
    print(time.strftime("start_time:%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    pred_time = start_time + 10252.23166012764
    print(time.strftime("pred_time:%Y-%m-%d %H:%M:%S", time.localtime(pred_time)))

    """
    Down-scaled model.
    Neurons and indegrees are both scaled down to 10 %.
    Can usually be simulated on a local machine.

    Warning: This will not yield reasonable dynamical results from the
    network and is only meant to demonstrate the simulation workflow.
    """

    conn_params = {'replace_non_simulated_areas': 'hom_poisson_stat',
                   'g': -4.,
                   'g_H' : -4.,
                   'g_V' : -4.,
                   'g_S' : -4.,
                   'g_P' : -4.,
                    'g_dict': g_dict,
                #    'K_stable': os.path.join(base_path, "K_stable.npy"),
                    'K_stable':None,
                   'fac_nu_ext_TH': 1.2,
                   'fac_nu_ext_23E': 2.0,
                   'fac_nu_ext_4E': 2.0,
                   'fac_nu_ext_5E': 1.2,
                   'fac_nu_ext_6E': 1.,
                   'fac_nu_ext_23': 1.,
                   'fac_nu_ext_4': 1.0,
                   'fac_nu_ext_5': 1.0,
                   'fac_nu_ext_6': 1.,
                   'fac_nu_ext_1H':  1.0,
                   'fac_nu_ext_23V': 1.2,
                   'fac_nu_ext_4V': 3.,
                   'fac_nu_ext_5V': 3.0,
                   'fac_nu_ext_6V': 3.,  
                   'fac_nu_ext_23S': 0.5,
                   'fac_nu_ext_4S': 0.3,
                   'fac_nu_ext_5S': 0.3,
                   'fac_nu_ext_6S': 0.3,   
                   'fac_nu_ext_23P': 1.,
                   'fac_nu_ext_4P': 1.,
                   'fac_nu_ext_5P': 1.,
                   'fac_nu_ext_6P': 1.,              
                   'PSP_e_23_4': 0.30,
                   'PSP_e_5_h1': 0.15,
                   'PSP_e': 0.15,
                   'av_indegree_V1': 3950.}
    input_params = {'rate_ext': 40, 
                    'input_factor_E' : 0.5 ,
                    'poisson_input': False,
                    }
    neuron_params = {'V0_mean': -150.,
                     'V0_sd': 50.,
                    # neuron parameters
                    'single_neuron_dict': single_neuron_dict,                     
                     }
    network_params = {'N_scaling': 1.,
                      'K_scaling': 1.,
                      'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates2.json'),
                      'input_params': input_params,
                      'connection_params': conn_params,
                      'neuron_params': neuron_params}

    sim_params = {'t_sim': 1000.,
                  'master_seed': 20,
                  'num_processes': 1,
                  'local_num_threads': 1,
                  'recording_dict': {'record_vm': False},
                  'areas_simulated': ['V1']}

    theory_params = {'dt': 0.1}

    M = MultiAreaModel(network_params, simulation=False,
                       theory=False,
                       theory_spec=theory_params,
                       analysis= False)
    # p, r = M.theory.integrate_siegert()
    # print("Mean-field theory predicts an average "
    #       "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
    # M.simulation.simulate()
    
    K_in = extract_area_dict(M.K, M.structure, 'V1','V1')
    # print(K_in)
    W_in = extract_area_dict(M.W, M.structure, 'V1','V1')
    # print(W_in)
    
    if False:
        dim = len(pop_list_norm)
        K_matrix = np.zeros((dim, dim))
        W_matrix = np.zeros((dim, dim))
        dim_i = 0
        for target_pop in pop_list_norm:
            dim_j = 0
            for source_pop in pop_list_norm:
                K_matrix[dim_i,dim_j] = K_in[target_pop][source_pop]
                W_matrix[dim_i,dim_j] = W_in[target_pop][source_pop]
                dim_j = dim_j + 1
            dim_i = dim_i + 1

        # print("W_matrix=",W_matrix)
        # print("W_matrix*one",np.dot(W_matrix,np.ones((17,1))))

        g_weight_all = solve_homogeneous_system(K_matrix*W_matrix)
        print("g_weight_all",g_weight_all)
        print("zuiyou_weight=",solve_homogeneous_system(K_matrix*W_matrix))
        current_all = np.dot(K_matrix*W_matrix,g_weight_all)*0.5/1000.
        print("current=",current_all)    

        # dim_i = 0
        # for target_pop in pop_list_norm:
        #     dim_j = 0
        #     for source_pop in pop_list_norm:

        #         print("dim_i=",dim_i)
        #         print("target_pop=",target_pop)
        #         if source_pop[1:] == target_pop[1:]:
        #             W_matrix[dim_i,dim_j] = W_in[target_pop][source_pop]
        #             print("layer{}=layer{}".format(source_pop[1:],target_pop[1:]))
        #         dim_j = dim_j + 1
        #     dim_i = dim_i + 1

        K_in = extract_area_dict(M.K, M.structure, 'V1','V1')
        W_in = extract_area_dict(M.W, M.structure, 'V1','V1')
        # print(K_in)
        dim = len(pop_list_norm)
        K_matrix = np.zeros((4, 4))
        W_matrix = np.zeros((4,4))
        for layer in ['23','4','5','6']:
            dim_i = 0
            for target_pop in ["E","S","P","V"]:
                dim_j = 0
                for source_pop in ["E","S","P","V"]:
                    K_matrix[dim_i,dim_j] = K_in[target_pop+layer][source_pop+layer]
                    W_matrix[dim_i,dim_j] = W_in[target_pop+layer][source_pop+layer]
                    dim_j = dim_j + 1
                dim_i = dim_i + 1
            print("layer=",layer)
            # print("zuiyou_weight=",solve_homogeneous_system(K_matrix*W_matrix))
            g_weight = solve_homogeneous_system(K_matrix*W_matrix)
            print("zuiyou_weight=",solve_homogeneous_system(K_matrix*W_matrix))
            current = np.dot(K_matrix*W_matrix,g_weight)*0.5/1000.
            print("current=",current)       
    
    # print("W=",extract_area_dict(M.W, M.structure, 'V1','V1'))
    # print(len(M.K_matrix[0]))
    # print(len(M.K_matrix[0]))
    tau = []
    # para = deepcopy(M.params)
    # dictionary defining single-cell parameters
    complete_area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                      'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                      'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                      'STPa', '46', 'AITd', 'TH']
    
    currents = {}
    
    updated_g = deepcopy(g_dict)
    beta = 0.01
    
    for target_pop in pop_list_norm:
        currents[target_pop] = {}
        # print(source_pop[0])
        # tau.append(single_neuron_dict[source_pop[0]]['tau_syn'])  
        # print("K_0=",K_in[source_pop])
        # print("W_0=",W_in[source_pop])
        # print(tau)       
        rate = 10
        i_total = 0.
        for source_pop in pop_list_norm:
            i_total = i_total + K_in[target_pop][source_pop]*W_in[target_pop][source_pop]*0.5*rate*1e-3
            currents[target_pop][source_pop] = K_in[target_pop][source_pop]*W_in[target_pop][source_pop]*0.5*rate*1e-3
            # print("i_predict=",K_in[target_pop][source_pop]*W_in[target_pop][source_pop]*0.5*rate) 
        currents[target_pop]["total"] = i_total
        print("i_total=",currents[target_pop]["total"]) 
        
        # updated_value = 0.
        # for source_pop in pop_list_norm:
        #     if K_in[target_pop][source_pop] > 0:
        #         # if source_pop[0] == "E":
        #         if True:
        #             # print(source_pop)
        #             updated_value = updated_g[source_pop] - beta * currents[target_pop]["total"]
        #             if updated_value > 0:
        #                 updated_g[source_pop] = updated_value
        #             else: 
        #                 updated_g[source_pop] = 0.   
            
        # print("updated_g=",updated_g)
                # else:
                #     updated_value = updated_g[source_pop] - beta * currents[target_pop]["total"] / (K_in[target_pop][source_pop]*0.5*rate*(-4.))

                # if updated_value > 0:
                #     updated_g[source_pop] = updated_value
                # else: 
                #     updated_g[source_pop] = 0    
    
    if showmatrix:    
        # 将 K_in 和 W_in 转换为 pandas DataFrame
        K_in_df = pd.DataFrame(K_in).applymap(round_to_three)
        W_in_df = pd.DataFrame(W_in).applymap(round_to_three)
        currents_df = pd.DataFrame(currents).applymap(round_to_three)
        #保留三位小数
        currents_df = currents_df.round(3)

        # 绘制 K_in 表格并保存为图片
        plt.figure(figsize=(8, 4))
        plt.axis('off')

        table = plt.table(cellText=K_in_df.values, colLabels=K_in_df.columns, rowLabels=K_in_df.index, cellLoc='center', loc='center')

        # 调整表格字体大小和边框
        table.auto_set_font_size(False)
        table.set_fontsize(12)  # 增大字体
        table.scale(1.5, 1.5)  # 调整表格大小


        # 调整单元格边框
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.5)  # 变细边框
        # plt.title('K_in Table')
        plt.savefig('K_in_table.png', bbox_inches='tight', dpi=300)
        plt.close()

        # 绘制 W_in 表格并保存为图片
        plt.figure(figsize=(8, 4))
        plt.axis('off')
        table = plt.table(cellText=W_in_df.values, colLabels=W_in_df.columns, rowLabels=W_in_df.index, cellLoc='center', loc='center')
        # 调整表格字体大小和边框
        table.auto_set_font_size(False)
        table.set_fontsize(5)  # 增大字体
        # table.scale(1.5, 1.5)  # 调整表格大小

        # 调整单元格边框
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.5)  # 变细边框    
        # plt.title('W_in Table')
        plt.savefig('W_in_table.png', bbox_inches='tight', dpi=300)
        plt.close()       

        #绘制神经元之间的电流值
        plt.figure(figsize=(8, 4))
        plt.axis('off')
        table = plt.table(cellText=currents_df.values, colLabels=currents_df.columns, rowLabels=currents_df.index, cellLoc='center', loc='center')

        # 调整表格字体大小和边框
        table.auto_set_font_size(False)
        table.set_fontsize(5)  # 增大字体

        # 找到每列的最大值
        max_values = currents_df.max()

        # 调整单元格边框并标记最大值
        for (i, j), cell in table.get_celld().items():
            cell.set_linewidth(0.5)  # 变细边框

            # 检查是否是数据单元格并且是最大值
            if i > 0 and j >= 0:  # 排除标题行
                if currents_df.iloc[i - 1, j] == max_values[j]:
                    cell.set_facecolor('yellow')  # 设置背景色为黄色

        # 找到每列的最小值
        min_values = currents_df.min()

        # 调整单元格边框并标记最小值
        for (i, j), cell in table.get_celld().items():
            cell.set_linewidth(0.5)  # 变细边框

            # 检查是否是数据单元格并且是最小值
            if i > 0 and j >= 0:  # 排除标题行
                if currents_df.iloc[i - 1, j] == min_values[j]:
                    cell.set_facecolor('red')  # 设置背景色为红色

        # plt.title('W_in Table')
        
        # 获取当前时间并格式化
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'currents/currents_{current_time}.png'  # 生成新的文件名
        print("filename=",filename)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    out_g = deepcopy(updated_g)
    return out_g

def pltmatrixs(K_in,W_in,currents):
        # 将 K_in 和 W_in 转换为 pandas DataFrame
        K_in_df = pd.DataFrame(K_in).applymap(round_to_three)
        W_in_df = pd.DataFrame(W_in).applymap(round_to_three)
        currents_df = pd.DataFrame(currents).applymap(round_to_three)
        #保留三位小数
        currents_df = currents_df.round(3)

        # 绘制 K_in 表格并保存为图片
        plt.figure(figsize=(8, 4))
        plt.axis('off')

        table = plt.table(cellText=K_in_df.values, colLabels=K_in_df.columns, rowLabels=K_in_df.index, cellLoc='center', loc='center')

        # 调整表格字体大小和边框
        table.auto_set_font_size(False)
        table.set_fontsize(12)  # 增大字体
        table.scale(1.5, 1.5)  # 调整表格大小


        # 调整单元格边框
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.5)  # 变细边框
        # plt.title('K_in Table')
        plt.savefig('K_in_table.png', bbox_inches='tight', dpi=300)
        plt.close()

        # 绘制 W_in 表格并保存为图片
        plt.figure(figsize=(8, 4))
        plt.axis('off')
        table = plt.table(cellText=W_in_df.values, colLabels=W_in_df.columns, rowLabels=W_in_df.index, cellLoc='center', loc='center')
        # 调整表格字体大小和边框
        table.auto_set_font_size(False)
        table.set_fontsize(5)  # 增大字体
        # table.scale(1.5, 1.5)  # 调整表格大小

        # 调整单元格边框
        for key, cell in table.get_celld().items():
            cell.set_linewidth(0.5)  # 变细边框    
        # plt.title('W_in Table')
        plt.savefig('W_in_table.png', bbox_inches='tight', dpi=300)
        plt.close()       

        #绘制神经元之间的电流值
        plt.figure(figsize=(8, 4))
        plt.axis('off')
        table = plt.table(cellText=currents_df.values, colLabels=currents_df.columns, rowLabels=currents_df.index, cellLoc='center', loc='center')

        # 调整表格字体大小和边框
        table.auto_set_font_size(False)
        table.set_fontsize(5)  # 增大字体

        # 找到每列的最大值
        max_values = currents_df.max()

        # 调整单元格边框并标记最大值
        for (i, j), cell in table.get_celld().items():
            cell.set_linewidth(0.5)  # 变细边框

            # 检查是否是数据单元格并且是最大值
            if i > 0 and j >= 0:  # 排除标题行
                if currents_df.iloc[i - 1, j] == max_values[j]:
                    cell.set_facecolor('yellow')  # 设置背景色为黄色

        # 找到每列的最小值
        min_values = currents_df.min()

        # 调整单元格边框并标记最小值
        for (i, j), cell in table.get_celld().items():
            cell.set_linewidth(0.5)  # 变细边框

            # 检查是否是数据单元格并且是最小值
            if i > 0 and j >= 0:  # 排除标题行
                if currents_df.iloc[i - 1, j] == min_values[j]:
                    cell.set_facecolor('red')  # 设置背景色为红色

        # plt.title('W_in Table')
        
        # 获取当前时间并格式化
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'currents/currents_{current_time}.png'  # 生成新的文件名
        print("filename=",filename)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()    


# for i in range(1):
if False:
    # # 修改字典中的值为随机生成的-1到1之间的值，然后取2的幂
    # for key in g_dict.keys():
    #     random_value = random.uniform(-1, 1)  # 生成-1到1之间的随机数
    #     g_dict[key] = 2 ** random_value  # 取2的幂

    weight_dict= {'H1': 0.1, 'E23': 1., 'S23': 1., 'V23': 1., 'P23': 1., 'E4': 0.8, 'S4': 1., 'V4': 1., 'P4': 1., 'E5': 1.6, 'S5': 0.5, 'V5': 3., 'P5': 1., 'E6': 0.8, 'S6': 1., 'V6': 2., 'P6': 2.}

    print(weight_dict)

    computer_current(weight_dict,showmatrix=True)
    # weight_dict = computer_current(weight_dict,showmatrix=True)
    if False:
        for i in range(300):
            # if i%10 == 0:
            if True:
                print("i=",i)
                weight_dict = computer_current(weight_dict,showmatrix=True)
            else:
                weight_dict = computer_current(weight_dict)
            print("weight_dict=",weight_dict)

# input_current = {
#                 "H1" : 83.+17.5+10.*0,
#                 "V23" : 83.+17.5+10.*0,
#                 "S23" : 441+10.*0,
#                 "E23" : 501+10.*0,
#                 "P23" : 720+0.1+10.*0,
#                 "V4" : 83.+17.5+10.*0,
#                 "S4" : 441+10.*0,
#                 "E4" : 501+10.*0,
#                 "P4" : 720+0.1+10.*0,
#                 "V5" : 83.+17.5+10.*0,
#                 "S5" : 441+10.*0,
#                 "E5" : 501+10.*0,
#                 "P5" : 720+0.1+10.*0,
#                 "V6" : 83.+17.5+10.*0,
#                 "S6" : 441+10.*0,
#                 "E6" : 501+10.*0,
#                 "P6" : 720+0.1+10.*0,                        
#             }

# input_current = {
#                 "H1" : 501+10.*0,
#                 "V23" : 501+10.*0,
#                 "S23" : 501+10.*0,
#                 "E23" : 501+10.*0,
#                 "P23" : 501+10.*0,
#                 "V4" : 501+10.*0,
#                 "S4" : 501+10.*0,
#                 "E4" : 501+10.*1,
#                 "P4" : 501+10.*0,
#                 "V5" : 501+10.*0,
#                 "S5" : 501+10.*0,
#                 "E5" : 501+10.*1,
#                 "P5" : 501+10.*0,
#                 "V6" : 501+10.*0,
#                 "S6" : 501+10.*0,
#                 "E6" : 501+10.*1,
#                 "P6" : 501+10.*0,              
#             }

# input_update = deepcopy(input_current)

# weight_dict= {'H1': 0.1, 'E23': 1., 'S23': 1., 'V23': 1., 'P23': 1., 'E4': 0.8, 'S4': 1., 'V4': 1., 'P4': 1., 'E5': 1.6, 'S5': 0.5, 'V5': 3., 'P5': 1., 'E6': 0.8, 'S6': 1., 'V6': 2., 'P6': 2.}

for i in range(1):
    
    wi = 1.
    input_current = {
                    "H1" : 501+10.*0,
                    "V23" : 501+10.*0,
                    "S23" : 501-100.*0,
                    "E23" : 501+200.*0,
                    "P23" : 501+10.*0,
                    "V4" : 501+10.*0,
                    "S4" : 501+10.*0,
                    "E4" : 501+100*0,
                    "P4" : 501+10.*0,
                    "V5" : 501+10.*0,
                    "S5" : 501+10.*0,
                    "E5" : 501+100*0,
                    "P5" : 501+10.*0,
                    "V6" : 501+10.*0,
                    "S6" : 501+10.*0,
                    "E6" : 501+100*0,
                    "P6" : 501+10.*0,              
                }

    input_update = deepcopy(input_current)

    weight_dict= {'H1': 0.1, 'E23': 1., 'S23': 1., 'V23': 1., 'P23': 1., 'E4': 0.8, 'S4': 1., 'V4': 1., 'P4': 1., 'E5': 1.6, 'S5': 0.5, 'V5': 3., 'P5': 1., 'E6': 0.8, 'S6': 1., 'V6': 2., 'P6': 2.}

    computer_current(weight_dict,showmatrix=True)
    start_time = time.time()
    # 将新的时间戳转换为本地时间的struct_time对象
    # local_time = time.localtime(start_time)
    print(time.strftime("start_time:%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    pred_time = start_time + 10252.23166012764
    print(time.strftime("pred_time:%Y-%m-%d %H:%M:%S", time.localtime(pred_time)))

    """
    Down-scaled model.
    Neurons and indegrees are both scaled down to 10 %.
    Can usually be simulated on a local machine.

    Warning: This will not yield reasonable dynamical results from the
    network and is only meant to demonstrate the simulation workflow.
    """

    conn_params = {'replace_non_simulated_areas': 'hom_poisson_stat',
                   'g': -4.,
                   'g_H' : -4.,
                   'g_V' : -4.,
                   'g_S' : -4.,
                   'g_P' : -4.,
                    'g_dict': weight_dict,                
                   
                #    'K_stable': os.path.join(base_path, "K_stable.npy"),
                    'K_stable':None,
                   'fac_nu_ext_TH': 1.2,
                   'fac_nu_ext_23E': 2.0,
                   'fac_nu_ext_4E': 2.0,
                   'fac_nu_ext_5E': 1.2,
                   'fac_nu_ext_6E': 1.,
                   'fac_nu_ext_23': 1.,
                   'fac_nu_ext_4': 1.0,
                   'fac_nu_ext_5': 1.0,
                   'fac_nu_ext_6': 1.,
                   'fac_nu_ext_1H':  1.0,
                   'fac_nu_ext_23V': 1.2,
                   'fac_nu_ext_4V': 3.,
                   'fac_nu_ext_5V': 3.0,
                   'fac_nu_ext_6V': 3.,  
                   'fac_nu_ext_23S': 0.5,
                   'fac_nu_ext_4S': 0.3,
                   'fac_nu_ext_5S': 0.3,
                   'fac_nu_ext_6S': 0.3,   
                   'fac_nu_ext_23P': 1.,
                   'fac_nu_ext_4P': 1.,
                   'fac_nu_ext_5P': 1.,
                   'fac_nu_ext_6P': 1.,              
                   'PSP_e_23_4': 0.30,
                   'PSP_e_5_h1': 0.15,
                   'PSP_e': 0.15,
                   'av_indegree_V1': 3950.}
    input_params = {'rate_ext': 40, 
                    'input_factor_E' : 0.5,
                    "input": input_update,
                    'poisson_input': False,}
    neuron_params = {'V0_mean': -150.,
                     'V0_sd': 50.,
                     'single_neuron_dict': single_neuron_dict
                     }
    network_params = {'N_scaling': 1.,
                      'K_scaling': 1.,
                      'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates2.json'),
                      'input_params': input_params,
                      'connection_params': conn_params,
                      'neuron_params': neuron_params}

    sim_params = {'t_sim': 1000.,
                  'master_seed': 20,
                  'num_processes': 1,
                  'local_num_threads': 1,
                  'recording_dict': {'record_vm': False},
                  'areas_simulated': ['V1'],
                  "cut_connect" : False}

    theory_params = {'dt': 0.1}

    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=False,
                       theory_spec=theory_params,
                       analysis= True)
    # p, r = M.theory.integrate_siegert()
    # print("Mean-field theory predicts an average "
    #       "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
    M.simulation.simulate()
    # M.simulation.prepare()
    # K_in = extract_area_dict(M.K, M.structure, 'V1','V1')
    # W_in = extract_area_dict(M.W, M.structure, 'V1','V1')
    # print("W=",extract_area_dict(M.W, M.structure, 'V1','V1'))
    # print(len(M.K_matrix[0]))
    # print(len(M.K_matrix[0]))
    # tau = []
    # para = deepcopy(M.params)
    # dictionary defining single-cell parameters
    # complete_area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
    #                   'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
    #                   'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
    #                   'STPa', '46', 'AITd', 'TH']
    # for target_pop in population_list:
        # print(source_pop[0])
        # tau.append(single_neuron_dict[source_pop[0]]['tau_syn'])  
        # print("K_0=",K_in[source_pop])
        # print("W_0=",W_in[source_pop])
        # print(tau)       
        # rate = 10
        # i_total = 0.
        # for source_pop in population_list:
        #     i_total = i_total + K_in[target_pop][source_pop]*W_in[target_pop][source_pop]*0.5*rate
        #     # print("i_predict=",K_in[target_pop][source_pop]*W_in[target_pop][source_pop]*0.5*rate) 
            
        # print("i_total=",i_total*1e-6)            
    # print(M.params['neuron_params']['single_neuron_dict']['V']['tau_syn'])
    # print(len(tau))
    M.analysis.load_data()
    M.analysis.create_pop_rates()
    
    K_in = extract_area_dict(M.K, M.structure, 'V1','V1')
    W_in = extract_area_dict(M.W, M.structure, 'V1','V1')
    current_matrix = np.zeros((17,17))
    dim_i = 0
    for target_pop in pop_list_norm:
        dim_j = 0
        for source_pop in pop_list_norm:
            rate = M.analysis.pop_rates['V1'][target_pop][0]
            current = rate*K_in[target_pop][source_pop]*W_in[target_pop][source_pop]*0.5*rate
            current_matrix[dim_i,dim_j] = current
            dim_j = dim_j + 1
            
        dim_i = dim_i + 1
        
    # Plot the current matrix
    # plt.figure(figsize=(8, 6))
    # plt.imshow(current_matrix, cmap='viridis', interpolation='nearest')
    # plt.colorbar(label='Current Value')
    plt.imshow(current_matrix)
    # plt.title('Current Matrix')
    # plt.xlabel('Source Population Index')
    # plt.ylabel('Target Population Index')
    plt.savefig("current.png")
    
    M.analysis.create_pop_rate_dists()
    M.analysis.create_rate_time_series()

    for area in M.analysis.areas_loaded:
        if area == 'V1':
            M.analysis.multi_rate_display(area,pops = pop_list_norm,output = "png")
            M.analysis.multi_voltage_display(area,pops = pop_list_norm,output = "png")
            M.analysis.multi_current_display(area,pops = pop_list_norm,output = "png")
            current_dict = M.analysis.avg_current_display(area,pops = pop_list_norm,t_min=500,output = "png")
        # else:
            # M.analysis.multi_rate_display(area,pops = pop_list_TH,output = "png")
            
    for area in M.analysis.areas_loaded:
        if area == 'V1':
            for pop in pop_list_norm:
                print(pop)
                print(pop_list_norm)
                M.analysis.single_rate_display(area=area,pop=pop,output = "png")
                # frac_neurons : float, [0,1]
                frac_neurons = 0.01
                M.analysis.single_dot_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                M.analysis.single_voltage_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                M.analysis.single_current_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                M.analysis.single_power_display(area=area,pop=pop,output = "png",resolution=0.2)

                if pop[0] == "S":
                     input_update[pop] = 0.1*(input_current[pop] - current_dict[pop]) + input_current[pop]
    
    print("input_update=",input_update)
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
from multiarea_model.default_params import single_neuron_dict
import os
import numpy as np
import matplotlib.pyplot as plt
from pygenn import GeNNModel, init_postsynaptic, init_sparse_connectivity, init_var, init_weight_update
from multiarea_model import MultiAreaModel
from config import base_path
import csv
from scipy.optimize import fsolve

# print(single_neuron_dict['E'])

#设置GPU的序号
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#定义网络模型
conn_params = {'replace_non_simulated_areas': 'het_poisson_stat',
               'g': -3.,
            #    'K_stable': os.path.join(base_path, "K_stable.npy"),
                'K_stable':None,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 2,
               'PSP_e_23_4': 0.15,
               'av_indegree_V1': 3950.}
input_params = {'rate_ext': 20}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
network_params = {'N_scaling': 0.5,
                  'K_scaling': 0.5,
                  'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates2.json'),
                  'input_params': input_params,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': 1000.,
              'master_seed': 20,
              'num_processes': 1,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False}}

theory_params = {'dt': 0.1}

M = MultiAreaModel(network_params, simulation=False,
                   sim_spec=sim_params,
                   theory=False,
                   theory_spec=theory_params,
                   analysis= False)

K_ext = M.K["V1"]["E23"]['external']['external']
W_ext = M.W["V1"]["E23"]['external']['external']
print("k=",K_ext)
print("w=",W_ext)
tau_syn = M.params['neuron_params']['single_neuron_dict']['E']['tau_syn']
DC = K_ext * W_ext * tau_syn * 1.e-3 * \
                    M.params['input_params']['rate_ext']
                
# print(M.synapses["V1"]["E23"]["V1"]["E23"])
# print(M.K_matrix.shape)


# Neuron number
num = 10

def computer_firing_rate(rate_ext = 10,neu_type = "P"):
    model = GeNNModel("float","f-i-curve")
    model.dt = 0.1

    lif_params = {
                  "E":
                  {"C": single_neuron_dict["E"]['C_m'] / 1000.0, 
                  "TauM": single_neuron_dict["E"]['tau_m'], 
                  "Vrest": single_neuron_dict["E"]['E_L'], 
                  "Vreset": single_neuron_dict["E"]['V_reset'], 
                  "Vthresh" : single_neuron_dict["E"]['V_th'],
                  "TauRefrac": single_neuron_dict["E"]['t_ref']},
                  "S":
                  {"C": single_neuron_dict["S"]['C_m'] / 1000.0, 
                  "TauM": single_neuron_dict["S"]['tau_m'], 
                  "Vrest": single_neuron_dict["S"]['E_L'], 
                  "Vreset": single_neuron_dict["S"]['V_reset'], 
                  "Vthresh" : single_neuron_dict["S"]['V_th'],
                  "TauRefrac": single_neuron_dict["S"]['t_ref']},
                  "P":
                  {"C": single_neuron_dict["P"]['C_m'] / 1000.0, 
                  "TauM": single_neuron_dict["P"]['tau_m'], 
                  "Vrest": single_neuron_dict["P"]['E_L'], 
                  "Vreset": single_neuron_dict["P"]['V_reset'], 
                  "Vthresh" : single_neuron_dict["P"]['V_th'],
                  "TauRefrac": single_neuron_dict["P"]['t_ref']},
                  "H":
                  {"C": single_neuron_dict["H"]['C_m'] / 1000.0, 
                  "TauM": single_neuron_dict["H"]['tau_m'], 
                  "Vrest": single_neuron_dict["H"]['E_L'], 
                  "Vreset": single_neuron_dict["H"]['V_reset'], 
                  "Vthresh" : single_neuron_dict["H"]['V_th'],
                  "TauRefrac": single_neuron_dict["H"]['t_ref']},
                  "V":
                  {"C": single_neuron_dict["V"]['C_m'] / 1000.0, 
                  "TauM": single_neuron_dict["V"]['tau_m'], 
                  "Vrest": single_neuron_dict["V"]['E_L'], 
                  "Vreset": single_neuron_dict["V"]['V_reset'], 
                  "Vthresh" : single_neuron_dict["V"]['V_th'],
                  "TauRefrac": single_neuron_dict["V"]['t_ref']}
                  } 

    pop_lif_params = lif_params[neu_type]
    pop_lif_params['Ioffset'] = 0.0

    # lif_params = {"C": single_neuron_dict['E']['C_m'], "TauM": 20.0, "Vrest": -49.0, "Vreset": -60.0,
    #               "Vthresh": -50.0, "Ioffset": 0.0, "TauRefrac": 5.0}

    lif_init = {"V": init_var("Uniform", {"min": -60.0, "max": -50.0}),
                "RefracTime": 0.0}

    pop = model.add_neuron_population(neu_type, num, "LIF", pop_lif_params, lif_init)

    pop.spike_recording_enabled = True

    # model.add_current_source("CurrentSource", "DC", pop, {"amp": current_input}, {})
    ext_weight = 87.8e-3   # nA
    # ext_input_rate = NUM_EXTERNAL_INPUTS[layer][pop] * args.connectivity_scale * BACKGROUND_RATE
    
    # rate_ext = 10
    K_ext = M.K["V1"][neu_type+"23"]['external']['external']
    # ext_input_rate = K_ext* 1.0 * rate_ext
    ext_input_rate = rate_ext
    # ext_input_rate = 10
    
    
    poisson_params = {"weight": ext_weight, "tauSyn": 0.5, "rate": ext_input_rate}
    poisson_init = {"current": 0.0}
    model.add_current_source("CurrentSource" + "_poisson", "PoissonExp", pop, poisson_params, poisson_init)

    model.build()
    model.load(num_recording_timesteps=2000)

    voltage = pop.vars["V"]

    voltages = []
    while model.t < 200.0:
        model.step_time()
        voltage.pull_from_device()
        voltages.append(voltage.values)

    # Stack voltages together into a 2000x4 matrix
    voltages = np.vstack(voltages)

    model.pull_recording_buffers_from_device()
    _, spike_ids = pop.spike_recording_data[0]
    # print("f=",len(pop.spike_recording_data[0]))
    rate = len(spike_ids)/(num*0.2)
    # print(spike_ids)
    print("rate=",rate)
    return rate 

# rates = np.arange(0.0, 20.0, 1)
# firing_rates = [computer_firing_rate(i,"S") for i in rates]
# plt.plot(rates, firing_rates)
# plt.xlabel("rates(Hz)")
# plt.ylabel("firing rate(Hz)")
# plt.savefig("20240804.png")

# # 保存数据
# with open('firing_rates.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['input_rate(Hz)', 'firing_rate(Hz)', 'neu_type'])  # 写入表头
#     for neu_type in ["E", "P", "S", "V"]:
#         rates = np.arange(5000, 6000.0, 10)
#         firing_rates = [computer_firing_rate(i, neu_type) for i in rates]
#         for rate_value, firing_rate in zip(rates, firing_rates):
#             writer.writerow([rate_value, firing_rate, neu_type])

# # 从文件中读取数据并绘图
# data = {}
# with open('firing_rates.csv', 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)  # 跳过表头
#     for row in reader:
#         rate_value, firing_rate, neu_type = float(row[0]), float(row[1]), row[2]
#         if neu_type not in data:
#             data[neu_type] = {'input_rate(Hz)': [], 'firing_rate(Hz)': []}
#         data[neu_type]['input_rate(Hz)'].append(rate_value)
#         data[neu_type]['firing_rate(Hz)'].append(firing_rate)

# # 画图
# for neu_type in data:
#     plt.plot(data[neu_type]['input_rate(Hz)'], data[neu_type]['firing_rate(Hz)'], label=neu_type)
    
# # 添加图例和标签
# plt.legend()
# plt.xlabel('input rate(Hz)')
# plt.ylabel('firing rate(Hz)')
# plt.savefig("20240813.png")
# plt.show()


firing_function = lambda r: computer_firing_rate(r,neu_type = "E")*1000-10
solution = fsolve(firing_function,5800)
print("solution=",solution)
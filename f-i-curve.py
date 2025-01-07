from multiarea_model.default_params import single_neuron_dict
import numpy as np
import matplotlib.pyplot as plt
from pygenn import GeNNModel, init_postsynaptic, init_sparse_connectivity, init_var, init_weight_update,create_neuron_model
import csv
import os 

#设置GPU的序号
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
    "S":{
    # Leak potential of the neurons .
    'E_L': -76.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 800.0, # pF
    # Membrane time constant .
    'tau_m': 50.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 1.0 # ms
    },
    "P":{
    # Leak potential of the neurons .
    'E_L': -86.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -60.0, # mV
    # Membrane capacitance .
    'C_m': 200.0, # pF
    # Membrane time constant .
    'tau_m': 10.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 1.0 # ms
    },
    'V':{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -65.0, # mV
    # Membrane capacitance .
    'C_m': 100.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 1.0 # ms
    },
    'H':{
    # Leak potential of the neurons .
    'E_L': -70.0, # mV
    # Threshold potential of the neurons .
    'V_th': -50.0, # mV
    # Membrane potential after a spike .
    'V_reset': -65.0, # mV
    # Membrane capacitance .
    'C_m': 100.0, # pF
    # Membrane time constant .
    'tau_m': 20.0, # ms
    # Time constant of postsynaptic currents .
    'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 1.0 # ms
    }
    } 


# #定义网络模型
# conn_params = {'replace_non_simulated_areas': 'het_poisson_stat',
#                'g': -3.,
#             #    'K_stable': os.path.join(base_path, "K_stable.npy"),
#                 'K_stable':None,
#                'fac_nu_ext_TH': 1.2,
#                'fac_nu_ext_5E': 1.125,
#                'fac_nu_ext_6E': 2,
#                'PSP_e_23_4': 0.15,
#                'av_indegree_V1': 3950.}
# input_params = {'rate_ext': 20}
# neuron_params = {'V0_mean': -150.,
#                  'V0_sd': 50.}
# network_params = {'N_scaling': 0.5,
#                   'K_scaling': 0.5,
#                   'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates2.json'),
#                   'input_params': input_params,
#                   'connection_params': conn_params,
#                   'neuron_params': neuron_params}

# sim_params = {'t_sim': 1000.,
#               'master_seed': 20,
#               'num_processes': 1,
#               'local_num_threads': 1,
#               'recording_dict': {'record_vm': False}}

# theory_params = {'dt': 0.1}

# M = MultiAreaModel(network_params, simulation=True,
#                    sim_spec=sim_params,
#                    theory=False,
#                    theory_spec=theory_params,
#                    analysis= True)

# print(M.K)
# print(single_neuron_dict['E'])

lif_model = create_neuron_model(
    "lif_model",
    params=["Vthresh", "TauM", "TauRefrac", "C", "Vrest", "Vreset","Ioffset"],  # 添加参数
    vars=[("V", "scalar"), ("RefracTime", "scalar")],  # 添加不应期状态变量
    sim_code="""
    if (RefracTime <= 0.0) {
        scalar alpha = ((Isyn + Ioffset) * Rmembrane) + Vrest;
         V = alpha - (ExpTC * (alpha - V));
    } else {
        RefracTime -= dt;
    }
    """,  # 更新电压并考虑膜电容和膜电位
    threshold_condition_code="(RefracTime <= 0.0) && (V >= Vthresh)",  # 仅在不应期外检查阈值条件
    reset_code="""
    V = Vreset;
    RefracTime = TauRefrac;
    """,  # 触发后重置电压并进入不应期
    derived_params=[("ExpTC", lambda pars, dt: np.exp(-dt / pars["TauM"])),
                    ("Rmembrane",lambda pars,dt:pars["TauM"]/pars["C"])
                    ],
)


# Neuron number
num = 1000

def computer_firing_rate(current_input,neu_type = "P"):
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

    # pop = model.add_neuron_population(neu_type, num, lif_model, pop_lif_params, lif_init)
    pop = model.add_neuron_population(neu_type, num, lif_model, pop_lif_params, lif_init)

    pop.spike_recording_enabled = True

    model.add_current_source("CurrentSource", "DC", pop, {"amp": current_input}, {})

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

# for neu_type in ["E","P","S","V"]:
#     current_inputs = np.arange(0.0, 20.0, 0.1)
#     firing_rates = [computer_firing_rate(i,"V") for i in current_inputs]
#     plt.plot(current_inputs, firing_rates,label = neu_type)
    
    
# # 添加图例
# plt.legend()
# plt.xlabel("input(nA)")
# plt.ylabel("firing rate(Hz)")
# plt.savefig("20240804.png")

# 保存数据
with open('firing_inputs.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['input(nA)', 'firing_rate', 'neu_type'])  # 写入表头
    for neu_type in ["E", "P", "S", "V"]:
        current_inputs = np.arange(0.0, 2.0, 0.01)
        firing_rates = [computer_firing_rate(i, neu_type) for i in current_inputs]
        for input_value, firing_rate in zip(current_inputs, firing_rates):
            writer.writerow([input_value, firing_rate, neu_type])

# 从文件中读取数据并绘图
data = {}
with open('firing_inputs.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过表头
    for row in reader:
        input_value, firing_rate, neu_type = float(row[0]), float(row[1]), row[2]
        if neu_type not in data:
            data[neu_type] = {"inputs": [], "rates": []}
        data[neu_type]["inputs"].append(input_value)
        data[neu_type]["rates"].append(firing_rate)

# 画图
for neu_type in data:
    plt.plot(data[neu_type]["inputs"], data[neu_type]["rates"], label=neu_type)

# 添加图例和标签
plt.legend()
plt.xlabel("input(nA)")
plt.ylabel("firing rate(Hz)")
plt.savefig("20240826.png")
plt.show()
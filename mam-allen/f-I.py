import numpy as np
import matplotlib.pyplot as plt
from multiarea_model.default_params import single_neuron_dict
import csv

def non_noise(tau_ref, V_reset, E_L, I, V_th, C_m, tau_m):
    V_ss = E_L + I*tau_m/C_m
    # I_c = C_m/tau_m*(V_th-V_reset)
    if V_ss > V_th:
        r_reciprocal = tau_ref + tau_m*np.log((V_ss-V_reset)/(V_ss-V_th))
        return 1/r_reciprocal
    else:
        return 0.

single_neuron_dict = {
    "V1":{
    # Leak potential of the neurons .
    'E_L': -65.50, # mV
    # Threshold potential of the neurons .
    'V_th': -40.20, # mV
    # Membrane potential after a spike .
    'V_reset': -65.50, # mV, 
    # Membrane capacitance .
    'C_m': 37.11, # pF
    # Membrane time constant .
    # 'tau_m': 20.0, # ms
    # leak conductance
    'g_L': 4.07, #ns
    # Time constant of postsynaptic currents .
    # 'tau_syn': 0.5, # ms
    # Refractory period of the neurons after a spike .
    't_ref': 3.5 # ms
    },    
    
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
color_map = {
    'E' : '#5555df',
    'P' : '#048006',
    'V' : '#a6a123',
    'S' : '#c82528'
}
                # if pop.find('E') > (-1):
                #     # pcolor = '#595289'
                #     pcolor = '#5555df'
                # elif pop.find('P') > (-1):
                #     # pcolor = '#af143c'
                #     pcolor = '#048006'
                # elif pop.find('H') > (-1):
                #     # pcolor = '#6B8E23'
                #     pcolor = '#a6a123'
                # else:
                #     # pcolor = '#006400'
                #     pcolor = '#c82528'
for neu_type in data:
    plt.plot(data[neu_type]["inputs"], data[neu_type]["rates"], label=neu_type + '_sim',color=color_map.get(neu_type, 'black'))

    # lif_params = {
    #               "E":
    #               {"C": single_neuron_dict["E"]['C_m'] / 1000.0, 
    #               "TauM": single_neuron_dict["E"]['tau_m'], 
    #               "Vrest": single_neuron_dict["E"]['E_L'], 
    #               "Vreset": single_neuron_dict["E"]['V_reset'], 
    #               "Vthresh" : single_neuron_dict["E"]['V_th'],
    #               "TauRefrac": single_neuron_dict["E"]['t_ref']},
    C_m = lif_params[neu_type]["C"]
    tau_m = lif_params[neu_type]["TauM"]
    E_L = lif_params[neu_type]["Vrest"]
    V_reset = lif_params[neu_type]["Vreset"]
    V_th = lif_params[neu_type]["Vthresh"]
    tau_ref = lif_params[neu_type]["TauRefrac"]
    
    y_values = [non_noise(tau_ref, V_reset, E_L, i, V_th, C_m, tau_m)*1000 for i in data[neu_type]["inputs"]]
    plt.plot(data[neu_type]["inputs"], y_values, label=neu_type + '_theory',color=color_map.get(neu_type, 'black'),linestyle='--')
# y_values = [integrand(V_ss) for V_ss in V_values]

# 画图
# plt.plot(I, y_values)
plt.xlabel('I(nA)')
plt.ylabel('firing rate' )
plt.title(r'f-I curve')
plt.legend()
plt.grid(True)

plt.show()
plt.savefig("20240826.png")
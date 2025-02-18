"""
default_parameters.py
=====================
This script defines the default values of all
parameters and defines functions to compute
single neuron and synapse parameters and to
properly set the seed of the random generators.

Authors
-------
Maximilian Schmidt
"""

from config import base_path
import json
import os
# import nest

import numpy as np

complete_area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                      'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                      'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                      'STPa', '46', 'AITd', 'TH']

# population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

def get_population_list():
    pop_list = []
    neuron_types = ["E","S","P",'V']
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

f1 = open(os.path.join(base_path, 'multiarea_model/data_multiarea',
                       'viscortex_raw_data.json'), 'r')
raw_data = json.load(f1)
f1.close()
av_indegree_Cragg = raw_data['av_indegree_Cragg']
av_indegree_OKusky = raw_data['av_indegree_OKusky']


"""
Simulation parameters
"""
sim_params = {
    # master seed for random number generators
    'master_seed': 10,
    # simulation step 
    'dt': 0.1, #ms
    # simulated time 
    't_sim': 10.0, #ms
    # no. of MPI processes:
    'num_processes': 1,
    # no. of threads per MPI process':
    'local_num_threads': 1,
    # Areas represented in the network
    'areas_simulated': complete_area_list,
    # Should GeNN record timings
    'timing_enabled': True,
    # Should GeNN use procedural connectivity?
    'procedural_connectivity': True,
    # How many threads per spike should GeNN use?
    'num_threads_per_spike': 1,
    # Should GeNN rebuild model or use existing code?
    'rebuild_model': True,
    # number of recording timesteps
    'recording_buffer_timesteps' : 10,
    # Cut the connection between different layers?
    "cut_connect" : False
}

"""
Network parameters
"""
network_params = {
    # Surface area of each area in mm^2
    'surface': 1.0,
    # Scaling of population sizes
    'N_scaling': 1.,
    # Scaling of indegrees
    'K_scaling': 1.,
    # Absolute path to the file holding full-scale rates for scaling
    # synaptic weights
    'fullscale_rates': None
}


"""
Single-neuron parameters
"""

sim_params.update(
    {
        'initial_state': {
            # mean of initial membrane potential 
            'V_m_mean': -58.0, #mV
            # std of initial membrane potential 
            'V_m_std': 10.0 #mV
        }
    })

# dictionary defining single-cell parameters
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

neuron_params = {
    # neuron model
    'neuron_model': 'iaf_psc_exp',
    # neuron parameters
    'single_neuron_dict': single_neuron_dict,
    # Mean and standard deviation for the
    # distribution of initial membrane potentials
    'V0_mean': -150., # mV
    'V0_sd': 50.} # mV

network_params.update({'neuron_params': neuron_params})


"""
General connection parameters
"""
connection_params = {
    # Whether to apply the stabilization method of
    # Schuecker, Schmidt et al. (2017). Default is False.
    # Options are True to perform the stabilization or
    # a string that specifies the name of a binary
    # numpy file containing the connectivity matrix
    'K_stable': False,

    # Whether to replace all cortico-cortical connections by stationary
    # Poisson input with population-specific rates (het_poisson_stat)
    # or by time-varying current input (het_current_nonstat)
    # while still simulating all areas. In both cases, the data to replace
    # the cortico-cortical input is loaded from `replace_cc_input_source`.
    'replace_cc': False,

    # Whether to replace non-simulated areas by Poisson sources
    # with the same global rate rate_ext ('hom_poisson_stat') or
    # by specific rates ('het_poisson_stat')
    # or by time-varying specific current ('het_current_nonstat')
    # In the two latter cases, the data to replace the cortico-cortical
    # input is loaded from `replace_cc_input_source`
    'replace_non_simulated_areas': None,

    # Source of the input rates to replace cortico-cortical input
    # Either a json file (has to end on .json) holding a scalar values
    # for each population or
    # a base name such that files with names
    # $(replace_cc_input_source)-area-population.npy
    # (e.g. '$(replace_cc_input_source)-V1-23E.npy')
    # contain the time series for each population.
    # We recommend using absolute paths rather than relative paths.
    'replace_cc_input_source': None,

    # whether to redistribute CC synapse to meet literature value
    # of E-specificity
    'E_specificity': True,

    # Relative inhibitory synaptic strength (in relative units).
    'g': -16.,
    'g_H': -2.,
    'g_V': -2.,
    'g_P': -2.,
    'g_S': -2.,
    
    'alpha_norm': {
          'H1':  1,
          'E23': 1,
          'S23': 1,
          'V23': 1,
          'P23': 1,  
          'E4':  1,
          'S4':  1,
          'V4':  1,
          'P4':  1,
          'E5':  1,
          'S5':  1,
          'V5':  1,
          'P5':  1,
          'E6':  1,
          'S6':  1,
          'V6':  1,
          'P6':  1,
        },
    'alpha_TH':
        {
          'H1':  1,
          'E23': 1,
          'S23': 1,
          'V23': 1,
          'P23': 1,  
          'E4':  1,
          'S4':  1,
          'V4':  1,
          'P4':  1,
          'E5':  1,
          'S5':  1,
          'V5':  1,
          'P5':  1,
          'E6':  1,
          'S6':  1,
          'V6':  1,
          'P6':  1,
        },
    'beta_norm':{"H1" : 3.9,
                 "E23" : 0.71, "P23" : 0.48, "S23" : 1., "V23" :0.9,
                 "E4" : 1.66, "S4" : 0.24, "V4" : 0.46, "P4" : 0.8,
                 "E5" : 0.95, "S5" : 0.48, "V5" : 1.2, "P5" :1.09,
                 "E6" : 1.12, "S6" : 0.63, "V6" : 0.5, "P6" : 0.42},
    'beta_TH':{"H1" : 1.,
               "E23" : 1., "P23" : 1., "S23" : 1., "V23" : 1.,
               "E4" : 1.,  "S4" : 1.,  "V4" : 1.,  "P4" : 1.,
               "E5" : 1., "S5" : 1.,  "V5" : 1.,  "P5" : 1.,
               "E6" : 1.,  "S6" : 1.,  "V6" : 1.,  "P6" : 1. },
        
    # compute average indegree in V1 from data
    'av_indegree_V1': np.mean([av_indegree_Cragg, av_indegree_OKusky]),

    # synaptic volume density
    # area-specific --> conserves average in-degree
    # constant --> conserve syn. volume density
    'rho_syn': 'constant',

    # Increase the external Poisson indegree onto 23E, 4E, 5E and 6E
    'fac_nu_ext_23E': 1.,
    'fac_nu_ext_4E': 1.,
    'fac_nu_ext_5E': 1.,
    'fac_nu_ext_6E': 1.,
    'fac_nu_ext_23': 1.,
    'fac_nu_ext_4': 1.,
    'fac_nu_ext_5': 1.,
    'fac_nu_ext_6': 1.,
    'fac_nu_ext_1H':  1.,
    'fac_nu_ext_23V': 1.,
    'fac_nu_ext_4V': 1.,
    'fac_nu_ext_5V': 1.,
    'fac_nu_ext_6V': 1.,
    'fac_nu_ext_23S': 1.,
    'fac_nu_ext_4S': 1.,
    'fac_nu_ext_5S': 1.,
    'fac_nu_ext_6S': 1.,
    'fac_nu_ext_23P': 1.,
    'fac_nu_ext_4P': 1.,
    'fac_nu_ext_5P': 1.,
    'fac_nu_ext_6P': 1.,
    # to increase the ext. input to 23E and 5E in area TH
    'fac_nu_ext_TH': 1.,

    # synapse weight parameters for current-based neurons
    # excitatory intracortical synaptic weight 
    'PSP_e': 0.15, # mV
    'PSP_e_23_4': 0.3, #mV
    'PSP_e_5_h1': 0.15, #mV
    # synaptic weight  for external input
    'PSP_ext': 0.15, #mV

    # relative SD of normally distributed synaptic weights
    'PSC_rel_sd_normal': 0.1,
    # relative SD of lognormally distributed synaptic weights
    'PSC_rel_sd_lognormal': 3.0,

    # scaling factor for cortico-cortical connections (chi)
    'cc_weights_factor': 1.,
    # factor to scale cortico-cortical inh. weights in relation
    # to exc. weights (chi_I)
    'cc_weights_I_factor': 0.8,

    # 'switch whether to distribute weights lognormally
    'lognormal_weights': False,
    # 'switch whether to distribute only EE weight lognormally if
    # 'lognormal_weights': True
    'lognormal_EE_only': False,
}

network_params.update({'connection_params': connection_params})

"""
Delays
"""
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
network_params.update({'delay_params': delay_params})

"""
Input parameters
"""
input_params = {
    # Whether to use Poisson or DC input (True or False)
    'poisson_input': True,

    # synapse type for Poisson input
    'syn_type_ext': 'static_synapse_hpc',

    # Rate of the Poissonian spike generator .
    'rate_ext': 10., # Hz
    
    #a factor changes input intensity of excitatory neurons
    'input_factor_E' : 0.5, 

    # Whether to switch on time-dependent DC input
    'dc_stimulus': False,
    
    # rate of current
    "rate_current" : 0.8,
    
    #Input when  rate is 10Hz
    'input':{
                "H1" : 83.+17.5+10.*0,
                "V23" : 83.+17.5+10.*0,
                "S23" : 441+10.*0,
                "E23" : 501+10.*0,
                "P23" : 720+0.1+10.*0,
                "V4" : 83.+17.5+10.*0,
                "S4" : 441+10.*0,
                "E4" : 501+10.*0,
                "P4" : 720+0.1+10.*0,
                "V5" : 83.+17.5+10.*0,
                "S5" : 441+10.*0,
                "E5" : 501+10.*0,
                "P5" : 720+0.1+10.*0,
                "V6" : 83.+17.5+10.*0,
                "S6" : 441+10.*0,
                "E6" : 501+10.*0,
                "P6" : 720+0.1+10.*0,                        
            },
}

network_params.update({'input_params': input_params})

"""
Recording settings
"""
recording_dict = {
    # Which areas to record spike data from
    'areas_recorded': None,

    # voltmeter
    'record_vm':  False,
    # Fraction of neurons to record membrane potentials from
    # in each population if record_vm is True
    'Nrec_vm_fraction': 0.01,

    # Parameters for the spike detectors
    'spike_dict': {
        'label': 'spikes',
        'withtime': True,
        'record_to': ['file'],
        'start': 0.},
    # Parameters for the voltmeters
    'vm_dict': {
        'label': 'vm',
        'start': 0.,
        'stop': 1000.,
        'interval': 0.1,
        'withtime': True,
        'record_to': ['file']}
    }
sim_params.update({'recording_dict': recording_dict})

"""
Theory params
"""

theory_params = {'neuron_params': neuron_params,
                 # Initial rates can be None (start integration at
                 # zero rates), a numpy.ndarray defining the initial
                 # rates or 'random_uniform' which leads to randomly
                 # drawn initial rates from a uniform distribution.
                 'initial_rates': None,
                 # If 'initial_rates' is set to 'random_uniform',
                 # 'initial_rates_iter' defines the number of
                 # different initial conditions
                 'initial_rates_iter': None,
                 # If 'initial_rates' is set to 'random_uniform',
                 # 'initial_rates_max' defines the maximum rate of the
                 # uniform distribution to draw the initial rates from
                 'initial_rates_max': 1000.,
                 # The simulation time of the mean-field theory integration
                 'T': 50.,
                 # The time step of the mean-field theory integration
                 'dt': 0.1,
                 # Time interval for recording the trajectory of the mean-field calcuation
                 # If None, then the interval is set to dt
                 'rec_interval': None}


"""
Helper function to update default parameters with custom
parameters
"""


def nested_update(d, d2):
    for key in d2:
        if isinstance(d2[key], dict) and key in d:
            nested_update(d[key], d2[key])
        else:
            d[key] = d2[key]


def check_custom_params(d, def_d):
    for key, val in d.items():
        if isinstance(val, dict):
            check_custom_params(d[key], def_d[key])
        else:
            try:
                def_val = def_d[key]
            except KeyError:
                raise KeyError('Unused key in custom parameter dictionary: {}'.format(key))

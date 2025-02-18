"""
Model
================

This script modified the model described in Schmidt et al. (2018).(by Maximilian Schmidt,Sacha van Albada)

Authors
--------
Liugangqiang

"""

import numpy as np
import json
import re
import os
import scipy
import scipy.integrate
import pprint
from copy import deepcopy
from nested_dict import nested_dict
from itertools import product
import sys
from config import base_path,data_path
# from default_params import pop_list_norm, pop_list_TH

try:
    from multiarea_model.default_params import network_params, nested_update,pop_list_norm, pop_list_TH
    from multiarea_model.data_multiarea.VisualCortex_Data import process_raw_data
except:
    
    # Get the absolute path of this script's directory
    basepath = os.path.abspath(os.path.dirname(__file__))
    # parent_dir_path = os.path.dirname(os.path.abspath(__file__))
    grandparent_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add the absolute path to the system path
    if basepath not in sys.path:
        sys.path.append(basepath)  
    if grandparent_dir_path not in sys.path:
        sys.path.append(grandparent_dir_path)
    from default_params import network_params, nested_update
    from VisualCortex_Data import process_raw_data

def compute_Model_params(out_label='', mode='default'):
    """
    Compute the parameters of the network, in particular the size
    of populations, external inputs to them, and number of synapses
    in every connection.

    Parameters
    ----------
    out_label : str
        label that is appended to the output files.
    mode : str
        Mode of the function. There are three different modes:
        - default mode (mode='default')
          In default mode, all parameters are set to their default
          values defined in default_params.py .
        - custom mode (mode='custom')
          In custom mode, custom parameters are loaded from a json file
          that has to be stored in 'custom_data_files' and named as
          'custom_$(out_label)_parameter_dict.json' where $(out_label)
         is the string defined in `out_label`.
    """
    basepath = os.path.abspath(os.path.join(os.path.dirname(__file__)))

    # Load and process raw data
    process_raw_data()
    raw_fn = os.path.join(basepath, 'viscortex_raw_data.json')
    proc_fn = os.path.join(basepath, 'viscortex_processed_data.json')
    
    """
    Load data
    """
    with open(raw_fn, 'r') as f:
        raw_data = json.load(f)
    with open(proc_fn, 'r') as f:
        processed_data = json.load(f)

    FLN_EDR_completed = processed_data['FLN_completed']
    SLN_Data = processed_data['SLN_completed']
    Coco_Data = processed_data['cocomac_completed']
    Distance_Data = raw_data['median_distance_data']
    Area_surfaces = raw_data['surface_data']
    Intra_areal = raw_data['Intrinsic_Connectivity']
    # exchange source pops and target pops
    
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
    
    population_list = get_population_list()
    Intra_areal = {}
    for target_pop in population_list:
        Intra_areal[target_pop] = {}
        for source_pop in population_list:
            Intra_areal[target_pop][source_pop]  = raw_data['Intrinsic_Connectivity'][source_pop][target_pop]

    total_thicknesses = processed_data['total_thicknesses']
    laminar_thicknesses = processed_data['laminar_thicknesses']
    Intrinsic_FLN_Data = raw_data['Intrinsic_FLN_Data']
    neuronal_numbers_fullscale = processed_data['realistic_neuronal_numbers']
    num_V1 = raw_data['num_V1']
    binzegger_data = raw_data['Binzegger_Data']

    """
    Define area and population lists.
    Define termination and origin patterns according
    to Felleman and van Essen 91
    """
    # This list of areas is ordered according
    # to their architectural type
    area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                 'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                 'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                 'STPa', '46', 'AITd', 'TH']

    termination_layers = {'F': ['4'], 'M': ['1', '23', '5', '6'], 'C': [
        '1', '23', '4', '5', '6'], 'S': ['1', '23']}
    termination_layers2 = {'F': [4], 'M': [
        1, 2, 3, 5, 6], 'C': [1, 2, 3, 4, 5, 6], 'S': [1, 2, 3]}
    origin_patterns = {'S': ['E23'], 'I': ['E5', 'E6'], 'B': ['E23', 'E5', 'E6']}

    binzegger_pops = list(binzegger_data.keys())
    binzegger_I_pops = [binzegger_pops[i] for i in range(
        len(binzegger_pops)) if binzegger_pops[i].find('b') != -1]
    binzegger_E_pops = [binzegger_pops[i] for i in range(
        len(binzegger_pops)) if binzegger_pops[i].find('b') == -1]
    binzegger_nb_pops = [binzegger_I_pops[i] for i in range(
        len(binzegger_I_pops)) if binzegger_I_pops[i].find('n') != -1]
    binzegger_b_pops = [binzegger_I_pops[i] for i in range(
        len(binzegger_I_pops)) if binzegger_I_pops[i].find('n') == -1]

    # Create structure dictionary with entries for each area
    # and population that actually contains neurons
    structure = {}
    for area in area_list:
        structure[area] = []
        for pop in population_list:
            if neuronal_numbers_fullscale[area][pop] > 0.0:
                structure[area].append(pop)

    """
    If run in custom mode, load custom parameter file and
    overwrite default by custom values for parameters specified
    in the parameter file.
    """
    net_params = deepcopy(network_params)
    if mode == 'default':
        prefix = 'default'
    elif mode == 'custom':
        # prefix = 'custom_data_files/custom'
        prefix = 'custom'
        # print("data_path=",data_path)
        print("parameter_path=",os.path.join(data_path, '.'.join((
            '_'.join((prefix,out_label,'parameter_dict')),
                                                   'json'))))        
        with open(os.path.join(data_path, '.'.join((
            '_'.join((prefix,out_label,'parameter_dict')),
                                                   'json'))), 'r') as f:
            custom_params = json.load(f)
        nested_update(net_params, custom_params)
        # print information on overwritten parameters
        print("\n")
        print("========================================")
        print("Customized parameters")
        print("--------------------")
        pprint.pprint(custom_params)
        print("========================================")

    """
    Define parameter values
    """
    # surface area of each area in mm^2
    surface = net_params['surface']

    conn_params = net_params['connection_params']

    # average indegree in V1 to compute
    # synaptic volume density (determined for V1 and
    # taken to be constant across areas)
    av_indegree_V1 = conn_params['av_indegree_V1']

    # Increase the external poisson indegree onto 23E、4E、5E and 6E
    fac_nu_ext_23E = conn_params['fac_nu_ext_23E']
    fac_nu_ext_4E = conn_params['fac_nu_ext_4E']
    fac_nu_ext_5E = conn_params['fac_nu_ext_5E']
    fac_nu_ext_6E = conn_params['fac_nu_ext_6E']
    fac_nu_ext_23 = conn_params['fac_nu_ext_23']
    fac_nu_ext_4 = conn_params['fac_nu_ext_4']
    fac_nu_ext_5 = conn_params['fac_nu_ext_5']
    fac_nu_ext_6 = conn_params['fac_nu_ext_6']
    fac_nu_ext_1H = conn_params['fac_nu_ext_1H']
    # print("fac_nu_ext_1H=",fac_nu_ext_1H)
    fac_nu_ext_23V = conn_params['fac_nu_ext_23V']
    fac_nu_ext_4V = conn_params['fac_nu_ext_4V']
    fac_nu_ext_5V = conn_params['fac_nu_ext_5V']
    fac_nu_ext_6V = conn_params['fac_nu_ext_6V']
    fac_nu_ext_23S = conn_params['fac_nu_ext_23S']
    fac_nu_ext_4S = conn_params['fac_nu_ext_4S']
    fac_nu_ext_5S = conn_params['fac_nu_ext_5S']
    fac_nu_ext_6S = conn_params['fac_nu_ext_6S']
    fac_nu_ext_23P = conn_params['fac_nu_ext_23P']
    fac_nu_ext_4P = conn_params['fac_nu_ext_4P']
    fac_nu_ext_5P = conn_params['fac_nu_ext_5P']
    fac_nu_ext_6P = conn_params['fac_nu_ext_6P']
    # to increase the ext. input to 23E and 5E in area TH
    fac_nu_ext_TH = conn_params['fac_nu_ext_TH']

    # Single neuron parameters, important to determine synaptic weights
    single_neuron_dict = net_params['neuron_params']['single_neuron_dict']
    C_m_E = single_neuron_dict['E']['C_m']
    C_m_S = single_neuron_dict['S']['C_m']
    C_m_P = single_neuron_dict['P']['C_m']
    C_m_H = single_neuron_dict['H']['C_m']
    C_m_V = single_neuron_dict['V']['C_m']
    tau_m_E = single_neuron_dict['E']['tau_m']
    tau_m_S = single_neuron_dict['S']['tau_m']
    tau_m_P = single_neuron_dict['P']['tau_m']
    tau_m_H = single_neuron_dict['H']['tau_m']
    tau_m_V = single_neuron_dict['V']['tau_m']
    tau_syn_E = single_neuron_dict['E']['tau_syn']
    tau_syn_S = single_neuron_dict['S']['tau_syn']
    tau_syn_P = single_neuron_dict['P']['tau_syn']
    tau_syn_H = single_neuron_dict['H']['tau_syn']
    tau_syn_V = single_neuron_dict['V']['tau_syn']

    # synapse weight parameters for current-based neurons
    # excitatory intracortical synaptic weight (mV)
    PSP_e = conn_params['PSP_e']
    PSP_e_23_4 = conn_params['PSP_e_23_4']
    PSP_e_5_h1 = conn_params['PSP_e_5_h1']
    # synaptic weight (mV) for external input
    PSP_ext = conn_params['PSP_ext']
    # relative strength of inhibitory versus excitatory synapses for CUBA neurons
    g = conn_params['g']
    g_H = conn_params['g_H']
    g_V = conn_params['g_V']
    g_S = conn_params['g_S']
    g_P = conn_params['g_P']
    alpha_TH = conn_params['alpha_TH']
    alpha_norm = conn_params['alpha_norm']
    beta_TH = conn_params['beta_TH']
    beta_norm = conn_params['beta_norm']
    # print("model_g=",alpha_TH)

    # relative SD of normally distributed synaptic weights
    PSC_rel_sd_normal = conn_params['PSC_rel_sd_normal']
    # relative SD of lognormally distributed synaptic weights
    PSC_rel_sd_lognormal = conn_params['PSC_rel_sd_lognormal']

    # scaling factor for cortico-cortical connections (chi)
    cc_weights_factor = conn_params['cc_weights_factor']
    # factor to scale cortico-cortical inh. weights in relation to exc. weights (chi_I)
    cc_weights_I_factor = conn_params['cc_weights_I_factor']

    # switch whether to distribute weights lognormally
    lognormal_weights = conn_params['lognormal_weights']
    # switch whether to distribute only EE weight lognormally if
    # switch_lognormal_weights = True
    lognormal_EE_only = conn_params['lognormal_EE_only']

    # whether to redistribute CC synapse to meet literature value
    # of E-specificity
    E_specificity = True

    """
    Data processing
    ===============

    Neuronal numbers
    ----------------
    """
    # Determine a synaptic volume density for each area
    rho_syn = {}
    for area in area_list:
        # note: the total thickness includes L1. Since L1 can be approximated
        # as having no neurons, rho_syn is a synapse density across all layers.
        rho_syn[area] = av_indegree_V1 * neuronal_numbers_fullscale['V1']['total'] / \
                        (Area_surfaces['V1'] * total_thicknesses['V1'])

    # Compute population sizes by scaling the realistic population
    # sizes down to the assumed area surface
    neuronal_numbers = {}
    b = 0
    for area in neuronal_numbers_fullscale:
        neuronal_numbers[area] = {}
        for pop in neuronal_numbers_fullscale[area]:
            neuronal_numbers[area][pop] = int(neuronal_numbers_fullscale[
                area][pop] / Area_surfaces[area] * surface+0.5)
            neuronal_numbers_fullscale[area][pop] = int(neuronal_numbers_fullscale[area][pop]+0.5)
                
        neuronal_numbers[area]["total"] = 0
        for pop in population_list:
            neuronal_numbers[area]["total"] += neuronal_numbers[area][pop]
        
        b += neuronal_numbers[area]["total"]

    """
    Intrinsic synapses
    ------------------
    The workflow is as follows:
    1. Compute the connection probabilities C'(R) of the
       microcircuit of Potjans & Diesmann (2014) depending on the area radius.
       For this, transform the connection probabilities from
       Potjans & Diesmann (2014), computed with a different
       method of averaging the Gaussian, called C_PD14 (R).
       Then, compute the in-degrees for a microcircuit with
       realistic surface and 1mm2 surface.

    2. Transform this to each area with its specific laminar
       compositions with an area-specific conversion factor
       based on the preservation of relative in-degree between
       different connections.

    3. Compute number of type I synapses.

    4. Compute number of type II synapses as the difference between
       synapses within the full-size area and the 1mm2 area.
    """

    """
    1. Radius-dependent connection probabilities of the microcircuit.
    """

    # constants for the connection probability transfer
    # from Potjans & Diesmann (2014) (PD14)
    # sigma = 0.29653208289812366  # mm
    sigma = 0.30 #mm

    # compute average connection probability with method from PD14
    r_PD14 = np.sqrt(1. / np.pi)
    C_relative_PD14 = 2. / (r_PD14 ** 2) * sigma ** 2 * \
        (1. - np.exp(-r_PD14 ** 2 / (2 * sigma ** 2)))
    
    # New calculation based on Sheng (1985), The distance between two random
    # points in plane regions, Theorem 2.4 on the expectation value of
    # arbitrary functions of distance between points in disks.    
    """
    Define integrand for Gaussian averaging
    """
    def integrand(r, R, sig):
        gauss = np.exp(-r ** 2 / (2 * sig ** 2))
        x1 = np.arctan(np.sqrt((2 * R - r) / (2 * R + r)))
        x2 = np.sin(4 * np.arctan(np.sqrt((2 * R - r) / (2 * R + r))))
        factor = 4 * x1 - x2
        return r * gauss * factor

    """
    To determine the conversion from the microcircuit model to the
    area-specific composition in our model properly, we have to
    scale down the intrinsic FLN from 0.79 to a lower value,
    detailed explanation below. Therefore, we execute the procedure
    twice: First, for realistic area size to obtain numbers for
    Indegree_prime_fullscale and then for 1mm2 areas (Indegree_prime).

    Determine mean connection probability, indegrees and intrinsic FLN
    for full-scale areas.
    """
    """
    Define approximation for function log(1-x) needed for large areas
    """
    def log_approx(x, limit):
        res = 0.
        for k in range(limit):
            res += x ** (k + 1) * (-1.) ** k / (k + 1)
        return res    

    def average_connection(area_surface):
        r_area = np.sqrt(area_surface / np.pi)
        return 2 / area_surface * scipy.integrate.quad(integrand, 0, 2 * r_area, args=(r_area, sigma))[0]

    #modified the method of schmidt
    def average_connection2(area_surface):
        r_area = np.sqrt(area_surface / np.pi)
        return 2. / (r_area ** 2) * sigma ** 2 * (1. - np.exp(-r_area ** 2 / (2 * sigma ** 2)))
    
    # for target_pop in population_list:
    #     print(target_pop)
    #     print([Intra_areal[target_pop][source_pop] * average_connection(1)  for source_pop in population_list])
        
    def get_indegree(target_pop,source_pop,area_surface):
        C = Intra_areal[target_pop][source_pop] * average_connection(area_surface) 
        if area_surface < 100.:  # Limit to choose between np.log and log_approx
            K = int(round(np.log(1.0 - C) / np.log(1. - 1. / (num_V1[target_pop][
                'neurons'] * num_V1[source_pop]['neurons'] * area_surface ** 2)))) / (
                    num_V1[target_pop]['neurons'] * area_surface)
        else:
            K = int(round(log_approx(C, 20) / log_approx(1. / (num_V1[target_pop][
                'neurons'] * num_V1[source_pop]['neurons'] * area_surface ** 2), 20))) / (
                    num_V1[target_pop]['neurons'] * area_surface)
        return K  
     
    Indegree_prime_fullscale = nested_dict()
    for area, target_pop, source_pop in product(area_list, population_list, population_list):
        Indegree_prime_fullscale[area][target_pop][source_pop] = get_indegree(target_pop,source_pop,Area_surfaces[area])

    # Assign the average intrinsic FLN to each area
    mean_Intrinsic_FLN = Intrinsic_FLN_Data['mean']['mean']
    mean_Intrinsic_error = Intrinsic_FLN_Data['mean']['error']
    Intrinsic_FLN_completed_fullscale = {}
    for area in area_list:
        Intrinsic_FLN_completed_fullscale[area] = {
            'mean': mean_Intrinsic_FLN, 'error': mean_Intrinsic_error}

    """
    Determine mean connection probability, indegrees and intrinsic FLN
    for areas with 1mm2 surface area.
    """

    Indegree_prime = nested_dict()
    for area, target_pop, source_pop in product(area_list, population_list, population_list):
        Indegree_prime[area][target_pop][source_pop] = get_indegree(target_pop,source_pop,1)

    Indegree_prime = Indegree_prime.to_dict()
    # print(Indegree_prime["V1"])
    
    Intrinsic_FLN_completed = {}
    mean_Intrinsic_FLN = Intrinsic_FLN_Data['mean']['mean']
    mean_Intrinsic_error = Intrinsic_FLN_Data['mean']['error']

    for area in area_list:
        average_relation_indegrees = []
        for pop in population_list:
            for pop2 in population_list:
                if Indegree_prime_fullscale[area][pop][pop2] > 0.:
                    average_relation_indegrees.append(Indegree_prime[
                        area][pop][pop2] / Indegree_prime_fullscale[area][pop][pop2])
        Intrinsic_FLN_completed[area] = {'mean': mean_Intrinsic_FLN * np.mean(
            average_relation_indegrees), 'error': mean_Intrinsic_error}

    """
    2. Compute the conversion factors between microcircuit
    and multi-area model areas (c_A(R)) for down-scaled and fullscale areas.
    """

    conversion_factor = {}
    for area in area_list:
        Nsyn_int_prime = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                Nsyn_int_prime += Indegree_prime[area][target_pop][
                    source_pop] * neuronal_numbers[area][target_pop]
        conversion_factor[area] = Intrinsic_FLN_completed[area][
            'mean'] * rho_syn[area] * surface * total_thicknesses[area] / Nsyn_int_prime

    conversion_factor_fullscale = {}
    for area in area_list:
        Nsyn_int_prime = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                Nsyn_int_prime += Indegree_prime_fullscale[area][target_pop][
                    source_pop] * neuronal_numbers_fullscale[area][target_pop]
        conversion_factor_fullscale[area] = Intrinsic_FLN_completed_fullscale[area][
            'mean'] * rho_syn[area] * Area_surfaces[area] * total_thicknesses[area] / Nsyn_int_prime

    def num_IA_synapses(area,  target_pop, source_pop, area_model='micro'):
        """
        Computes the number of intrinsic synapses from target population
        to source population in an area.

        Parameters
        ----------
        area : str
            Area for which to compute connectivity.
        target_pop : str
            Target population of the connection
        source_pop : str
            Source population of the connection
        area_model : str
            Whether to compute the number of synapses
            for the area with realistic surface area
            ('real') or 1mm2 surface area ('micro')
            Defaults to 'micro'.

        Returns
        -------
        Nsyn : float
            Number of synapses
        """
        if area_model == 'micro':
            c_area = conversion_factor[area]
            In_degree = Indegree_prime[area][
                target_pop][source_pop]
            num_source = neuronal_numbers[area][source_pop]
            num_target = neuronal_numbers[area][target_pop]
        if area_model == 'real':
            c_area = conversion_factor_fullscale[area]
            In_degree = Indegree_prime_fullscale[area][
                target_pop][source_pop]
            num_source = neuronal_numbers_fullscale[area][source_pop]
            num_target = neuronal_numbers_fullscale[area][target_pop]

        if num_source == 0 or num_target == 0:
            Nsyn = 0
        else:
            Nsyn = c_area * In_degree * num_target
        return Nsyn

    """
    3. Compute number of intrinsic (type I) synapses
    """
    synapse_numbers = nested_dict()
    for area, target_pop, source_pop in product(
            area_list, population_list, population_list):
        N_syn = num_IA_synapses(area, target_pop, source_pop)
        synapse_numbers[area][target_pop][area][source_pop] = N_syn

    # Create dictionary with total number of type I synapses for each area
    synapses_type_I = {}
    for area in area_list:
        N_syn_i = 0.0
        for target_pop in population_list:
            for source_pop in population_list:
                N_syn_i += num_IA_synapses(area, source_pop, target_pop)
        synapses_type_I[area] = N_syn_i

    """
    4. Compute number of type II synapses
    """
    synapses_type_II = {}
    s = 0.0
    for target_area in area_list:
        s_area = 0.0
        for target_pop in population_list:
            syn = 0.0
            if neuronal_numbers[target_area][target_pop] != 0.0:
                for source_pop in population_list:
                    micro_in_degree = num_IA_synapses(target_area,target_pop, source_pop) / neuronal_numbers[target_area][target_pop]
                    real_in_degree = (num_IA_synapses(target_area, target_pop, source_pop,area_model='real') / neuronal_numbers_fullscale[target_area][target_pop])
                    syn += (real_in_degree - micro_in_degree) * neuronal_numbers[target_area][target_pop]
            s_area += syn
        synapses_type_II[target_area] = s_area

    """
    Cortico-cortical synapses
    ------------------
    1. Normalize FLN values of cortico-cortical connection
       to (1 - FLN_i - 0.013).
       1.3%: subcortical inputs, data from Markov et al. (2011)
    """
    FLN_completed = {}
    for target_area in FLN_EDR_completed:
        FLN_completed[target_area] = {}
        cc_proportion = (1.-Intrinsic_FLN_completed_fullscale[target_area]['mean']-0.013)
        norm_factor = cc_proportion / sum(FLN_EDR_completed[target_area].values())
        for source_area in FLN_EDR_completed[target_area]:
            FLN_completed[target_area][source_area] = norm_factor * FLN_EDR_completed[target_area][source_area]

    """
    2. Process Binzegger data
       The notation follows Eqs. (11-12 and following) in
       Schmidt et al. (2018):
       v : layer of cortico-cortical synapse
       cb : cell type
       cell_layer : layer of the cell
       i : population in the model
    """

    # # Determine the relative numbers of the 8 populations in Binzegger's data
    # relative_numbers_binzegger = {'23E': 0.0, '23I': 0.0,
    #                               '4E': 0.0, '4I': 0.0,
    #                               '5E': 0.0, '5I': 0.0,
    #                               '6E': 0.0, '6I': 0.0}
    # s = 0.0
    # for cb in binzegger_data:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     if cell_layer not in ['', '1']:
    #         s += binzegger_data[cb]['occurrence']

    # for cb in binzegger_data:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     if cell_layer not in ['', '1']:
    #         if cb in binzegger_E_pops:
    #             relative_numbers_binzegger[
    #                 cell_layer + 'E'] += binzegger_data[cb]['occurrence'] / s
    #         if cb in binzegger_I_pops:
    #             relative_numbers_binzegger[
    #                 cell_layer + 'I'] += binzegger_data[cb]['occurrence'] / s

    # # Determine the relative numbers of the 8 populations in V1
    # relative_numbers_model = {'23E': 0.0, '23I': 0.0,
    #                           '4E': 0.0, '4I': 0.0,
    #                           '5E': 0.0, '5I': 0.0,
    #                           '6E': 0.0, '6I': 0.0}

    # for pop in neuronal_numbers['V1']:
    #     relative_numbers_model[pop] = neuronal_numbers[
    #         'V1'][pop] / neuronal_numbers['V1']['total']

    # # Compute number of CC synapses formed in each layer
    # num_cc_synapses = {'1': 0.0, '23': 0.0, '4': 0.0, '5': 0.0, '6': 0.0}
    # for cb in binzegger_data:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     if cb in binzegger_E_pops:
    #         i = cell_layer + 'E'
    #     if cb in binzegger_I_pops:
    #         i = cell_layer + 'I'
    #     if i != '1I':
    #         for v in binzegger_data[cb]['syn_dict']:
    #             if v in num_cc_synapses:
    #                 num_ratio = relative_numbers_model[i] / relative_numbers_binzegger[i]
    #                 print("num_ratio=",num_ratio)
    #                 cc_syn_num = (binzegger_data[cb]['syn_dict'][v]['corticocortical'] / 100.0 *
    #                               binzegger_data[cb]['syn_dict'][v][
    #                                   'number of synapses per neuron'] *
    #                               binzegger_data[cb]['occurrence'] / 100.0 * num_ratio)

    #                 num_cc_synapses[v] += cc_syn_num
    
    # # Compute cond. probability
    # synapse_to_cell_body_basis = {}
    # for cb in binzegger_data:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     if cb in binzegger_E_pops:
    #         i = cell_layer + 'E'
    #     else:
    #         i = cell_layer + 'I'
            
    #     for v in binzegger_data[cb]['syn_dict']:
    #         if v in num_cc_synapses:
    #             if i != '1I':  # We do not model cell types in layer 1
    #                 num_ratio = relative_numbers_model[i] / relative_numbers_binzegger[i]
    #                 value = (binzegger_data[cb]['syn_dict'][v]['corticocortical'] / 100.0 *
    #                          binzegger_data[cb]['syn_dict'][v]['number of synapses per neuron'] *
    #                          binzegger_data[cb]['occurrence'] / 100.0 * num_ratio)
    #                 if num_cc_synapses[v] >0:
    #                     cond_prob = value/num_cc_synapses[v]
    #                 else:
    #                     cond_prob = 0                    
    #                 # cond_prob = value / num_cc_synapses[v]
    #                 if v in synapse_to_cell_body_basis:
    #                     if i in synapse_to_cell_body_basis[v]:
    #                         synapse_to_cell_body_basis[
    #                             v][i] += cond_prob
    #                     else:
    #                         synapse_to_cell_body_basis[
    #                             v].update({i: cond_prob})
    #                 else:
    #                     synapse_to_cell_body_basis.update(
    #                         {v: {i: cond_prob}})

    # print("synapse=",synapse_to_cell_body_basis)
    # # Make synapse_to_cell_body area-specific to account for
    # # missing layers in some areas (area TH)
    # synapse_to_cell_body = {}
    # for area in area_list:
    #     synapse_to_cell_body[area] = deepcopy(synapse_to_cell_body_basis)



    # # Determine the relative numbers of the 8 populations in Binzegger's data
    # relative_numbers_binzegger = {'23E': 0.0, '23I': 0.0,
    #                               '4E': 0.0, '4I': 0.0,
    #                               '5E': 0.0, '5I': 0.0,
    #                               '6E': 0.0, '6I': 0.0}
    # s = 0.0
    # for cb in binzegger_data:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     if cell_layer not in ['', '1']:
    #         s += binzegger_data[cb]['occurrence']

    # for cb in binzegger_data:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     if cell_layer not in ['', '1']:
    #         if cb in binzegger_E_pops:
    #             relative_numbers_binzegger[
    #                 cell_layer + 'E'] += binzegger_data[cb]['occurrence'] / s
    #         if cb in binzegger_I_pops:
    #             relative_numbers_binzegger[
    #                 cell_layer + 'I'] += binzegger_data[cb]['occurrence'] / s

    # # Determine the relative numbers of the 8 populations in V1
    # relative_numbers_model = {'23E': 0.0, '23I': 0.0,
    #                           '4E': 0.0, '4I': 0.0,
    #                           '5E': 0.0, '5I': 0.0,
    #                           '6E': 0.0, '6I': 0.0}

    # for pop in neuronal_numbers['V1']:
    #     relative_numbers_model[pop] = neuronal_numbers[
    #         'V1'][pop] / neuronal_numbers['V1']['total']
    

    # Process Binzegger data into conditional probabilities: What is the
    # probability of having a cell body in layer u if a cortico-cortical
    # connection forms a synapse in layer v ?

    # Compute number of CC synapses formed in each layer
    synapse_cc = {}
    for cb in binzegger_data:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        for v in binzegger_data[cb]['syn_dict']:
                value = (binzegger_data[cb]['syn_dict'][v]['corticocortical'] / 100.0 *
                         binzegger_data[cb]['syn_dict'][v]['number of synapses per neuron'] *
                         binzegger_data[cb]['occurrence'] / 100.0 )
                if v in synapse_cc:
                    synapse_cc[v] += value
                else:
                    synapse_cc.update({v: value})
    
    # print("synapse_cc:",synapse_cc)

    # Compute prob of synapses from a layer to cell body
    synapse_to_cell_body_basis = {}
    for cb in binzegger_data:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        
        for v in binzegger_data[cb]['syn_dict']:
            # print(cb,":",v)
            value = (binzegger_data[cb]['syn_dict'][v]['corticocortical'] / 100.0 *
                     binzegger_data[cb]['syn_dict'][v]['number of synapses per neuron'] *
                     binzegger_data[cb]['occurrence'] / 100.0 )
            if synapse_cc[v] >0:
                cond_prob = value/synapse_cc[v]
            else:
                cond_prob = 0
            if v in synapse_to_cell_body_basis:
                if cb in synapse_to_cell_body_basis[v]:
                    synapse_to_cell_body_basis[v][cb] += cond_prob
                else:
                    synapse_to_cell_body_basis[v].update({cb:cond_prob})
            else:
                synapse_to_cell_body_basis.update(
                    {v: {cb: cond_prob}})
    
    #transform cell type
    # transform_cell_type = {}
    
    # for cb in binzegger_data:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    
    # for cb in binzegger_E_pops:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     transform_cell_type.update({cb:{"E"+cell_layer:1.}})
    # for cb in binzegger_b_pops:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     transform_cell_type.update({cb:{"P"+cell_layer:1.}})
    # for cb in binzegger_nb_pops:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    #     transform_cell_type.update({cb:{"H"+cell_layer:1.}})
        
    transform_cell_type = {}
    
    # for cb in binzegger_data:
    #     cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
    
    for cb in binzegger_E_pops:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        # print("cell_layer=",cell_layer)
        if "E"+cell_layer not in transform_cell_type:
            transform_cell_type.update({("E"+cell_layer):{cb:1.}})
        else:
            transform_cell_type["E"+cell_layer].update({cb:1.})
    for cb in binzegger_b_pops:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        if "P"+cell_layer not in transform_cell_type:
            transform_cell_type.update({("P"+cell_layer):{cb:1.}})
        else:
            transform_cell_type["P"+cell_layer].update({cb:1.})
    for cb in binzegger_nb_pops:
        cell_layer = re.sub("\D", "", re.sub("\(.*\)", "", cb))
        if cell_layer == '1':
            if "H"+cell_layer not in transform_cell_type:
                transform_cell_type.update({("H"+cell_layer):{cb:1.}})
            else:
                transform_cell_type[("H"+cell_layer)].update({cb:1.})     
        else:
            # print("num=",neuronal_numbers_fullscale['V1'])
            # print("cell_layer=",cell_layer)
            if "V"+cell_layer not in transform_cell_type:
                transform_cell_type.update({("V"+cell_layer):{cb: float(neuronal_numbers_fullscale['V1'][("V"+cell_layer)]) / float(neuronal_numbers_fullscale['V1'][("V"+cell_layer)]+neuronal_numbers_fullscale['V1'][("S"+cell_layer)])}})
            else:
                transform_cell_type[("V"+cell_layer)].update({cb: float(neuronal_numbers_fullscale['V1'][("V"+cell_layer)]) / float(neuronal_numbers_fullscale['V1'][("V"+cell_layer)]+neuronal_numbers_fullscale['V1'][("S"+cell_layer)])})     
            if "S"+cell_layer not in transform_cell_type:
                transform_cell_type.update({("S"+cell_layer):{cb: float(neuronal_numbers_fullscale['V1'][("S"+cell_layer)]) / float(neuronal_numbers_fullscale['V1'][("V"+cell_layer)]+neuronal_numbers_fullscale['V1'][("S"+cell_layer)])}})
            else:
                transform_cell_type[("S"+cell_layer)].update({cb: float(neuronal_numbers_fullscale['V1'][("S"+cell_layer)]) / float(neuronal_numbers_fullscale['V1'][("V"+cell_layer)]+neuronal_numbers_fullscale['V1'][("S"+cell_layer)])})     

            # print(transform_cell_type[("S"+cell_layer)][cb]+transform_cell_type[("V"+cell_layer)][cb])

    # print("tranform_cell_type=",transform_cell_type)
    # # transform_cell_type.update({"nb1":{"H1":1.}})
    # # transform_cell_type.update({"p2/3":{"E23":1.}})
    # # transform_cell_type.update({"b2/3":{"P23":1.}})
    # # transform_cell_type.update({"nb2/3":{"H23":}})
    
    
    # # print(synapse_to_cell_body_basis)
    
    # synpase_to_neuron = {}
    # for v in synapse_to_cell_body_basis:
    #     for cb in synapse_to_cell_body_basis[v]:
    #         binzegger_data[cb]['occurrence']
    #         synpase_to_neuron.update({v:{transform_cell_type[cb]}})
    
    
    synapse_to_cell_body = {}
    for area in area_list:
        synapse_to_cell_body[area] = deepcopy(synapse_to_cell_body_basis)
        # for layer in synapse_to_cell_body[area]:    
        #     for pop in population_list:
        #         if pop[0] in ['E','H']:
        #             if pop [1] == ["4"]:
        #             synapse_to_cell_body[area][layer][pop] =deepcopy(synapse_to_cell_body)


    for layer in synapse_to_cell_body['TH']:        
        for neu_type in ["E","S","V","P"]:
            l = 0.
            for layer0 in ['23', '5', '6']:
                l += laminar_thicknesses['TH'][layer0]
            for layer0 in ['23', '5', '6']:
                pop = neu_type + layer0
                if neu_type+'4' in synapse_to_cell_body['TH'][layer]:
                    if pop in synapse_to_cell_body['TH'][layer]:
                        synapse_to_cell_body['TH'][layer][pop] += synapse_to_cell_body[
                            'TH'][layer][neu_type+'4'] * laminar_thicknesses['TH'][layer0] / l
                    else:
                        synapse_to_cell_body['TH'][layer][pop] = synapse_to_cell_body[
                            'TH'][layer][neu_type+'4'] * laminar_thicknesses['TH'][layer0] / l

    for layer in synapse_to_cell_body['TH']:
        for neu_type in ["E","S","V","P"]:
            if neu_type+'4' in synapse_to_cell_body['TH'][layer]:
                del synapse_to_cell_body['TH'][layer][neu_type+'4']

    def prob_CC_synapses(target_area, syn_layer, source_area, source_pop):
        """
        Compute connection prob of synapses from populations to layers in different areas

        Parameters
        ----------
        target_area : str
            Target area of the connection
        syn_laer : str
            Target layer of the connection
        source_area : str
            Source area of the connection
        source_pop : str
            Source population of the connection

        Returns
        -------
        Nsyn : float
            Number of synapses of the connection.
        """

        prob = 0.0

        # Test if the connection exists.
        if (source_area in Coco_Data[target_area] and
           source_pop in ["E23","E5","E6"] and
           neuronal_numbers[target_area][target_pop] != 0):
            
            num_source = neuronal_numbers_fullscale[source_area][source_pop]

            # information on the area level
            FLN_BA = FLN_completed[target_area][source_area]

            # source side
            # if there is laminar information in CoCoMac, use it
            if Coco_Data[target_area][source_area]['source_pattern'] is not None:
                sp = np.array(Coco_Data[target_area][source_area][
                              'source_pattern'], dtype=float)

                # Manually determine SLN, based on CoCoMac:
                # from supragranular, then SLN=0.,
                # no connections from infragranular --> SLN=1.
                if np.all(sp[:3] == 0):
                    SLN_value = 0.
                elif np.all(sp[-2:] == 0):
                    SLN_value = 1.
                else:
                    SLN_value = SLN_Data[target_area][source_area]

                if source_pop in origin_patterns['S']:
                    if np.any(sp[:3] != 0):
                        X = SLN_value
                        Y = 1.  # Only layer 2/3 is part of the supragranular pattern
                    else:
                        X = 0.
                        Y = 0.

                elif source_pop in origin_patterns['I']:
                    if np.any(sp[-2:] != 0):
                        # Distribute between 5 and 6 according to CocoMac values
                        index = list(range(1, 7)).index(int(source_pop[1:]))
                        if sp[index] != 0:
                            X = 1. - SLN_value
                            Y = 10 ** (sp[index]) / np.sum(10 **
                                                           sp[-2:][np.where(sp[-2:] != 0)])
                        else:
                            X = 0.
                            Y = 0.
                    else:
                        X = 0.
                        Y = 0.
            # otherwise, use neuronal numbers
            else:
                if source_pop in origin_patterns['S']:
                    X = SLN_Data[target_area][source_area]
                    Y = 1.0  # Only layer 2/3 is part of the supragranular pattern

                elif source_pop in origin_patterns['I']:
                    X = 1.0 - SLN_Data[target_area][source_area]
                    infra_neurons = 0.0
                    for i in origin_patterns['I']:
                        infra_neurons += neuronal_numbers_fullscale[
                            source_area][i]
                    Y = num_source / infra_neurons

            # target side
            # if there is laminar data in CoCoMac, use this
            if Coco_Data[target_area][source_area]['target_pattern'] is not None:
                tp = np.array(Coco_Data[target_area][source_area][
                              'target_pattern'], dtype=float)

                # If there is a '?' (=-1) in the data, check if this layer is in
                # the termination pattern induced by hierarchy and insert a 2 if
                # yes
                if -1 in tp:
                    if (SLN_Data[target_area][source_area] > 0.35 and
                            SLN_Data[target_area][source_area] <= 0.65):
                        T_hierarchy = termination_layers2['C']
                    elif SLN_Data[target_area][source_area] < 0.35:
                        T_hierarchy = termination_layers2['M']
                    elif SLN_Data[target_area][source_area] > 0.65:
                        T_hierarchy = termination_layers2['F']
                    for l in T_hierarchy:
                        if tp[l - 1] == -1:
                            tp[l - 1] = 2
                T = np.where(tp > 0.)[0] + 1  # '+1' transforms indices to layers
                # Here we treat the values as numbers of labeled neurons rather
                # than densities for the sake of simplicity
                p_T = np.sum(10 ** tp[np.where(tp > 0.)[0]])
                Nsyn = 0.0
                su = 0.
                for i in range(len(T)):
                    if T[i] in [2, 3]:
                        syn_layer = '23'
                    else:
                        syn_layer = str(T[i])
                    Z = 10 ** tp[np.where(tp > 0.)[0]][i] / p_T
                    prob = FLN_BA * X * Y * Z

                    su += Z

            # otherwise use laminar thicknesses
            else:
                if (SLN_Data[target_area][source_area] > 0.35 and
                        SLN_Data[target_area][source_area] <= 0.65):
                    T = termination_layers['C']
                elif SLN_Data[target_area][source_area] < 0.35:
                    T = termination_layers['M']
                elif SLN_Data[target_area][source_area] > 0.65:
                    T = termination_layers['F']

                p_T = 0.0
                for i in T:
                    if i != '1':
                        p_T += laminar_thicknesses[target_area][i]

                prob = 0.0

                if syn_layer == '1':
                    Z = 0.5
                else:
                    if '1' in T:
                        Z = 0.5 * \
                            laminar_thicknesses[
                                target_area][syn_layer] / p_T
                    else:
                        Z = laminar_thicknesses[
                            target_area][syn_layer] / p_T
                prob = FLN_BA * X * Y * Z

        # print("source_pop=",source_pop)
        # print("prob=",prob)
        return prob
    
    
    def num_CC_synapses(target_area, target_pop, source_area, source_pop):
        """
        Compute number of synapses between two populations in different areas

        Parameters
        ----------
        target_area : str
            Target area of the connection
        target_pop : str
            Target population of the connection
        source_area : str
            Source area of the connection
        source_pop : str
            Source population of the connection

        Returns
        -------
        Nsyn : float
            Number of synapses of the connection.
        """
        N_syn = 0.0
        Nsyn_tot = rho_syn[target_area] * \
                    net_params['surface'] * total_thicknesses[target_area]
        
        # print("Nsyn_tot=",Nsyn_tot)
        # print("rho=",rho_syn[target_area])
        # print("surface=",net_params['surface'])
        # print("thick=",total_thicknesses[target_area])
        
        for syn_layer in synapse_to_cell_body[target_area]:
            # synapse_to_cell_body[target_area][syn_layer][target_pop]
            # print("prob_cc=",prob_CC_synapses(target_area, syn_layer, source_area, source_pop))
            if target_pop in transform_cell_type:
                for cb in transform_cell_type[target_pop]:
                    # print("target_pop=",target_pop)
                    # print(synapse_to_cell_body[target_area][syn_layer]['nb1'])
                    if cb in synapse_to_cell_body[target_area][syn_layer]:
                    # synapse_to_cell_body[target_area][syn_layer][str(cb)] 
                        # N_syn += prob_CC_synapses(target_area, syn_layer, source_area, source_pop)*Nsyn_tot*synapse_to_cell_body[target_area][syn_layer][cb]
                        N_syn += prob_CC_synapses(target_area, syn_layer, source_area, source_pop)*Nsyn_tot*synapse_to_cell_body[target_area][syn_layer][cb]*transform_cell_type[target_pop][cb]
                        # if prob_CC_synapses(target_area, syn_layer, source_area, source_pop) > 0 :
                            # print("target_area=",target_area)
                            # print("source_area=",source_area)
                            # print("syn_layer=",syn_layer)
                            # print("souce_pop=",source_pop)
                            # print("prob=",prob_CC_synapses(target_area, syn_layer, source_area, source_pop))
        return N_syn   

    # def num_CC_synapses(target_area, target_pop, source_area, source_pop):
    #     """
    #     Compute number of synapses between two populations in different areas

    #     Parameters
    #     ----------
    #     target_area : str
    #         Target area of the connection
    #     target_pop : str
    #         Target population of the connection
    #     source_area : str
    #         Source area of the connection
    #     source_pop : str
    #         Source population of the connection

    #     Returns
    #     -------
    #     Nsyn : float
    #         Number of synapses of the connection.
    #     """

    #     Nsyn = 0.0

    #     # Test if the connection exists.
    #     if (source_area in Coco_Data[target_area] and
    #         source_pop in ["E23","E5","E6"] and
    #         neuronal_numbers[target_area][target_pop] != 0):
    #         # print(source_pop+":"+source_pop)

    #         num_source = neuronal_numbers_fullscale[source_area][source_pop]
    #         # print(num_source)

    #         # information on the area level
    #         FLN_BA = FLN_completed[target_area][source_area]
    #         Nsyn_tot = rho_syn[target_area] * \
    #             Area_surfaces[target_area] * total_thicknesses[target_area]

    #         # source side
    #         # if there is laminar information in CoCoMac, use it
    #         if Coco_Data[target_area][source_area]['source_pattern'] is not None:
    #             sp = np.array(Coco_Data[target_area][source_area][
    #                           'source_pattern'], dtype=float)

    #             # Manually determine SLN, based on CoCoMac:
    #             # from supragranular, then SLN=0.,
    #             # no connections from infragranular --> SLN=1.
    #             if np.all(sp[:3] == 0):
    #                 SLN_value = 0.
    #             elif np.all(sp[-2:] == 0):
    #                 SLN_value = 1.
    #             else:
    #                 SLN_value = SLN_Data[target_area][source_area]

    #             if source_pop in origin_patterns['S']:
    #                 if np.any(sp[:3] != 0):
    #                     X = SLN_value
    #                     Y = 1.  # Only layer 2/3 is part of the supragranular pattern
    #                 else:
    #                     X = 0.
    #                     Y = 0.

    #             elif source_pop in origin_patterns['I']:
    #                 if np.any(sp[-2:] != 0):
    #                     # Distribute between 5 and 6 according to CocoMac values
    #                     index = list(range(1, 7)).index(int(source_pop[1:]))
    #                     if sp[index] != 0:
    #                         X = 1. - SLN_value
    #                         Y = 10 ** (sp[index]) / np.sum(10 **
    #                                                        sp[-2:][np.where(sp[-2:] != 0)])
    #                     else:
    #                         X = 0.
    #                         Y = 0.
    #                 else:
    #                     X = 0.
    #                     Y = 0.
    #         # otherwise, use neuronal numbers
    #         else:
    #             if source_pop in origin_patterns['S']:
    #                 X = SLN_Data[target_area][source_area]
    #                 Y = 1.0  # Only layer 2/3 is part of the supragranular pattern

    #             elif source_pop in origin_patterns['I']:
    #                 X = 1.0 - SLN_Data[target_area][source_area]
    #                 infra_neurons = 0.0
    #                 for i in origin_patterns['I']:
    #                     infra_neurons += neuronal_numbers_fullscale[
    #                         source_area][i]
    #                 Y = num_source / infra_neurons

    #         # target side
    #         # if there is laminar data in CoCoMac, use this
    #         if Coco_Data[target_area][source_area]['target_pattern'] is not None:
    #             tp = np.array(Coco_Data[target_area][source_area][
    #                           'target_pattern'], dtype=float)

    #             # If there is a '?' (=-1) in the data, check if this layer is in
    #             # the termination pattern induced by hierarchy and insert a 2 if
    #             # yes
    #             if -1 in tp:
    #                 if (SLN_Data[target_area][source_area] > 0.35 and
    #                         SLN_Data[target_area][source_area] <= 0.65):
    #                     T_hierarchy = termination_layers2['C']
    #                 elif SLN_Data[target_area][source_area] < 0.35:
    #                     T_hierarchy = termination_layers2['M']
    #                 elif SLN_Data[target_area][source_area] > 0.65:
    #                     T_hierarchy = termination_layers2['F']
    #                 for l in T_hierarchy:
    #                     if tp[l - 1] == -1:
    #                         tp[l - 1] = 2
    #             T = np.where(tp > 0.)[0] + 1  # '+1' transforms indices to layers
    #             # Here we treat the values as numbers of labeled neurons rather
    #             # than densities for the sake of simplicity
    #             p_T = np.sum(10 ** tp[np.where(tp > 0.)[0]])
    #             Nsyn = 0.0
    #             su = 0.
    #             for i in range(len(T)):
    #                 if T[i] in [2, 3]:
    #                     syn_layer = '23'
    #                 else:
    #                     syn_layer = str(T[i])
    #                 Z = 10 ** tp[np.where(tp > 0.)[0]][i] / p_T
    #                 if target_pop in synapse_to_cell_body[target_area][syn_layer]:
    #                     print("target_pop=",target_pop)
    #                     Nsyn += synapse_to_cell_body[target_area][syn_layer][
    #                         target_pop] * Nsyn_tot * FLN_BA * X * Y * Z

    #                 su += Z

    #         # otherwise use laminar thicknesses
    #         else:
    #             if (SLN_Data[target_area][source_area] > 0.35 and
    #                     SLN_Data[target_area][source_area] <= 0.65):
    #                 T = termination_layers['C']
    #             elif SLN_Data[target_area][source_area] < 0.35:
    #                 T = termination_layers['M']
    #             elif SLN_Data[target_area][source_area] > 0.65:
    #                 T = termination_layers['F']

    #             p_T = 0.0
    #             for i in T:
    #                 if i != '1':
    #                     p_T += laminar_thicknesses[target_area][i]

    #             Nsyn = 0.0
    #             for syn_layer in T:
    #                 if target_pop in synapse_to_cell_body[target_area][syn_layer]:
    #                     if syn_layer == '1':
    #                         Z = 0.5
    #                     else:
    #                         if '1' in T:
    #                             Z = 0.5 * \
    #                                 laminar_thicknesses[
    #                                     target_area][syn_layer] / p_T
    #                         else:
    #                             Z = laminar_thicknesses[
    #                                 target_area][syn_layer] / p_T
    #                     Nsyn += synapse_to_cell_body[target_area][syn_layer][
    #                         target_pop] * Nsyn_tot * FLN_BA * X * Y * Z

    #     return Nsyn

    """
    Compute the number of cortico-cortical synapses
    for each pair of populations.
    """
    # area TH does not have a granular layer
    # neuronal_numbers_fullscale['TH']['4E'] = 0.0
    # neuronal_numbers['TH']['4E'] = 0.0
    # neuronal_numbers_fullscale['TH']['4I'] = 0.0
    # neuronal_numbers['TH']['4I'] = 0.0

    for target_area, target_pop, source_area, source_pop in product(area_list, population_list,
                                                                    area_list, population_list):
        if target_area != source_area:
            N_fullscale = neuronal_numbers_fullscale[target_area][target_pop]
            N = neuronal_numbers[target_area][target_pop]
            if N != 0:
                N_syn = num_CC_synapses(target_area, target_pop,
                                        source_area, source_pop) / N_fullscale * N
            else:
                N_syn = 0.0
                
            # print("target_area=",target_area)
            # print("souce_area=",source_area)
            # print("N_syn=",N_syn)
            synapse_numbers[target_area][target_pop][source_area][source_pop] = N_syn

    synapse_numbers = synapse_numbers.to_dict()

    """
    If switch_E_specificity is True, redistribute
    the synapses of feedback connections to achieve
    the E_specific_factor of 0.93
    """
    if E_specificity:
        E_specific_factor = 0.93
        for target_area in area_list:
            for source_area in area_list:
                if (target_area != source_area and source_area in Coco_Data[target_area] and
                        SLN_Data[target_area][source_area] < 0.35):
                    syn_I = 0.0
                    syn_E = 0.0
                    for target_pop in synapse_numbers[target_area]:
                        for source_pop in synapse_numbers[target_area][target_pop][source_area]:
                            if target_pop.find('E') > -1:
                                syn_E += synapse_numbers[target_area][
                                    target_pop][source_area][source_pop]
                            else:
                                syn_I += synapse_numbers[target_area][
                                    target_pop][source_area][source_pop]
                    if syn_E > 0.0 or syn_I > 0.0:
                        alpha_E = syn_E / (syn_E + syn_I)
                        alpha_I = syn_I / (syn_E + syn_I)
                        if alpha_I != 0.0 and alpha_E != 0.0:
                            for target_pop in synapse_numbers[target_area]:
                                for source_pop in synapse_numbers[target_area][
                                        target_pop][source_area]:
                                    N_syn = synapse_numbers[target_area][target_pop][
                                        source_area][source_pop]
                                    if target_pop.find('E') > -1:
                                        synapse_numbers[target_area][target_pop][source_area][
                                            source_pop] = E_specific_factor / alpha_E * N_syn
                                    else:
                                        synapse_numbers[target_area][target_pop][source_area][
                                            source_pop] = (1. - E_specific_factor) / alpha_I * N_syn

    """
    External inputs
    ---------------
    To determine the number of external inputs to each
    population, we compute the total number of external
    to an area and then distribute the synapses such that
    each population receives the same indegree from external
    Poisson sources.


    1. Compute the total number of external synapses to each
       area as the difference between the total number of
       synapses and the intrinsic (type I) and cortico-cortical
       (type III) synapses.
    """
    External_synapses = {}
    for target_area in area_list:
        N_syn_tot = surface * total_thicknesses[target_area] * rho_syn[target_area]
        CC_synapses = 0.0
        for target_pop, source_area, source_pop in product(population_list, area_list,
                                                           population_list):
            if source_area != target_area:
                CC_synapses += synapse_numbers[target_area][target_pop][source_area][source_pop]
        ext_syn = N_syn_tot * (1. - Intrinsic_FLN_completed[target_area]['mean']) - CC_synapses
        # print("ext_syn=",ext_syn)
        # if ext_syn > 0:
        #     External_synapses[target_area] = ext_syn
        # else:
        #     ext_syn = 0.
        # External_synapses[target_area] = 0.
        External_synapses[target_area] = ext_syn
    # print("External_indegrees=",[External_synapses[area] /neuronal_numbers[area]['total'] for area in area_list])

    """
    2. Distribute poisson sources among populations such that each
       population receives the same Poisson indegree.
       For this, we construct a system of linear equations and solve
       this using a least-squares algorithm (numpy.linalg.lstsq).
    """
    
    for area in area_list:
    #     nonvisual_fraction_matrix = np.zeros(
    #         (len(structure[area]) + 1, len(structure[area])))
    #     for i in range(len(structure[area])):
    #         nonvisual_fraction_matrix[
    #             i] = 1. / len(structure[area]) * np.ones(len(structure[area]))
    #         nonvisual_fraction_matrix[i][i] -= 1

    #     for i in range(len(structure[area])):
    #         nonvisual_fraction_matrix[-1][
    #             i] = neuronal_numbers[area][structure[area][i]]

    #     vector = np.zeros(len(structure[area]) + 1)
    #     ext_syn = External_synapses[area]
    #     vector[-1] = ext_syn
    #     solution, residues, rank, s = np.linalg.lstsq(
    #         nonvisual_fraction_matrix, vector, rcond=-1)
    #     # for i, pop in enumerate(structure[area]):
    #     #     synapse_numbers[area][pop]['external'] = {
    #     #         'external': solution[i] * neuronal_numbers[area][pop]}
        
    #     for i, pop in enumerate(structure[area]):
    #         synapse_numbers[area][pop]['external'] = {
    #             'external':0. * neuronal_numbers[area][pop]}
    
        #   for i, pop in enumerate(structure[area]):
        #     synapse_numbers[area][pop]['external'] = {
        #         'external':External_synapses[area] * neuronal_numbers[area][pop]/neuronal_numbers[area]['total']}
            for i, pop in enumerate(structure[area]):
                synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']) * neuronal_numbers[area][pop]}

                # if pop[0] == 'E':
                #     synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']) * neuronal_numbers[area][pop]}
                # if pop[0] == 'H':
                #     if pop == 'H1':
                #         synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']) *0.6* neuronal_numbers[area][pop]}
                #     # print(pop)
                #     else:
                #         synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']) * neuronal_numbers[area][pop]}
                # if pop[0] == 'P':
                #     if (pop == 'P23' or pop == 'P4' or pop == 'P5' ):
                #         # synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']+3000.) * neuronal_numbers[area][pop]}
                #         synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']) * neuronal_numbers[area][pop]}
                #     else:
                #         # synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']+2000.) * neuronal_numbers[area][pop]}
                #         synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']) * neuronal_numbers[area][pop]}
                #     if pop == 'P4':
                #         synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']) * neuronal_numbers[area][pop]}
                # # if (pop == 'E23' or pop == 'H23' or pop == 'S23' or pop == 'E4' or pop == 'H4' or 'S4'):
                # #     synapse_numbers[area][pop]['external']['external'] = synapse_numbers[area][pop]['external']['external'] + 1000
                # if pop[0] == 'S':
                #     synapse_numbers[area][pop]['external'] = {'external':(External_synapses[area]/neuronal_numbers[area]['total']) * neuronal_numbers[area][pop]}
    # print(synapse_numbers)    
    synapse_numbers['TH']['E4']['external'] = {'external': 0.0}
    synapse_numbers['TH']['P4']['external'] = {'external': 0.0}
    synapse_numbers['TH']['S4']['external'] = {'external': 0.0}
    synapse_numbers['TH']['V4']['external'] = {'external': 0.0}
    
    # print(synapse_numbers)
    """
    Modify external inputs according to additional factors
    """
    for target_area in area_list:
        for target_pop in synapse_numbers[target_area]:
            if target_pop in ['E23']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_23E * synapse_numbers[target_area][target_pop][
                        'external']['external']
            if target_pop in ['E4']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_4E * synapse_numbers[target_area][target_pop][
                        'external']['external']
                # print("fac_4E=",fac_nu_ext_4E)
            if target_pop in ['E5']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_5E * synapse_numbers[target_area][target_pop][
                        'external']['external']
            if target_pop in ['E6']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_6E * synapse_numbers[target_area][target_pop][
                        'external']['external']         
            if target_pop[1:] == "23":
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_23 * synapse_numbers[target_area][target_pop][
                        'external']['external']    
            if target_pop[1:] == "4":
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_4 * synapse_numbers[target_area][target_pop][
                        'external']['external']  
            if target_pop[1:] == "5":
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_5 * synapse_numbers[target_area][target_pop][
                        'external']['external']  
            if target_pop[1:] == "6":
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_6 * synapse_numbers[target_area][target_pop][
                        'external']['external']  
            
            if target_pop in ['H1']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_1H * synapse_numbers[target_area][target_pop][
                        'external']['external']                    
            if target_pop in ['V23']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_23V * synapse_numbers[target_area][target_pop][
                        'external']['external']
            if target_pop in ['V4']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_4V * synapse_numbers[target_area][target_pop][
                        'external']['external']
            if target_pop in ['V5']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_5V * synapse_numbers[target_area][target_pop][
                        'external']['external']
            if target_pop in ['V6']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_6V * synapse_numbers[target_area][target_pop][
                        'external']['external']   
            if target_pop in ['S23']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_23S * synapse_numbers[target_area][target_pop][
                        'external']['external']   
            if target_pop in ['S4']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_4S * synapse_numbers[target_area][target_pop][
                        'external']['external']   
            if target_pop in ['S5']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_5S * synapse_numbers[target_area][target_pop][
                        'external']['external']   
            if target_pop in ['S6']:
                synapse_numbers[target_area][target_pop]['external'][
                    'external'] = fac_nu_ext_6S * synapse_numbers[target_area][target_pop][
                        'external']['external']  
                     
    synapse_numbers['TH']['E23']['external']['external'] *= fac_nu_ext_TH
    synapse_numbers['TH']['P23']['external']['external'] *= fac_nu_ext_TH
    synapse_numbers['TH']['S23']['external']['external'] *= fac_nu_ext_TH
    synapse_numbers['TH']['V23']['external']['external'] *= fac_nu_ext_TH
    
    """
    Synaptic weights
    ----------------
    Create dictionaries with the mean and standard deviation
    of the synaptic weight of each connection in the network.
    Depends on the chosen neuron model.
    """

    # for current-based neurons
    PSC_e_over_PSP_e = ((C_m_E**(-1) * tau_m_E * tau_syn_E / (tau_syn_E - tau_m_E) *
                         ((tau_m_E / tau_syn_E) ** (- tau_m_E / (tau_m_E - tau_syn_E)) -
                          (tau_m_E / tau_syn_E) ** (- tau_syn_E / (tau_m_E - tau_syn_E)))) ** (-1))
    PSC_h_over_PSP_h = ((C_m_H**(-1) * tau_m_H * tau_syn_H / (tau_syn_H - tau_m_H) *
                         ((tau_m_H / tau_syn_H) ** (- tau_m_H / (tau_m_H - tau_syn_H)) -
                          (tau_m_H / tau_syn_H) ** (- tau_syn_H / (tau_m_H - tau_syn_H)))) ** (-1))
    PSC_s_over_PSP_s = ((C_m_S**(-1) * tau_m_S * tau_syn_S / (tau_syn_S - tau_m_S) *
                         ((tau_m_S / tau_syn_S) ** (- tau_m_S / (tau_m_S - tau_syn_S)) -
                          (tau_m_S / tau_syn_S) ** (- tau_syn_S / (tau_m_S - tau_syn_S)))) ** (-1))
    PSC_p_over_PSP_p = ((C_m_P**(-1) * tau_m_P * tau_syn_P / (tau_syn_P - tau_m_P) *
                     ((tau_m_P / tau_syn_P) ** (- tau_m_P / (tau_m_P - tau_syn_P)) -
                      (tau_m_P / tau_syn_P) ** (- tau_syn_P / (tau_m_P - tau_syn_P)))) ** (-1))
    PSC_v_over_PSP_v = ((C_m_V**(-1) * tau_m_V * tau_syn_V / (tau_syn_V - tau_m_V) *
                     ((tau_m_V / tau_syn_V) ** (- tau_m_V / (tau_m_V - tau_syn_V)) -
                      (tau_m_V / tau_syn_V) ** (- tau_syn_V / (tau_m_V - tau_syn_V)))) ** (-1))
    
    # print("PSC_h_over_PSP_h=",PSC_h_over_PSP_h)
    # print("PSC_s_over_PSP_s=",PSC_s_over_PSP_s)
    # print("PSC_p_over_PSP_p=",PSC_p_over_PSP_p)
    # print("PSC_v_over_PSP_v=",PSC_v_over_PSP_v)
 

    beta_norm = {
        "H1" : 1.,
        "E23" : 0.71, "P23" : 0.48, "S23" : 1., "V23" :0.9,
        "E4" : 1.66, "S4" : 0.24, "V4" : 0.46, "P4" : 0.8,
        "E5" : 0.95, "S5" : 0.48, "V5" : 1.2, "P5" :1.09,
        "E6" : 1.12, "S6" : 0.63, "V6" : 0.5, "P6" : 0.42,
    }   
    
    beta_TH = {
        "H1" : 2.2,
        "E23" : 1.9, "P23" : 0.82, "S23" : 0.98, "V23" : 1.42,
        "E4" :1., "S4" :1., "V4": 1., "P4":1.,
        "E5" : 1.2, "S5" : 0.5, "V5" : 1.36, "P5" :1.6,
        "E6" : 1.5, "S6" : 0.7, "V6" : 0.6, "P6" : 0.5,
    }   
    
    # beta_norm = {
    #     "H1" : 3.9,
    #     "E23" : 0.8, "P23" : 0.48, "S23" : 1., "V23" :0.9,
    #     "E4" : 1.66, "S4" : 0.24, "V4" : 0.44, "P4" : 0.78,
    #     "E5" : 0.91, "S5" : 0.48, "V5" : 1.1, "P5" :1.05,
    #     "E6" : 1.12, "S6" : 0.63, "V6" : 0.4, "P6" : 0.42,
    # }        
    # beta_norm = {
    #     "H1" : 3.9,
    #     "E23" : 1.3, "P23" : 0.5, "S23" : 1., "V23" :1.,
    #     "E4" : 2.1, "S4" : 0.3, "V4" : 0.4, "P4" : 0.7,
    #     "E5" : 1.35, "S5" : 0.48, "V5" : 1.6, "P5" :1.05,
    #     "E6" : 1.5, "S6" : 0.63, "V6" : 0.4, "P6" : 0.42,
    # }        
    
    # print("beta_norm=",beta_norm)    
    
    synapse_weights_mean = nested_dict()
    synapse_current_mean = nested_dict()
    # print("alpha_TH=",alpha_TH)
    # print("alpha_norm=",alpha_norm)
    #计算神经元之间的突触权重
    # if source_area == 'TH':
    #     source_pop_list = pop_list_TH
    # else:
    #     source_pop_list = pop_list_norm
    
    # if target_area == 'TH':
    #     target_pop_list = pop_list_TH
    # else:
    #     target_pop_list = pop_list_norm
    
    
    for target_area, target_pop, source_area, source_pop in product(area_list, population_list,area_list, population_list):
        if source_area == 'TH':
            if 'E' in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_e_over_PSP_e * alpha_TH[source_pop]*beta_TH[target_pop] * PSP_e
            if 'H' in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_h_over_PSP_h * g_H * alpha_TH[source_pop] * PSP_e
            if 'P' in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_p_over_PSP_p * g_P * alpha_TH[source_pop] * PSP_e
            if "S" in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_s_over_PSP_s * g_S * alpha_TH[source_pop] * PSP_e
            if "V" in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_v_over_PSP_v * g_V * alpha_TH[source_pop] * PSP_e

            #调整到H1的输入
            if  target_pop == 'H1':
                if source_pop[0] == 'E':
                    pass
                else:
                    synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.5*synapse_weights_mean[target_area][target_pop][source_area][source_pop]        


        else:
        # for target_area, target_pop, source_area, source_pop in product(area_list, population_list,area_list, population_list):
            if 'E' in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_e_over_PSP_e * alpha_norm[source_pop]*beta_norm[target_pop] * PSP_e
            if 'H' in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_h_over_PSP_h * g_H * alpha_norm[source_pop] * PSP_e
            if 'P' in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_p_over_PSP_p * g_P * alpha_norm[source_pop] * PSP_e
            if "S" in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_s_over_PSP_s * g_S * alpha_norm[source_pop] * PSP_e
            if "V" in source_pop:
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_v_over_PSP_v * g_V * alpha_norm[source_pop] * PSP_e

            #调整到H1的输入
            if  target_pop == 'H1':
                if source_pop[0] == 'E':
                    pass
                else:
                    synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.5*synapse_weights_mean[target_area][target_pop][source_area][source_pop]        


    # for target_area, target_pop, source_area, source_pop in product(area_list, population_list,area_list, population_list):
        # if area == "TH":
        #     # print("source_area=",source_area)
        #     if 'E' in source_pop:
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_e_over_PSP_e * alpha_TH[source_pop] * PSP_e
        #     if 'H' in source_pop:
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_h_over_PSP_h * g_H * alpha_TH[source_pop] * PSP_e
        #     if 'P' in source_pop:
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_p_over_PSP_p * g_P * alpha_TH[source_pop] * PSP_e
        #     if "S" in source_pop:
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_s_over_PSP_s * g_S * alpha_TH[source_pop] * PSP_e
        #     if "V" in source_pop:
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_v_over_PSP_v * g_V * alpha_TH[source_pop] * PSP_e

        #     if target_area == "CITd" :
        #         if source_area == "V4":
        #             synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.     

        #     if source_pop[0] == 'E':
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] *= beta_TH[target_pop]


        #     if source_pop == 'S23' and target_pop == 'S23':
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 2*synapse_weights_mean[target_area][target_pop][source_area][source_pop]        

        #     if source_pop == 'V23' and target_pop == 'V23':
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 2.*synapse_weights_mean[target_area][target_pop][source_area][source_pop]        

        #     if source_pop == 'S23' and target_pop == 'V23':
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.66*synapse_weights_mean[target_area][target_pop][source_area][source_pop]          

        #     if source_pop == 'P23' and target_pop == 'P23':
        #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.6*synapse_weights_mean[target_area][target_pop][source_area][source_pop]          

        #     if source_pop == 'E5':
        #         if target_pop == 'E23' or target_pop == 'E4':
        #             synapse_weights_mean[target_area][target_pop][source_area][source_pop] *= 1.2            
        # else:
            # print("source_area=",source_area)
            # if 'E' in source_pop:
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_e_over_PSP_e * alpha_norm[source_pop] * PSP_e
            # if 'H' in source_pop:
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_h_over_PSP_h * g_H * alpha_norm[source_pop] * PSP_e
            # if 'P' in source_pop:
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_p_over_PSP_p * g_P * alpha_norm[source_pop] * PSP_e
            # if "S" in source_pop:
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_s_over_PSP_s * g_S * alpha_norm[source_pop] * PSP_e
            # if "V" in source_pop:
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = PSC_v_over_PSP_v * g_V * alpha_norm[source_pop] * PSP_e

            # if target_area == "CITd" :
            #     if source_area == "V4":
            #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.     

            # if source_pop[0] == 'E':
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] *= beta_norm[target_pop]

            # #调整到H1的输入
            # if  target_pop == 'H1':
            #     if source_pop[0] == 'E':
            #         pass
            #     else:
            #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.5*synapse_weights_mean[target_area][target_pop][source_area][source_pop]        

            # if source_pop == 'S23' and target_pop == 'S23':
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 2*synapse_weights_mean[target_area][target_pop][source_area][source_pop]        

            # if source_pop == 'V23' and target_pop == 'V23':
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 2.*synapse_weights_mean[target_area][target_pop][source_area][source_pop]        

            # if source_pop == 'S23' and target_pop == 'V23':
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.66*synapse_weights_mean[target_area][target_pop][source_area][source_pop]          

            # if source_pop == 'P23' and target_pop == 'P23':
            #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.6*synapse_weights_mean[target_area][target_pop][source_area][source_pop]          

            # if source_pop == 'E5':
            #     if target_pop == 'E23' or target_pop == 'E4':
            #         synapse_weights_mean[target_area][target_pop][source_area][source_pop] *= 1.2            
             
    #给突触权重加入噪声
    synapse_weights_sd = nested_dict()
    for target_area, target_pop, source_area, source_pop in product(area_list, population_list,
                                                                    area_list, population_list):
        mean = abs(synapse_weights_mean[target_area][target_pop][source_area][source_pop])
        if ((lognormal_weights and 'E' in target_pop and 'E' in source_pop) or lognormal_weights and not lognormal_EE_only):
            sd = PSC_rel_sd_lognormal * mean
        else:
            sd = PSC_rel_sd_normal * mean
        synapse_weights_sd[target_area][target_pop][source_area][source_pop] = sd
    
        # if target_pop == 'H23':
        #     synapse_weights_mean[target_area][target_pop][source_area][source_pop] = 0.
            
    # Apply specific weight for intra_areal 4E-->23E connections
    for area in area_list:
        synapse_weights_mean[area]['E23'][area]['E4'] = PSP_e_23_4 * PSC_e_over_PSP_e
        synapse_weights_sd[area]['E23'][area]['E4'] = (PSC_rel_sd_normal * PSP_e_23_4* PSC_e_over_PSP_e)    
        
    # # Apply specific weight for intra_areal E5-->H1 connections
    # for area in area_list:
    #     synapse_weights_mean[area]['H1'][area]['E5'] = 0.0*synapse_weights_mean[area]['H1'][area]['E5']
    #     synapse_weights_sd[area]['H1'][area]['E5'] = 0.0*synapse_weights_sd[area]['H1'][area]['E5']

    # Apply cc_weights_factor for all CC connections
    for target_area, source_area in product(area_list, area_list):
        if source_area != target_area:
            for target_pop, source_pop in product(population_list, population_list):
                synapse_weights_mean[target_area][target_pop][source_area][source_pop] *= cc_weights_factor
                synapse_weights_sd[target_area][target_pop][source_area][source_pop] *= cc_weights_factor

    # Apply cc_weights_I_factor for all CC connections
    for target_area, source_area in product(area_list, area_list):
        if source_area != target_area:
            for target_pop, source_pop in product(population_list, population_list):
                if 'E' not in target_pop:
                    synapse_weights_mean[target_area][target_pop][source_area][source_pop] *= cc_weights_I_factor
                    synapse_weights_sd[target_area][target_pop][source_area][source_pop] *= cc_weights_I_factor

    # Synaptic weights for external input
    for target_area in area_list:
        for target_pop in population_list:
            synapse_weights_mean[target_area][target_pop]['external'] = {
                'external': PSC_e_over_PSP_e * PSP_ext}
    
    # print("w_ext=",PSC_e_over_PSP_e * PSP_ext)

    synapse_weights_mean = synapse_weights_mean.to_dict()
    synapse_weights_sd = synapse_weights_sd.to_dict()
    
    """
    Output section
    --------------
    All data are saved to a json file with the name structure:
    '$(prefix) + '_Data_Model' + $(out_label) + .json'.
    """

    # from treelib import Node, Tree

    # def add_nodes(tree, parent, dictionary):
    #     for key, value in dictionary.items():
    #         unique_id = f"{parent}_{key}"  # 生成唯一 ID
    #         if isinstance(value, dict):
    #             tree.create_node(key, unique_id, parent=parent)
    #             add_nodes(tree, unique_id, value)
    #         else:
    #             tree.create_node(f"{key}: {value}", f"{unique_id}_leaf", parent=parent)
    
    # tree = Tree()  
    # tree.create_node("Root", "root")
    # add_nodes(tree, "root", synapse_weights_mean)
    # tree.show()
    
    collected_data = {'area_list': area_list,
                      'av_indegree_V1': av_indegree_V1,
                      'population_list': population_list,
                      'structure': structure,
                      'synapses_orig': synapse_numbers,
                      'synapses': synapse_numbers,
                      'realistic_neuron_numbers': neuronal_numbers_fullscale,
                      'realistic_synapses': synapse_numbers,
                      'neuron_numbers': neuronal_numbers,
                      'synapses_type_I': synapses_type_I,
                      'synapses_type_II': synapses_type_II,
                      'distances': Distance_Data,
                      'binzegger_processed': synapse_to_cell_body,
                      'Intrinsic_FLN_completed': Intrinsic_FLN_completed,
                      'synapse_weights_mean': synapse_weights_mean,
                      'synapse_weights_sd': synapse_weights_sd
                      }

    # print("synapse_weights_mean=",synapse_weights_mean)
    # print("path=",os.path.join(basepath,
    #                        '.'.join(('_'.join((prefix,
    #                                            'Data_Model',
    #                                            out_label)),
    #                                  'json'))))
    # print("path=",data_path,
    #                        '.'.join(('_'.join((prefix,
    #                                            'Data_Model',
    #                                            out_label)),
    #                                  'json')))
    with open(os.path.join(data_path,
                           '.'.join(('_'.join((prefix,
                                               'Data_Model',
                                               out_label)),
                                     'json'))), 'w') as f:
        json.dump(collected_data, f)

    # print(synapse_numbers["V1"]["E23"]["V4"])

if __name__ == '__main__':
    compute_Model_params()

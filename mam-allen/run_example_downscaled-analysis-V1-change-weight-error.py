import numpy as np
import os
import time
from multiarea_model import MultiAreaModel
from config import base_path
import matplotlib.pyplot as plt
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

input_dict = {
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
            }

input_update = deepcopy(input_dict)

current_dict = {}

for i in range(10):
    print("i=",i)
    print("input_update=",input_update)
    
    # i = 0
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
                   'g': -3.,
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
                   'PSP_e_23_4': 0.15*0,
                   'PSP_e_5_h1': 0.15*0,
                   'PSP_e': 0.15*0,
                   'av_indegree_V1': 3950.}
    input_params = {'rate_ext': 40, 
                    'input_factor_E' : 0.5,
                    'poisson_input': False,
                        # rate of current
                    "rate_current" : 1.,
    
                    #Input when  rate is 10Hz
                    'input':input_dict,}
    neuron_params = {'V0_mean': -150.,
                     'V0_sd': 50.}
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

    M.analysis.load_data()
    M.analysis.create_pop_rates()
    M.analysis.create_pop_rate_dists()
    M.analysis.create_rate_time_series()

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

    for area in M.analysis.areas_loaded:
        if area == 'V1':
            if area != 'TH':
                M.analysis.multi_power_display(area,output = "png",pops = pop_list_norm,resolution=0.2)
            else:
                M.analysis.multi_power_display(area,output = "png",pops = pop_list_TH,resolution=0.2)

    for area in M.analysis.areas_loaded:
        if area == 'V1':
            M.analysis.multi_rate_display(area,pops = pop_list_norm,output = "png")
            # M.analysis.multi_voltage_display(area,pops = pop_list_norm,output = "png")
            # M.analysis.multi_current_display(area,pops = pop_list_norm,output = "png")
            # current_dict = M.analysis.avg_current_display(area,pops = pop_list_norm,t_min=500,output = "png")
        # else:
            # M.analysis.multi_rate_display(area,pops = pop_list_TH,output = "png")

    # for area in M.analysis.areas_loaded:
    #     if area == 'V1':
    #         M.analysis.single_rate_display(area,output = "png")
    #         # frac_neurons : float, [0,1]
    #         frac_neurons = 0.01
    #         M.analysis.single_dot_display(area,frac_neurons,output = "png")
    #         M.analysis.single_power_display(area,output = "png",resolution=0.2)
            
    for area in M.analysis.areas_loaded:
        if area == 'V1':
            for pop in pop_list_norm:
                print(pop)
                print(pop_list_norm)
                M.analysis.single_rate_display(area=area,pop=pop,output = "png")
                # frac_neurons : float, [0,1]
                frac_neurons = 0.01
                M.analysis.single_dot_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                # M.analysis.single_voltage_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                # M.analysis.single_current_display(area=area,pop=pop,frac_neurons=frac_neurons,output = "png")
                M.analysis.single_power_display(area=area,pop=pop,output = "png",resolution=0.2)
                
                # input_upate[pop] = input_dict[pop] - current_dict[pop]

    print(M.K_matrix.shape)
    print(M.W_matrix.shape)
    print(M.params['connection_params']['K_stable'])
    # M.analysis.save()
    # M.analysis.show_rates(output = "png")

    # for area in M.analysis.pop_rates['Parameters']['areas']:
    #     for pop in M.analysis.pop_rates[area]:
    #         if pop != 'total':
    #             # print(M.analysis.pop_rates[area][pop][0])
    #             pass

    # structure_similarity_dict = dict()
    # activivity_similarity_dict = dict()

    # print(type(M.analysis.rate_time_series_pops['V1']['H1']))

    # for area1 in M.K:
    #     structure_similarity_dict[area1] = dict()
    #     activivity_similarity_dict[area1] = dict()
    #     for area2 in M.K:
    #         # print("area1=",area1)
    #         # print("area2=",area2)
    #         rate_sumsquared_1 = 0
    #         rate_sumsquared_2 = 0
    #         rate_sumconvolution = 0

    #         indegree_sumsquared_1 = 0
    #         indegree_sumsquared_2 = 0
    #         indegree_sumconvolution = 0

    #         if area1 == 'TH' or area2  == 'TH':
    #             pop_list = pop_list_TH
    #         else:
    #             pop_list = pop_list_norm

    #         for pop in pop_list:
    #             # rate_sumsquared_1 = rate_sumsquared_1 + M.analysis.pop_rates[area1][pop][0]*M.analysis.pop_rates[area1][pop][0]
    #             rate_sumsquared_1 = rate_sumsquared_1 + np.sum(M.analysis.rate_time_series_pops[area1][pop]*M.analysis.rate_time_series_pops[area1][pop])
    #             # rate_sumsquared_2 = rate_sumsquared_2 + M.analysis.pop_rates[area2][pop][0]*M.analysis.pop_rates[area2][pop][0]
    #             rate_sumsquared_2 = rate_sumsquared_2 + np.sum(M.analysis.rate_time_series_pops[area2][pop]*M.analysis.rate_time_series_pops[area2][pop])
    #             # rate_sumconvolution = rate_sumconvolution + M.analysis.pop_rates[area1][pop][0]*M.analysis.pop_rates[area2][pop][0]
    #             rate_sumconvolution = rate_sumconvolution + np.sum(M.analysis.rate_time_series_pops[area1][pop]*M.analysis.rate_time_series_pops[area2][pop])

    #             # print(M.K[k][pop])
    #             indegree_list_1 = []
    #             indegree_list_2 = []
    #             for a in M.K[area1][pop]:
    #                 for b in M.K[area1][pop][a]:
    #                     indegree_list_1.append(M.K[area1][pop][a][b])
    #                     indegree_list_2.append(M.K[area2][pop][a][b])
    #             # print("indegree_list=",np.array(indegree_list_1))        
    #             # print("indegree_list=",np.sum(np.array(indegree_list_1)*np.array(indegree_list_1)))
    #             indegree_sumsquared_1 = indegree_sumsquared_1 + np.sum(np.array(indegree_list_1)*np.array(indegree_list_1))
    #             # print("indegree_list=",np.sum(np.array(indegree_list_1)*np.array(indegree_list_2)))
    #             indegree_sumconvolution = indegree_sumconvolution + np.sum(np.array(indegree_list_1)*np.array(indegree_list_2))
    #             # print("indegree_list=",np.sum(np.array(indegree_list_2)*np.array(indegree_list_2)))
    #             indegree_sumsquared_2 = indegree_sumsquared_2 + np.sum(np.array(indegree_list_2)*np.array(indegree_list_2))

    #         activivity_similarity = rate_sumconvolution / (np.sqrt(rate_sumsquared_1) * np.sqrt(rate_sumsquared_2))
    #         activivity_similarity_dict[area1][area2] = activivity_similarity
    #         # print("activivity_similarity=",activivity_similarity)
    #         structure_similarity = indegree_sumconvolution / (np.sqrt(indegree_sumsquared_1) * np.sqrt(indegree_sumsquared_2))
    #         structure_similarity_dict[area1][area2] = structure_similarity  
    #         # print("structure_similarity=",structure_similarity)



    # # # 创建柱状图
    # # plt.bar(list(structure_similarity.keys()), list(structure_similarity.values()))

    # # # 添加标题和轴标签
    # # plt.title('Fruit Quantity')
    # # plt.xlabel('Fruit')
    # # plt.ylabel('Quantity')

    # # # 显示图表
    # # plt.show()

    # # Extracting keys and values for direct plotting
    # # print(structure_similarity_dict['V1'])
    # keys = structure_similarity_dict['V1'].keys()
    # values1 = structure_similarity_dict['V1'].values()
    # values2 = activivity_similarity_dict['V1'].values()

    # # Creating figure and axis object
    # fig, ax = plt.subplots()

    # # Set positions for each bar
    # ind = range(len(structure_similarity_dict['V1']))  # The x locations for the groups
    # width = 0.5  # The width of the bars

    # # Plotting both sets of data
    # rects1 = ax.bar(ind, [v for v in values1], width, label='structure')
    # rects2 = ax.bar([p + width for p in ind],[v for v in values2], width, label='activivity')

    # # Add some text for labels, title, and custom x-axis tick labels, etc.
    # ax.set_ylabel('similarity')
    # ax.set_title('structure and activity similarity across different area')
    # ax.set_xticks([p + width / 2 for p in ind])
    # ax.set_xticklabels(keys)
    # ax.legend()

    # # Show the plot
    # plt.savefig("test.png")

    # M.upload_file()

    # end_time = time.time()
    # print(time.strftime("end_time:%Y-%m-%d %H:%M:%S", time.localtime(end_time)))
    # print("real_time=",end_time - start_time)
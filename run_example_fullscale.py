#! /home/liugangqiang/miniconda3/envs/pygenn/bin/python

import numpy as np
import os

from multiarea_model import MultiAreaModel
from config import base_path
import multiprocessing



def main(id=1):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id+2)

    """
    Example script showing how to simulate the multi-area model
    on a cluster.

    We choose the same configuration as in
    Fig. 3 of Schmidt et al. (2018).

    """

    """
    Full model. Needs to be simulated with sufficient
    resources, for instance on a compute cluster.
    """
    d = {}
    conn_params = {'g': -2.,
                   'K_stable': None,
                   'fac_nu_ext_TH': 1.2,
                   'fac_nu_ext_5E': 1.125,
                   'fac_nu_ext_6E': 1.41666667,
                   'av_indegree_V1': 3950.,
                   'cc_weights_factor': 1.,
                   'cc_weights_I_factor': 1.}
    input_params = {'rate_ext': 40.}
    neuron_params = {'V0_mean': -150.,
                     'V0_sd': 50.}
    network_params = {'N_scaling': 1.,
                      'K_scaling': 1.,
                      'connection_params': conn_params,
                      'input_params': input_params,
                      'neuron_params': neuron_params,
                      'connection_params': {'replace_non_simulated_areas': 'hom_poisson_stat'}}

    sim_params = {'t_sim': 1000.,
                  'num_processes': 720,
                  'local_num_threads': 1,
                  'recording_dict': {'record_vm': False},
                                  #   'areas_simulated': ['V1'],
                  }
    
    theory_params = {'dt': 0.1}

    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=False,
                       theory_spec=theory_params, analysis=True)
    #p, r = M.theory.integrate_siegert()
    #print("Mean-field theory predicts an average "
    #      "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
    M.simulation.simulate()

    M.analysis.load_data()
    M.analysis.create_pop_rates()
    M.analysis.create_pop_rate_dists()
    M.analysis.create_rate_time_series()

    for area in M.analysis.areas_loaded:
        M.analysis.single_rate_display(area,output = "png")
        # frac_neurons : float, [0,1]
        frac_neurons = 0.01
        M.analysis.single_dot_display(area,frac_neurons,output = "png")
        # M.analysis.single_power_display(area,output = "png")

    M.analysis.show_rates(output = "png")
    

if __name__ == "__main__":
    
    # 创建多个进程，每个进程运行 main 函数，传递不同的 GPU ID
    main(7)
    # processes = []

    # # 在这里定义你希望运行的 GPU ID 列表
    # ids = [2]
    # # ids = [2]

    # for id in ids:
    #     process = multiprocessing.Process(target=main, args=(id,))
    #     processes.append(process)

    # # 启动所有进程
    # for process in processes:
    #     process.start()

    # # 等待所有进程完成
    # for process in processes:
    #     process.join()
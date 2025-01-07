import numpy as np
import os

from multiarea_model import MultiAreaModel
from config import base_path

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

"""
Down-scaled model.
Neurons and indegrees are both scaled down to 10 %.
Can usually be simulated on a local machine.

Warning: This will not yield reasonable dynamical results from the
network and is only meant to demonstrate the simulation workflow.
"""


d = {}
conn_params = {'replace_non_simulated_areas': 'het_poisson_stat',
               'g': -3.,
            #    'K_stable': os.path.join(base_path, "K_stable.npy"),
                'K_stable':None,
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 2,
               'PSP_e_23_4': 0.15,
               'PSP_ext': 0.3,
               'av_indegree_V1': 3950.}
input_params = {'rate_ext': 20}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
network_params = {'N_scaling': 0.005,
                  'K_scaling': 0.005,
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

M = MultiAreaModel(network_params, simulation=True,
                   sim_spec=sim_params,
                   theory=False,
                   theory_spec=theory_params,
                   analysis= True)
# p, r = M.theory.integrate_siegert()
# print("Mean-field theory predicts an average "
#       "rate of {0:.3f} spikes/s across all populations.".format(np.mean(r[:, -1])))
M.simulation.simulate()
M.upload_file()
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
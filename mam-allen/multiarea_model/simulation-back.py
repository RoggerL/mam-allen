
"""
multiarea_model
==============

Simulation class of the multi-area model of macaque visual vortex by
Schmidt et al. (2018).


Classes
-------
Simulation : Loads a parameter file that specifies simulation
parameters for a simulation of the instance of the model. A simulation
is identified by a unique hash label.

"""

import json
import numpy as np
import os
import pprint
import shutil
import time

from .analysis_helpers import _load_npy_to_dict, model_iter
from config import base_path, data_path
from copy import deepcopy
from .default_params import nested_update, sim_params
from .default_params import check_custom_params
from dicthash import dicthash
# from pygenn import genn_model, genn_wrapper,ModelSpec
from pygenn import (GeNNModel, VarLocation, init_postsynaptic,
                    init_sparse_connectivity, init_weight_update, init_var,ModelSpec,create_neuron_model)
# from pygenn.genn_wrapper.CUDABackend import BlockSizeSelect_MANUAL, DeviceSelect_MANUAL
from scipy.stats import norm
from six import iteritems, itervalues
from .multiarea_helpers import extract_area_dict, create_vector_mask
try:
    from .sumatra_helpers import register_runtime
    sumatra_found = True
except ImportError:
    sumatra_found = False

class Simulation:
    def __init__(self, network, sim_spec):
        """
        Simulation class.
        An instance of the simulation class with the given parameters.
        Can be created as a member class of a multiarea_model instance
        or standalone.

        Parameters
        ----------
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network to be simulated.
        params : dict
            custom simulation parameters that overwrite the
            default parameters defined in default_params.py
        """
        self.params = deepcopy(sim_params)
        if isinstance(sim_spec, dict):
            check_custom_params(sim_spec, self.params)
            self.custom_params = sim_spec
        else:
            fn = os.path.join(data_path,
                              sim_spec,
                              '_'.join(('custom_params',
                                        sim_spec)))
            with open(fn, 'r') as f:
                self.custom_params = json.load(f)['sim_params']

        nested_update(self.params, self.custom_params)

        self.network = network
        self.label = dicthash.generate_hash_from_dict({'params': self.params,
                                                       'network_label': self.network.label})

        print("Simulation label: {}".format(self.label))
        self.data_dir = os.path.join(data_path, self.label)
        try:
            os.mkdir(self.data_dir)
            os.mkdir(os.path.join(self.data_dir, 'recordings'))
        except OSError:
            pass
        self.copy_files()
        print("Copied files.")
        d = {'sim_params': self.custom_params,
             'network_params': self.network.custom_params,
             'network_label': self.network.label}
        with open(os.path.join(self.data_dir,
                               '_'.join(('custom_params', self.label))), 'w') as f:
            json.dump(d, f)
        print("Initialized simulation class.")

        self.areas_simulated = self.params['areas_simulated']
        self.areas_recorded = self.params['recording_dict']['areas_recorded']
        self.T = self.params['t_sim']
        self.extra_global_param_bytes = 0

    def __eq__(self, other):
        # Two simulations are equal if the simulation parameters and
        # the simulated networks are equal.
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        s = "Simulation {} of network {} with parameters:".format(self.label, self.network.label)
        s += pprint.pformat(self.params, width=1)
        return s

    def copy_files(self):
        """
        Copy all relevant files for the simulation to its data directory.
        """
        files = [os.path.join('multiarea_model',
                              'data_multiarea',
                              'Model.py'),
                 os.path.join('multiarea_model',
                              'data_multiarea',
                              'VisualCortex_Data.py'),
                 os.path.join('multiarea_model',
                              'multiarea_model.py'),
                 os.path.join('multiarea_model',
                              'simulation.py'),
                 os.path.join('multiarea_model',
                              'default_params.py'),
                 os.path.join('config_files',
                              ''.join(('custom_Data_Model_', self.network.label, '.json'))),
                 os.path.join('config_files',
                              '_'.join((self.network.label, 'config')))]
        if self.network.params['connection_params']['replace_cc_input_source'] is not None:
            fs = self.network.params['connection_params']['replace_cc_input_source']
            if '.json' in fs:
                files.append(fs)
            else:  # Assume that the cc input is stored in one npy file per population
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                for it in fn_iter:
                    fp_it = (fs,) + it
                    fp_ = '{}.npy'.format('-'.join(fp_it))
                    files.append(fp_)
        for f in files:
            shutil.copy2(os.path.join(base_path, f),
                         self.data_dir)

    def prepare(self):
        """
        Prepare GeNN model.
        """
        # self.model = genn_model.GeNNModel("float", "multi_area_model",
        #                                   code_gen_log_level=genn_wrapper.info, 
        #                                   deviceSelectMethod=DeviceSelect_MANUAL,
        #                                   blockSizeSelectMethod=BlockSizeSelect_MANUAL,
        #                                   selectGPUByDeviceID=True)
        self.model = GeNNModel("float", "multi_area_model",backend="cuda",manual_device_id=0)

        # self.model = genn_model.GeNNModel("float", "multi_area_model",
        #                                   code_gen_log_level=genn_wrapper.info, 
        #                                   deviceSelectMethod=DeviceSelect_MANUAL,
        #                                   blockSizeSelectMethod=BlockSizeSelect_MANUAL,
        #                                   selectGPUByDeviceID=True)
        self.model.dT = self.params['dt']
        # self.model._model.set_merge_postsynaptic_models(True)
        self.model.timing_enabled = self.params['timing_enabled']
        # self.model.default_var_location = genn_wrapper.VarLocation_HOST_DEVICE

        # self.model.default_sparse_connectivity_location = genn_wrapper.VarLocation_HOST_DEVICE
        # self.model._model.set_seed(self.params['master_seed'])
        ModelSpec.seed = (self.params['master_seed'])
        
        quantile = 0.9999
        normal_quantile_cdf = norm.ppf(quantile)
        
        # Calculate max delay for 
        ex_delay_sd = self.network.params['delay_params']['delay_e'] * self.network.params['delay_params']['delay_rel']
        in_delay_sd = self.network.params['delay_params']['delay_i'] * self.network.params['delay_params']['delay_rel']
        max_ex_delay = self.network.params['delay_params']['delay_e'] + (ex_delay_sd * normal_quantile_cdf)
        max_in_delay = self.network.params['delay_params']['delay_i'] + (in_delay_sd * normal_quantile_cdf)
        
        self.max_inter_area_delay = max(max_ex_delay, max_in_delay)
        print("Max inter-area delay:%fms" % self.max_inter_area_delay)

        # If we should create cortico-cortico connections
        if not self.network.params['connection_params']['replace_cc']:
            self.max_intra_area_delay = 0
            for target_area in self.areas_simulated:

                # Loop source area though complete list of areas
                for source_area in self.network.area_list:
                    if target_area != source_area:
                        # If source_area is part of the simulated network,
                        # connect it to target_area
                        if source_area in self.areas_simulated:
                            v = self.network.params['delay_params']['interarea_speed']
                            s = self.network.distances[target_area][source_area]
                            mean_delay = s / v

                            delay_sd = mean_delay * self.network.params['delay_params']['delay_rel']
                            self.max_intra_area_delay = max(self.max_intra_area_delay,
                                                            mean_delay + (delay_sd * normal_quantile_cdf))
            print("Max intra-area delay:%fms" % self.max_intra_area_delay)

    def create_areas(self):
        """
        Create all areas with their populations and internal connections.
        """
        self.areas = []
        for area_name in self.areas_simulated:
            a = Area(self, self.network, area_name)
            self.areas.append(a)

    def cortico_cortical_input(self):
        """
        Create connections between areas.
        """
        replace_cc = self.network.params['connection_params']['replace_cc']
        replace_non_simulated_areas = self.network.params['connection_params'][
            'replace_non_simulated_areas']
        if self.network.params['connection_params']['replace_cc_input_source'] is None:
            replace_cc_input_source = None
        else:
            replace_cc_input_source = os.path.join(self.data_dir,
                                                   self.network.params['connection_params'][
                                                       'replace_cc_input_source'])

        if not replace_cc and set(self.areas_simulated) != set(self.network.area_list):
            if replace_non_simulated_areas == 'het_current_nonstat':
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                non_simulated_cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
            elif replace_non_simulated_areas == 'het_poisson_stat':
                fn = self.network.params['connection_params']['replace_cc_input_source']
                with open(fn, 'r') as f:
                    non_simulated_cc_input = json.load(f)
            elif replace_non_simulated_areas == 'hom_poisson_stat':
                non_simulated_cc_input = {source_area_name:
                                          {source_pop:
                                           self.network.params['input_params']['rate_ext']
                                           for source_pop in
                                           self.network.structure[source_area_name]}
                                          for source_area_name in self.network.area_list}
            else:
                raise KeyError("Please define a valid method to"
                               " replace non-simulated areas.")

        if replace_cc == 'het_current_nonstat':
            fn_iter = model_iter(mode='single', areas=self.network.area_list)
            cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
        elif replace_cc == 'het_poisson_stat':
            with open(self.network.params['connection_params'][
                    'replace_cc_input_source'], 'r') as f:
                cc_input = json.load(f)
        elif replace_cc == 'hom_poisson_stat':
            cc_input = {source_area_name:
                        {source_pop:
                         self.network.params['input_params']['rate_ext']
                         for source_pop in
                         self.network.structure[source_area_name]}
                        for source_area_name in self.network.area_list}

        # Connections between simulated areas are not replaced
        if not replace_cc:
            for target_area in self.areas:
                # Loop source area though complete list of areas
                for source_area_name in self.network.area_list:
                    if target_area.name != source_area_name:
                        # If source_area is part of the simulated network,
                        # connect it to target_area
                        if source_area_name in self.areas:
                            source_area = self.areas[self.areas.index(source_area_name)]
                            target_area.connect(source_area)
                            # connect(self,
                            #         target_area,
                            #         source_area)
                        # Else, replace the input from source_area with the
                        # chosen method
                        else:
                            target_area.create_additional_input(replace_non_simulated_areas,
                                                                source_area_name,
                                                                non_simulated_cc_input[
                                                                    source_area_name])
        # Connections between all simulated areas are replaced
        else:
            for target_area in self.areas:
                for source_area in self.areas:
                    if source_area != target_area:
                        target_area.create_additional_input(replace_cc,
                                                            source_area.name,
                                                            cc_input[source_area.name])

    def simulate(self):
        """
        Create the network and execute simulation.
        Record used memory and wallclock time.
        """
        t0 = time.time()
        self.prepare()
        t1 = time.time()
        self.time_prepare = t1 - t0
        print("Prepared simulation in {0:.2f} seconds.".format(self.time_prepare))

        self.create_areas()
        t2 = time.time()
        self.time_network_local = t2 - t1
        print("Created areas and internal connections in {0:.2f} seconds.".format(
            self.time_network_local))

        self.cortico_cortical_input()
        t3 = time.time()
        self.time_network_global = t3 - t2
        print("Created cortico-cortical connections in {0:.2f} seconds.".format(
            self.time_network_global))

        print("Extra global parameters require: %u MB" % (self.extra_global_param_bytes / (1024 * 1024)))

        if self.params['rebuild_model']:
            self.model.build()
            t4 = time.time()
            self.time_genn_build = t4 - t3
            print("Built GeNN model in {0:.2f} seconds.".format(self.time_genn_build))
        else:
            t4 = t3
        self.model.load(num_recording_timesteps=self.params['recording_buffer_timesteps'])
        t5 = time.time()
        self.time_genn_load = t5 - t4
        print("Loaded GeNN model in {0:.2f} seconds.".format(self.time_genn_load))
        
        # Loop through simulation time
        while self.model.t < self.T:
            print("t=",self.model.t)
            # Step time
            self.model.step_time()
            # If recording buffer is full
            if not self.model.timestep % self.params['recording_buffer_timesteps']:
                # Download recording data
                self.model.pull_recording_buffers_from_device()
                print("recording")

                # Loop through areas
                for a in self.areas:
                    a.record()
            for a in self.areas:
                a.record_vm()
                # a.record_isyn()

        t6 = time.time()
        self.time_simulate = t6 - t5
        print("Simulated network in {0:.2f} seconds.".format(self.time_simulate))

        # If timing is enabled, read kernel timing from model
        if self.params['timing_enabled']:
            self.time_genn_init = 1000.0 * self.model.init_time
            self.time_genn_init_sparse = 1000.0 * self.model.init_sparse_time
            self.time_genn_neuron_update = 1000.0 * self.model.neuron_update_time
            self.time_genn_presynaptic_update = 1000.0 * self.model.presynaptic_update_time

        # Write recorded data to disk
        for a in self.areas:
            a.write_recorded_data()

        self.logging()

    def logging(self):
        """
        Write runtime to file.
        """
        d = {'time_prepare': self.time_prepare,
             'time_network_local': self.time_network_local,
             'time_network_global': self.time_network_global,
             'time_simulate': self.time_simulate}

        if self.params['timing_enabled']:
            d.update({'time_genn_init': self.time_genn_init,
                      'time_genn_init_sparse': self.time_genn_init_sparse,
                      'time_genn_neuron_update': self.time_genn_neuron_update,
                      'time_genn_presynaptic_update': self.time_genn_presynaptic_update})

        fn = os.path.join(self.data_dir, 'recordings','_'.join((self.label, 'logfile')))
        with open(fn, 'w') as f:
            json.dump(d, f)
    def register_runtime(self):
        if sumatra_found:
            register_runtime(self.label)
        else:
            raise ImportWarning('Sumatra is not installed, the '
                                'runtime cannot be registered.')


class Area:
    def __init__(self, simulation, network, name):
        """
        Area class.
        This class encapsulates a single area of the model.
        It creates all populations and the intrinsic connections between them.
        It provides an interface to allow connecting the area to other areas.

        Parameters
        ----------
        simulation : simulation
           An instance of the simulation class that specifies the
           simulation that the area is part of.
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network the area is part of.
        name : str
            Name of the area.
        """

        self.name = name
        self.simulation = simulation
        self.network = network
        self.neuron_numbers = network.N[name]
        self.synapses = extract_area_dict(network.synapses,
                                          network.structure,
                                          self.name,
                                          self.name)
        self.W = extract_area_dict(network.W,
                                   network.structure,
                                   self.name,
                                   self.name)
        # print("self_W=",self.W)
        self.W_sd = extract_area_dict(network.W_sd,
                                      network.structure,
                                      self.name,
                                      self.name)
        self.K = extract_area_dict(network.K,
                                      network.structure,
                                      self.name,
                                      self.name)
        self.populations = network.structure[name]

        self.external_synapses = {}
        for pop in self.populations:
            self.external_synapses[pop] = self.network.K[self.name][pop]['external']['external']

        self.create_populations()
        self.genn_syns = {}
        self.syns_currents = {}
        self.connect_populations()
        print("created area {} with {} local nodes".format(self.name, self.num_local_nodes))

    def __str__(self):
        s = "Area {} with {} neurons.".format(
            self.name, int(self.neuron_numbers['total']))
        return s

    def __eq__(self, other):
        # If other is an instance of area, it should be the exact same
        # area This opens the possibility to have multiple instance of
        # one cortical areas
        if isinstance(other, Area):
            return self.name == other.name #and self.gids == other.gids
        elif isinstance(other, str):
            return self.name == other

    def create_populations(self):
        """
        Create all populations of the area.
        """
        neuron_params = self.network.params['neuron_params']
        v_init_params = {"mean": neuron_params['V0_mean'], "sd": neuron_params['V0_sd']}
        lif_init = {"RefracTime": 0.0, "V": init_var("Normal", v_init_params),"current_syn" : 0.0}
        # lif_init = {"RefracTime": 0.0, "V": init_var("Normal", v_init_params)}
        lif_params = {
                      "E":
                      {"C": neuron_params['single_neuron_dict']["E"]['C_m'] / 1000.0, 
                      "TauM": neuron_params['single_neuron_dict']["E"]['tau_m'], 
                      "Vrest": neuron_params['single_neuron_dict']["E"]['E_L'], 
                      "Vreset": neuron_params['single_neuron_dict']["E"]['V_reset'], 
                      "Vthresh" : neuron_params['single_neuron_dict']["E"]['V_th'],
                      "TauRefrac": neuron_params['single_neuron_dict']["E"]['t_ref']},
                      "S":
                      {"C": neuron_params['single_neuron_dict']["S"]['C_m'] / 1000.0, 
                      "TauM": neuron_params['single_neuron_dict']["S"]['tau_m'], 
                      "Vrest": neuron_params['single_neuron_dict']["S"]['E_L'], 
                      "Vreset": neuron_params['single_neuron_dict']["S"]['V_reset'], 
                      "Vthresh" : neuron_params['single_neuron_dict']["S"]['V_th'],
                      "TauRefrac": neuron_params['single_neuron_dict']["S"]['t_ref']},
                      "P":
                      {"C": neuron_params['single_neuron_dict']["P"]['C_m'] / 1000.0, 
                      "TauM": neuron_params['single_neuron_dict']["P"]['tau_m'], 
                      "Vrest": neuron_params['single_neuron_dict']["P"]['E_L'], 
                      "Vreset": neuron_params['single_neuron_dict']["P"]['V_reset'], 
                      "Vthresh" : neuron_params['single_neuron_dict']["P"]['V_th'],
                      "TauRefrac": neuron_params['single_neuron_dict']["P"]['t_ref']},
                      "H":
                      {"C": neuron_params['single_neuron_dict']["H"]['C_m'] / 1000.0, 
                      "TauM": neuron_params['single_neuron_dict']["H"]['tau_m'], 
                      "Vrest": neuron_params['single_neuron_dict']["H"]['E_L'], 
                      "Vreset": neuron_params['single_neuron_dict']["H"]['V_reset'], 
                      "Vthresh" : neuron_params['single_neuron_dict']["H"]['V_th'],
                      "TauRefrac": neuron_params['single_neuron_dict']["H"]['t_ref']},
                      "V":
                      {"C": neuron_params['single_neuron_dict']["V"]['C_m'] / 1000.0, 
                      "TauM": neuron_params['single_neuron_dict']["V"]['tau_m'], 
                      "Vrest": neuron_params['single_neuron_dict']["V"]['E_L'], 
                      "Vreset": neuron_params['single_neuron_dict']["V"]['V_reset'], 
                      "Vthresh" : neuron_params['single_neuron_dict']["V"]['V_th'],
                      "TauRefrac": neuron_params['single_neuron_dict']["V"]['t_ref']}
                      } 

        # print("lif_parm=",lif_params)
        
        poisson_init = {"current": 0.0}

        self.genn_pops = {}
        # self.genn_syns = {}
        self.voltage_views = {}
        self.current_views = {}
        self.spike_times = {}
        self.spike_ids = {}
        self.num_local_nodes = 0
        for pop in self.populations:
            assert neuron_params['neuron_model'] == 'iaf_psc_exp'
            
            if pop[0] == 'E':
                pop_lif_params = deepcopy(lif_params['E'])
            if pop[0] == 'S':
                pop_lif_params = deepcopy(lif_params['S'])
            if pop[0] == 'P':
                pop_lif_params = deepcopy(lif_params['P'])
            if pop[0] == 'V':
                pop_lif_params = deepcopy(lif_params['V'])
            if pop[0] == 'H':
                pop_lif_params = deepcopy(lif_params['H'])

            mask = create_vector_mask(self.network.structure, areas=[self.name], pops=[pop])
            pop_lif_params['Ioffset'] = self.network.add_DC_drive[mask][0] / 1000.0
            # if not self.network.params['input_params']['poisson_input']:
            #     K_ext = self.network.K[self.name][pop]['external']['external']
            #     W_ext = self.network.W[self.name][pop]['extenal']['external']
            #     tau_syn = self.network.params['neuron_params']['single_neuron_dict'][pop]['tau_syn']
            #     DC = K_ext * W_ext * tau_syn * 1.e-3 * \
            #         self.network.params['rate_ext']
            #     pop_lif_params['Ioffset'] += (DC / 1000.0)
            if not self.network.params['input_params']['poisson_input']:
                print("self.K=",self.K)
                print("self.W=",self.W)
                rate = 10
                total = 0
                tau_syn = 0.5
                DC = self.network.params['input_params']['input'][pop]
                # print("pop=",pop)
                # print("origin_DC=",DC)
                for source_pop in self.populations:
                    total =total +  self.K[pop][source_pop]*self.W[pop][source_pop]*tau_syn*rate/1000.
                print("total=",total)
                # if pop != 'H1':
                # if total < 0:
                #     DC = self.network.params['input_params']['input'][pop]-total
                print("pop=",pop)
                print("DC=",DC)
                pop_lif_params['Ioffset'] += (DC / 1000.0)
            # define LIF model
            # lif_model = genn_model.create_custom_neuron_class(
            #     "LIF2",
            #     param_names=["C", "TauM", "Vrest", "Vreset", "Vthresh", "Ioffset", "TauRefrac"],
            #     var_name_types=[("V", "scalar"), ("RefracTime", "scalar")],
            #     sim_code="""
            #     if ($(RefracTime) <= 0.0) {
            #         scalar alpha = (($(Isyn) + $(Ioffset)) * $(Rmembrane)) + $(Vrest);
            #         $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
            #     } else {
            #         $(RefracTime) -= DT;
            #     }
            #     """,
            #     threshold_condition_code="$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)",
            #     reset_code="""
            #     $(V) = $(Vreset);
            #     $(RefracTime) = $(TauRefrac);
            #     """,
            #     derived_params=[
            #         ("ExpTC", lambda pars, dt: np.exp(-dt / pars["TauM"])),
            #         ("Rmembrane", lambda pars, dt: pars["TauM"] / pars["C"])
            #     ])
            
        # leaky_integrator_model = pygenn.create_neuron_model(
        #     "leaky_integrator",
        #     sim_code="""V += (-V + Isyn) * (dt / tau);""",
        #     threshold_condition_code="V >= 1.0",
        #     reset_code="""V = 0.0;""",
        #     params=["tau"],
        #     vars=[("V", "scalar", pygenn.VarAccess.READ_WRITE)])

                    #   {"C": neuron_params['single_neuron_dict']["E"]['C_m'] / 1000.0, 
                    #   "TauM": neuron_params['single_neuron_dict']["E"]['tau_m'], 
                    #   "Vrest": neuron_params['single_neuron_dict']["E"]['E_L'], 
                    #   "Vreset": neuron_params['single_neuron_dict']["E"]['V_reset'], 
                    #   "Vthresh" : neuron_params['single_neuron_dict']["E"]['V_th'],
                    #   "TauRefrac": neuron_params['single_neuron_dict']["E"]['t_ref']},
            
            lif_model = create_neuron_model(
                "lif_model",
                params=["Vthresh", "TauM", "TauRefrac", "C", "Vrest", "Vreset","Ioffset"],  # 添加参数
                vars=[("V", "scalar"), ("RefracTime", "scalar"),("current_syn" , "scalar")],  # 添加不应期状态变量
                sim_code="""
                if (RefracTime <= 0.0) {
                    scalar alpha = ((Isyn + Ioffset) * Rmembrane) + Vrest;
                     V = alpha - (ExpTC * (alpha - V));
                } else {
                    RefracTime -= dt;
                }
                current_syn = Isyn;
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
        
            # Create GeNN population
            pop_name = self.name + '_' + pop
            genn_pop = self.simulation.model.add_neuron_population(pop_name, int(self.neuron_numbers[pop]),
                                                                    lif_model, pop_lif_params, lif_init)
            # genn_pop = self.simulation.model.add_neuron_population(pop_name, int(self.neuron_numbers[pop]),
            #                                                         'LIF', pop_lif_params, lif_init)
            # genn_pop.pop.set_var_location("V", genn_wrapper.VarLocation_HOST_DEVICE)
            # genn_pop.pop.set_var_location("inSynInSyn0", genn_wrapper.VarLocation_HOST_DEVICE)
            # genn_pop.pop.set_var_location("Isyn", genn_wrapper.VarLocation_HOST_DEVICE)
            
            #recording vm and current
            genn_pop.spike_recording_enabled = True
            # genn_pop.vm_recording_enabled = True
            # genn_pop.current_recording_enabled = True

            # If Poisson input is required
            if self.network.params['input_params']['poisson_input']:
                # print("input_type is poisson")
                # Add current source
                # print("pop=",pop)
                # print("input_rate=",self.network.params['input_params']['rate_ext'] * self.external_synapses[pop])
                poisson_params = {"weight": self.network.W[self.name][pop]['external']['external'] / 1000.0,
                                  "tauSyn": neuron_params['single_neuron_dict']['E']['tau_syn'],
                                  "rate": self.network.params['input_params']['rate_ext']}
                self.simulation.model.add_current_source(pop_name + "_poisson", "PoissonExp", genn_pop,
                                                         poisson_params, poisson_init)
            # Add population to dictionary
            self.genn_pops[pop] = genn_pop
            self.voltage_views[pop] = []
            self.current_views[pop] = []
            self.spike_times[pop] = []
            self.spike_ids[pop] = []

    def connect_populations(self):
        """
        Create connections between populations.
        """
        self.connect(self)
        # connect(self.simulation,
        #         self,
        #         self)

    def record(self):
        # If anything should be recorded from this area
        # **YUCK** this is gonna be slow
        if self.name in self.simulation.params['recording_dict']['areas_recorded']:
            # Loop through GeNN populations in area
            # v = None
            for pop, genn_pop in iteritems(self.genn_pops):
                # print(pop)
                # for v in genn_pop.vars:
                    # print("v=",v)
                # for v_ref in genn_pop.vars_ref:
                #     print("v_ref=",v_ref)
                # print(genn_pop.vars)
                # genn_pop.pull_var_from_device("IsynRecord")
                # print(genn_pop.vars["IsynRecord"].view)
                # genn_pop.pull_var_from_device("V")
                # self.voltage_views[pop].append(np.mean(genn_pop.vars["V"].view))
                # print(np.mean(genn_pop.vars["V"].view))
                # self.voltage_views[pop].append(np.mean(genn_pop.vars["V"].view))
                # print("len(self.voltage_views[pop])=",len(self.voltage_views[pop]))
                # v = np.copy(self.voltage_views[pop]) if v is None else np.vstack((v, self.voltage_views[pop]))
                # print(v)
                # Extract spike recording data
                # print(type(genn_pop))
                spike_times, spike_ids = genn_pop.spike_recording_data[0]
                # state = genn_pop.pull_var_from_device("SpikeTimes")
                # print(state)
                
                # Add spike times and ids to list
                self.spike_times[pop].append(spike_times)
                self.spike_ids[pop].append(spike_ids)

    def record_vm(self):
        # If anything should be recorded from this area
        # **YUCK** this is gonna be slow
        if self.name in self.simulation.params['recording_dict']['areas_recorded']:
            # Loop through GeNN populations in area
            # v = None
            for pop, genn_pop in iteritems(self.genn_pops):
                # print(pop)
                # for v in genn_pop.vars:
                #     print(v)
                # print(genn_pop.vars)
                # genn_pop.pull_var_from_device("IsynRecord")
                # print(genn_pop.vars["IsynRecord"].view)
                genn_pop.pull_var_from_device("V") 
                # genn_pop.pull_var_from_device("inSynInSyn0")
                # self.voltage_views[pop].append(np.mean(genn_pop.vars["V"].view))
                # print(np.mean(genn_pop.vars["V"].view))
                self.voltage_views[pop].append(np.mean(genn_pop.vars["V"].view))
                
                
                # genn_pop.pull_var_from_device("current_syn") 
                self.current_views[pop].append(np.mean(genn_pop.vars["current_syn"].view))
                print("current_syn=",np.mean(genn_pop.vars["current_syn"].view))
                # print("len(self.voltage_views[pop])=",len(self.voltage_views[pop]))
                # v = np.copy(self.voltage_views[pop]) if v is None else np.vstack((v, self.voltage_views[pop]))
                # print(v)
                # Extract spike recording data
                # print(type(genn_pop))
                # spike_times, spike_ids = genn_pop.spike_recording_data
                # state = genn_pop.pull_var_from_device("SpikeTimes")
                # print(state)
                
                # Add spike times and ids to list
                # self.spike_times[pop].append(spike_times)
                # self.spike_ids[pop].append(spike_ids)                
        #record currents
        # for syn in self. 
        # pull_in_syn_from_device

    def record_isyn(self):
        # If anything should be recorded from this area
        # **YUCK** this is gonna be slow
        # if self.name in self.simulation.params['recording_dict']['areas_recorded']:
            # Loop through GeNN populations in area
            # v = None
            # for syn_name, syn_pop in iteritems(self.genn_syns):
                # syn_pop.pull_in_syn_from_device()
                # print("isyn=",syn_pop.vars["inSyn"].view)
                # self.syns_currents[syn_name].append(np.mean(syn_pop.vars["inSyn"].view))
        pass

    def write_recorded_data(self):
        # Determine path for recorded data
        recording_path = os.path.join(self.simulation.data_dir, 'recordings')

        timesteps = np.arange(0.0, self.simulation.T, self.simulation.params['dt'])

        # If anything should be recorded from this area
        # **YUCK** this is gonna be slow
        if self.name in self.simulation.params['recording_dict']['areas_recorded']:
            for pop in self.populations:
                # Stack all spike times and IDs together
                spike_times = np.hstack(self.spike_times[pop])
                spike_ids = np.hstack(self.spike_ids[pop])
                # Stack all spike times and IDs together
                # spike_times = np.array(self.spike_times[pop])
                # spike_ids = np.array(self.spike_ids[pop])
                # print("pop={},spike_times={}".format(spike_ids,spike_times))
                # print("type_times=",type(self.spike_times[pop][0]))
                # print("type_ids=",type(self.spike_ids[pop][0]))
                
                # print("spike_times=",self.spike_times[pop])
                # print("spike_ids=",self.spike_ids[pop])
                
                # Write recorded data to disk
                np.save(os.path.join(recording_path, self.name + "_" + pop + ".npy"), np.vstack([spike_times, spike_ids]))
                np.save(os.path.join(recording_path, self.name + "_" + pop + "_voltage" +".npy"), self.voltage_views[pop])
                np.save(os.path.join(recording_path, self.name + "_" + pop + "_current" +".npy"), self.current_views[pop])
                # print("save_path=",os.path.join(recording_path, self.name + "_" + pop + "_voltage" +".npy"))

    def create_additional_input(self, input_type, source_area_name, cc_input):
        """
        Replace the input from a source area by the chosen type of input.

        Parameters
        ----------
        input_type : str, {'het_current_nonstat', 'hom_poisson_stat',
                           'het_poisson_stat'}
            Type of input to replace source area. The source area can
            be replaced by Poisson sources with the same global rate
            rate_ext ('hom_poisson_stat') or by specific rates
            ('het_poisson_stat') or by time-varying specific current
            ('het_current_nonstat')
        source_area_name: str
            Name of the source area to be replaced.
        cc_input : dict
            Dictionary of cortico-cortical input of the process
            replacing the source area.
        """
        # assert False
        """
        synapses = extract_area_dict(self.network.synapses,
                                     self.network.structure,
                                     self.name,
                                     source_area_name)
        W = extract_area_dict(self.network.W,
                              self.network.structure,
                              self.name,
                              source_area_name)
        v = self.network.params['delay_params']['interarea_speed']
        s = self.network.distances[self.name][source_area_name]
        delay = s / v
        for pop in self.populations:
            for source_pop in self.network.structure[source_area_name]:
                syn_spec = {'weight': W[pop][source_pop],
                            'delay': delay}
                K = synapses[pop][source_pop] / self.neuron_numbers[pop]

                if input_type == 'het_current_nonstat':
                    curr_gen = nest.Create('step_current_generator', 1)
                    dt = self.simulation.params['dt']
                    T = self.simulation.params['t_sim']
                    assert(len(cc_input[source_pop]) == int(T))
                    nest.SetStatus(curr_gen, {'amplitude_values': K * cc_input[source_pop] * 1e-3,
                                              'amplitude_times': np.arange(dt,
                                                                           T + dt,
                                                                           1.)})
                    nest.Connect(curr_gen,
                                 tuple(
                                     range(self.gids[pop][0], self.gids[pop][1] + 1)),
                                 syn_spec=syn_spec)
                elif 'poisson_stat' in input_type:  # hom. and het. poisson lead here
                    pg = nest.Create('poisson_generator', 1)
                    nest.SetStatus(pg, {'rate': K * cc_input[source_pop]})
                    nest.Connect(pg,
                                 tuple(
                                     range(self.gids[pop][0], self.gids[pop][1] + 1)),
                                 syn_spec=syn_spec)
        """
    
    def connect(self, source_area):
        """
        source_area : Area instance
            Source area of the projection
        """       
        num_sub_rows = (self.simulation.params['num_threads_per_spike'] 
                        if self.simulation.params['procedural_connectivity'] 
                        else 1)     
        # matrix_type = "PROCEDURAL_PROCEDURALG" if self.simulation.params['procedural_connectivity']  else "SPARSE_INDIVIDUALG"    
        matrix_type = "PROCEDURAL" if self.simulation.params['procedural_connectivity'] else "SPARSE"
        network = self.simulation.network
        synapses = extract_area_dict(network.synapses,
                                     network.structure,
                                     self.name,
                                     source_area.name)
        # print("synapses=",synapses)
        W = extract_area_dict(network.W,
                              network.structure,
                              self.name,
                              source_area.name)
        W_sd = extract_area_dict(network.W_sd,
                                 network.structure,
                                 self.name,
                                 source_area.name)

        for target in self.populations:
            for source in source_area.populations:
                #Cut the connection between different layers.
                if target[1:] != source[1:] and self.simulation.params["cut_connect"]:
                    # print("cut connection")
                    continue
                
                # print("not cut off")    
                num_connections = int(synapses[target][source])
                conn_spec = {"num": num_connections}

                # syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0}
                exp_curr_params = {}

                if self == source_area:
                    max_delay = self.simulation.max_inter_area_delay
                    if 'E' in source:
                        mean_delay = network.params['delay_params']['delay_e']
                    else:
                        mean_delay = network.params['delay_params']['delay_i']
                else:
                    max_delay = self.simulation.max_intra_area_delay
                    v = network.params['delay_params']['interarea_speed']
                    s = network.distances[self.name][source_area.name]
                    mean_delay = s / v

                # if source[0] == 'E':
                #     # exp_curr_params.update({'tau': network.params['neuron_params']['single_neuron_dict']['E']['tau_syn']})
                #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['E']['tau_syn']})
                #     # syn_weight.update({'min': 0., 'max': float(np.finfo(np.float32).max)})
                #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0, "min": 0.0, "max": float(np.finfo(np.float32).max)}
                
                # if source[0] == 'V':
                #     # exp_curr_params.update({'tau': network.params['neuron_params']['single_neuron_dict']['V']['tau_syn']})
                #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['V']['tau_syn']})
                #     # syn_weight.update({'min': float(-np.finfo(np.float32).max), 'max': 0.})
                #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0, "min": float(-np.finfo(np.float32).max), "max": 0.0}

                # if source[0] == 'P':
                #     # exp_curr_params.update({'tau': network.params['neuron_params']['single_neuron_dict']['P']['tau_syn']})
                #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['P']['tau_syn']})
                #     # syn_weight.update({'min': float(-np.finfo(np.float32).max), 'max': 0.})
                #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0, "min": float(-np.finfo(np.float32).max), "max": 0.0}

                # if source[0] == 'S':
                #     # exp_curr_params.update({'tau': network.params['neuron_params']['single_neuron_dict']['S']['tau_syn']})
                #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['S']['tau_syn']})
                #     # syn_weight.update({'min': float(-np.finfo(np.float32).max), 'max': 0.})
                #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0, "min": float(-np.finfo(np.float32).max), "max": 0.0}

                # if source[0] == 'H':
                #     # exp_curr_params.update({'tau': network.params['neuron_params']['single_neuron_dict']['S']['tau_syn']})
                #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['H']['tau_syn']})
                #     # syn_weight.update({'min': float(-np.finfo(np.float32).max), 'max': 0.})
                #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0, "min": float(-np.finfo(np.float32).max), "max": 0.0}

                # if target == "H1":
                #     print("source=",source)                
                #     print("syn_weight=",syn_weight)
                
                exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})
                syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0, "min": float(-np.finfo(np.float32).max), "max": float(np.finfo(np.float32).max)}



                delay_sd = mean_delay * network.params['delay_params']['delay_rel']


                syn_delay = {'min': self.simulation.params['dt'],
                             'max': max_delay,
                             'mean': mean_delay,
                             'sd': delay_sd}
                


                syn_spec = {'g': init_var("NormalClipped", syn_weight),
                            'd': init_var("NormalClippedDelay", syn_delay)}

                # Add synapse population
                source_genn_pop = source_area.genn_pops[source]
                target_genn_pop = self.genn_pops[target]
                # syn_pop = self.simulation.model.add_synapse_population(source_genn_pop.name + "_" + target_genn_pop.name, 
                #     matrix_type, 0,
                #     source_genn_pop, target_genn_pop,
                #     "StaticPulseDendriticDelay", {}, syn_spec, {}, {},
                #     "ExpCurr", exp_curr_params, {},
                #     genn_model.init_connectivity("FixedNumberTotalWithReplacement", conn_spec))
                
                

                # Create weight parameters
                static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {}, syn_spec)

                syn_pop = self.simulation.model.add_synapse_population(source_genn_pop.name + "_" + target_genn_pop.name, matrix_type,
                                source_genn_pop, target_genn_pop,
                                static_synapse_init, exp_curr_init,
                                init_sparse_connectivity("FixedNumberTotalWithReplacement", conn_spec))

                # for v in syn_pop.pop.var:
                #     print(v)
                # syn_pop.pop.set_psvar_location("inSyn",genn_wrapper.VarLocation_HOST_DEVICE)
                # syn_pop.pop.set_in_syn_var_location(genn_wrapper.VarLocation_HOST_DEVICE)
                # syn_pop.pop.set_psvar_location('Isyn',genn_wrapper.VarLocation_HOST_DEVICE) 
                # syn_pop.pop.set_psvar_location("inSyn", genn_wrapper.VarLocation_HOST_DEVICE)
                self.genn_syns[source_genn_pop.name + "_" + target_genn_pop.name] = syn_pop
 
                # Add size of this allocation to total
                # self.simulation.extra_global_param_bytes += source_genn_pop.size * num_sub_rows * 2

                # Set max dendritic delay and span type
                syn_pop.max_dendritic_delay_timesteps = int(round(max_delay / self.simulation.params['dt']))
                # syn_pop.pop.set_max_dendritic_delay_timesteps(int(round(max_delay / self.simulation.params['dt'])))

                if self.simulation.params['procedural_connectivity']:
                    # syn_pop.pop.set_span_type(genn_wrapper.SynapseGroup.SpanType_PRESYNAPTIC)
                    syn_pop.num_threads_per_spike = self.simulation.params['num_threads_per_spike']
                    # syn_pop.pop.set_num_threads_per_spike(self.simulation.params['num_threads_per_spike'])    


def connect(simulation,
            target_area,
            source_area):
    """
    Connect two areas with each other.

    Parameters
    ----------
    simulation : Simulation instance
        Simulation simulating the network containing the two areas.
    target_area : Area instance
        Target area of the projection
    source_area : Area instance
        Source area of the projection
    """
    num_sub_rows = (simulation.params['num_threads_per_spike'] 
                    if simulation.params['procedural_connectivity'] 
                    else 1)

    matrix_type = "PROCEDURAL_PROCEDURALG" if simulation.params['procedural_connectivity']  else "SPARSE_INDIVIDUALG"

    network = simulation.network
    synapses = extract_area_dict(network.synapses,
                                 network.structure,
                                 target_area.name,
                                 source_area.name)
    # print("synapses=",synapses)
    W = extract_area_dict(network.W,
                          network.structure,
                          target_area.name,
                          source_area.name)
    W_sd = extract_area_dict(network.W_sd,
                             network.structure,
                             target_area.name,
                             source_area.name)

    for target in target_area.populations:
        for source in source_area.populations:
            num_connections = int(synapses[target][source])
            conn_spec = {"num": num_connections}

            # syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0}
            exp_curr_params = {}
            
            if target_area == source_area:
                max_delay = simulation.max_inter_area_delay
                if 'E' in source:
                    mean_delay = network.params['delay_params']['delay_e']
                else:
                    mean_delay = network.params['delay_params']['delay_i']
            else:
                max_delay = simulation.max_intra_area_delay
                v = network.params['delay_params']['interarea_speed']
                s = network.distances[target_area.name][source_area.name]
                mean_delay = s / v
            
            # if source[0] == 'E':
            #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['E']['tau_syn']})
            #     # syn_weight.update({})
            #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0,'min': 0., 'max': float(np.finfo(np.float32).max)}
            # if source[0] == 'V':
            #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['V']['tau_syn']})
            #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0,'min': float(-np.finfo(np.float32).max), 'max': 0.}
            # if source[0] == 'P':
            #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['P']['tau_syn']})
            #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0,'min': float(-np.finfo(np.float32).max), 'max': 0.}
            # if source[0] == 'S':
            #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['S']['tau_syn']})
            #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0,'min': float(-np.finfo(np.float32).max), 'max': 0.}
            # if source[0] == 'H':
            #     exp_curr_init = init_postsynaptic("ExpCurr", {"tau": network.params['neuron_params']['single_neuron_dict']['H']['tau_syn']})
            #     syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0,'min': float(-np.finfo(np.float32).max), 'max': 0.}
            
            # if target == "H1":
            #     print("source=",source)                
            #     print("syn_weight=",syn_weight)
            
            exp_curr_init = init_postsynaptic("ExpCurr", {"tau": 0.5})
            syn_weight = {"mean": W[target][source] / 1000.0, "sd": W_sd[target][source] / 1000.0,'min': float(-np.finfo(np.float32).max), 'max': float(np.finfo(np.float32).max)}


            
            delay_sd = mean_delay * network.params['delay_params']['delay_rel']

            
            syn_delay = {'min': simulation.params['dt'],
                         'max': max_delay,
                         'mean': mean_delay,
                         'sd': delay_sd}
            syn_spec = {'g': init_var("NormalClipped", syn_weight),
                        'd': init_var("NormalClippedDelay", syn_delay)}

            # Add synapse population
            source_genn_pop = source_area.genn_pops[source]
            target_genn_pop = target_area.genn_pops[target]
            
            static_synapse_init = init_weight_update("StaticPulseDendriticDelay", {}, syn_spec)

            syn_pop = simulation.model.add_synapse_population(source_genn_pop.name + "_" + target_genn_pop.name, matrix_type,
                                source_genn_pop, target_genn_pop,
                                static_synapse_init, exp_curr_init,
                                init_sparse_connectivity("FixedNumberTotalWithReplacement", conn_spec))
            
            # syn_pop = simulation.model.add_synapse_population(source_genn_pop.name + "_" + target_genn_pop.name, 
            #     matrix_type, 0,
            #     source_genn_pop, target_genn_pop,
            #     "StaticPulseDendriticDelay", {}, syn_spec, {}, {},
            #     "ExpCurr", exp_curr_params, {},
            #     genn_model.init_connectivity("FixedNumberTotalWithReplacement", conn_spec))

            # Add size of this allocation to total
            # simulation.extra_global_param_bytes += source_genn_pop.size * num_sub_rows * 2

            # Set max dendritic delay and span type
            # syn_pop.pop.set_max_dendritic_delay_timesteps(int(round(max_delay / simulation.params['dt'])))
            syn_pop.max_dendritic_delay_timesteps = int(round(max_delay / simulation.params['dt']))

            if simulation.params['procedural_connectivity']:
                # syn_pop.pop.set_span_type(genn_wrapper.SynapseGroup.SpanType_PRESYNAPTIC)
                syn_pop.num_threads_per_spike = simulation.params['num_threads_per_spike']
                # syn_pop.pop.set_num_threads_per_spike(simulation.params['num_threads_per_spike'])

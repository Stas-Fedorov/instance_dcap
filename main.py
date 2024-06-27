# -*- coding: utf-8 -*-

import os
import json
import logging
import numpy as np
import random
# Importing supportive libs
from simulator.district_generator import DistrictNetwork
from simulator.instanceDcap import InstanceDCAP
from simulator.zoneNetwork import ZoneNetwork
# Fixing seed to debug the possible problems
np.random.seed(42)
random.seed(42)

"""
Main launch test of DCAP

"""

if __name__ == '__main__':
    log_name = "logs/maindcap.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    # Secure the working dir
    os.chdir(os.path.dirname(__file__))  
    # Instance configuration load
    fp = open("./cfg/test_instance.json", 'r')
    cfg_setting = json.load(fp)
    
    
    # Input parameters
    # Number of zones on one side of the grid (test version)    
    n_nodes = 5 # corresponds to x^2 zones (5^2 = 25)
    # Number of districts to form from zones (+ 1 depot)
    n_districts = 15
    # Number of scnarios in the main DCAP problem
    n_scenarios = 10
    
    print("Starting...")
    # Generating instances
    zone_graph = ZoneNetwork(n_nodes)
    district_graph = DistrictNetwork(n_districts)
    # Choose one of the starting districting solutions:
    # Random districting
    district_graph.dummy_generator(zone_graph)
    # Deterministic districting
    #district_graph.dummy_generator_deterministic(zone_graph)
    # K_Means districting
    #district_graph.k_means_generator(zone_graph)
    
    # Instance generation    
    inst = InstanceDCAP(cfg_setting, district_graph, n_scenarios = n_scenarios)
    
    # Plot the district graph
    zone_graph.plot('Zones_graph')
    district_graph.plot('Districts_graph_random_new')
    district_graph.plot_districts('Districts_zones_graph_random_new')    
    
    # Get the parameters as dictionary
    inst_dict = inst.get_dict()
    print(inst_dict)
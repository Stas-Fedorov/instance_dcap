# -*- coding: utf-8 -*-
import logging
import numpy as np
from itertools import permutations
import random
import copy
"""
DCAP instance generation.
"""

class InstanceDCAP(): 
    
    def __init__(self, cfg_setting, district_graph, n_scenarios = None):
        
        # Loading graph
        self.district_graph = district_graph
        self.districts = district_graph.nodes
        self.n_districts = len(district_graph.nodes)
        # Distance matrix should be without 0 node
        self.distance_matrix = district_graph.distance_matrix
        self.barycenters = district_graph.pos
        # Arcs between the NON-ADJACENT districts
        self.complementary_set_arcs = district_graph.edges_complement
        
        # Loading data
        self.inst_name = cfg_setting['Instance_name']
        self.cost_extra_time = cfg_setting['cost_extra_time_h']
        self.cost_pass_one_zone_kg = cfg_setting['cost_pass_one_zone_kg']
        self.max_extra_capacity_distr_kg = cfg_setting['max_extra_capacity_distr_kg']
        self.max_extra_working_time_h = cfg_setting['max_extra_working_time_h']
        self.base_zone_capacity = cfg_setting['base_zone_capacity']
        self.time_pass_city_h = cfg_setting['time_pass_city_h'] 
        self.cost_extra_capacity_kg = cfg_setting['cost_extra_capacity_kg']
        self.daily_working_time = cfg_setting["daily_working_time"]
        logging.info("starting simulation...")
        
        # Generate extra capacity cost per district (future lenght from hub)
        self.cost_extra_capacity_kg = (
            self.cost_extra_capacity_kg  * 
            np.random.uniform(
                low= 0.9, 
                high = 1.1, 
                size = self.n_districts
                ) 
        )
        # Capacities of vehicles
        self.capacities = []
        self.costs = []
        for v in cfg_setting['vehicles']:
            for i in range(v['number']):
                self.capacities.append(v['capacity'])
                # Generating costs
                self.costs.append(v['cost']*random.uniform(0.5,1.5))
        # Vehicle total number
        self.n_vehicles = np.size(self.capacities)
        
        # Generate costs of the driver working time
        self.cost_driver_time_h = (
            cfg_setting['cost_driver_time_h'] *
            np.random.uniform(
            low= 0.9, 
            high = 1.1, 
            size = self.n_vehicles
            ) 
        )     
        # Cost to move between districts
        self.cost_move = np.ones([np.size(self.capacities), 
                                  self.n_districts, 
                                  self.n_districts]
                    )
        # Distance matrix should be without 0 node
        for v in range(np.size(self.capacities)):
            for i,j in permutations(self.districts,2):
                if i != 0 and j != 0:
                    self.cost_move[v,i,j] = (
                            self.cost_move[v,i,j]*
                            self.capacities[v]*
                            self.cost_pass_one_zone_kg*
                            self.distance_matrix[i-1,j-1]
                            )
                else:
                    self.cost_move[v,i,j] = (
                            self.cost_move[v,i,j]*
                            self.capacities[v]*
                            self.cost_pass_one_zone_kg
                            )
        # Starting dynamic parameters      
        if n_scenarios:
            self.n_scenarios = n_scenarios
        else:
            self.n_scenarios = cfg_setting['n_scenarios']
        
        # Preparing the demand distribution for zones (chi squared)
        # self.demand_zones = (np.random.chisquare(0.155, size = np.size(
        #       self.district_graph.zone_graph.nodes)))
        # Preparing the demand distribution for zones (normal)
        self.demand_zones = (np.random.normal(loc = 8, scale = 1.5, size = np.size(
              self.district_graph.zone_graph.nodes)))
        # Re-scaling and taking the base as the mean
        self.demand_zones = self.demand_zones * (1/np.max(self.demand_zones))
        self.demand_zones = self.demand_zones * (1/np.mean(self.demand_zones))
        self.demand_zones = self.demand_zones * self.base_zone_capacity * (1/np.size(
              self.district_graph.zone_graph.nodes))
        # Filtering outliers
        for z in range(np.size(
              self.district_graph.zone_graph.nodes)):
            if self.demand_zones[z] > np.max(self.capacities) * 0.7:
                self.demand_zones[z] = np.max(self.capacities)/2
        # Pass to scenarios
        self._generate_dynamic_params()
            
    def _generate_dynamic_params(self):
        # Generating volumes (kg) and time spend (h) INSIDE the district
        self.time_spent = np.zeros([self.n_districts, self.n_scenarios])
        self.district_volumes = np.zeros([self.n_districts, self.n_scenarios])
        self.zone_volumes = []
        for s in range(self.n_scenarios):
            # Generate zones volumes
            self.zone_volumes.append(self.demand_zones *
                                     np.random.uniform(
                            low = 0.5, 
                            high = 1.5, 
                            size = np.size(self.demand_zones)
                              )
                )
        for s in range(self.n_scenarios):
            for d in range(1,self.n_districts ):
                self.district_volumes[d,s] = 0
                for z in self.district_graph.districts[d]:
                    self.district_volumes[d,s] += self.zone_volumes[s][z]
            # Generation times through TSP for EACH district
            times = self.district_graph.simulate_pickup_times(
                rand_base = self.time_pass_city_h
                )
            # Check if district contained one zone only
            for t in range(np.size(times)):
                if times[t] == 0:
                    times[t] = (0.25 *
                                np.random.uniform(low = 0.9, 
                                                  high = 1.1, 
                                                  size = 1)
                                                )
            self.time_spent[:,s] = times
        # Compensate the 0 depot node
        self.district_volumes[0,:] = np.zeros(self.n_scenarios)
        self.time_spent[0,:] = np.zeros(self.n_scenarios)
        # Travel time between districts
        self.trav_time_between_districts = np.zeros([
                                    self.n_districts, 
                                    self.n_districts, 
                                    self.n_scenarios])
        for s in range(self.n_scenarios):
            for i,j in permutations(self.districts,2):
                dist = (
                    abs(self.barycenters[i][0] - 
                        self.barycenters[j][0]
                        ) +
                    abs(self.barycenters[i][1] - 
                        self.barycenters[j][1]
                        ) 
                    )
                if i != 0 and j != 0:
                    self.trav_time_between_districts[i,j,s] = (
                        self.distance_matrix[i-1,j-1] *
                        self.time_pass_city_h *
                        (1/(self.n_districts -1)) * 
                        dist/len(self.district_graph.zones) *
                        np.random.uniform(
                            low = 0.9, 
                            high = 1.1, 
                            size = 1 )
                                  )
                else:
                    self.trav_time_between_districts[i,j,s] = (
                        self.time_pass_city_h **
                        (1/(self.n_districts -1)) * 
                        dist/len(self.district_graph.zones) *
                        np.random.uniform(
                            low = 0.9, 
                            high = 1.1, 
                            size = 1 )
                                  )    
        # Generate probabilites of scenarios
        self.probabilities = (1 / (self.n_scenarios) * 
                              np.ones(shape=(self.n_scenarios))
                              )
    
    def generate_out_sample(self, n_scenarios):
        # Generate out of sample data
        self.n_scenarios = n_scenarios
        # Increase the cost of additional capacity and time
        self.cost_extra_capacity_kg = (
            self.cost_extra_capacity_kg  * 
            1.02 *
            np.random.uniform(
                low= 0.85, 
                high = 1.15, 
                size = self.n_districts
                ) 
        )
        self.cost_extra_time = self.cost_extra_time * 1.02
        # Generate rest of dynamic parameters
        self._generate_dynamic_params()
        
        
    def change_graph(self, district_graph):
        # Change the instance with the new districts
        self.district_graph = district_graph
        self.districts = district_graph.nodes
        self.n_districts = len(district_graph.nodes)
        # Distance matrix should be without 0 node
        self.distance_matrix = district_graph.distance_matrix
        self.barycenters = district_graph.pos
        # Arcs between the NON-ADJACENT districts
        self.complementary_set_arcs = district_graph.edges_complement
        
        # Generate extra capacity cost per district (future lenght from hub)
        self.cost_extra_capacity_kg = (
            np.mean(self.cost_extra_capacity_kg) * 
            np.random.uniform(
                low= 0.9, 
                high = 1.1, 
                size = self.n_districts
                ) 
        )
        
        # Cost to move between districts
        self.cost_move = np.ones([np.size(self.capacities), 
                                  self.n_districts, 
                                  self.n_districts]
                    )
        # Distance matrix should be without 0 node
        for v in range(np.size(self.capacities)):
            for i,j in permutations(self.districts,2):
                if i != 0 and j != 0:
                    self.cost_move[v,i,j] = (
                            self.cost_move[v,i,j]*
                            self.capacities[v]*
                            self.cost_pass_one_zone_kg*
                            self.distance_matrix[i-1,j-1]
                            )
                else:
                    self.cost_move[v,i,j] = (
                            self.cost_move[v,i,j]*
                            self.capacities[v]*
                            self.cost_pass_one_zone_kg
                            )
        # Generating volumes (kg) and time spend (h) INSIDE the district
        self.time_spent = np.zeros([self.n_districts, self.n_scenarios])
        self.district_volumes = np.zeros([self.n_districts, self.n_scenarios])
        for s in range(self.n_scenarios):
            for d in range(1,self.n_districts ):
                self.district_volumes[d,s] = 0
                for z in self.district_graph.districts[d]:
                    self.district_volumes[d,s] += self.zone_volumes[s][z]
            # Generation times through TSP for EACH district
            times = self.district_graph.simulate_pickup_times(
                            rand_base = self.time_pass_city_h
                            )
            # Check if district contained one zone only
            for t in range(np.size(times)):
                if times[t] == 0:
                    times[t] = (0.25 *
                                np.random.uniform(low = 0.98, 
                                                  high = 1.02, 
                                                  size = 1)
                                                )
            self.time_spent[:,s] = times
        # Compensate the 0 depot node
        self.district_volumes[0,:] = np.zeros(self.n_scenarios)
        self.time_spent[0,:] = np.zeros(self.n_scenarios)
        # Travel time between districts
        self.trav_time_between_districts = np.zeros([self.n_districts, 
                                   self.n_districts, 
                                   self.n_scenarios])
        for s in range(self.n_scenarios):
            for i,j in permutations(self.districts,2):
                dist = (
                    abs(self.barycenters[i][0] - 
                        self.barycenters[j][0]
                        ) +
                    abs(self.barycenters[i][1] - 
                        self.barycenters[j][1]
                        ) 
                    )
                if i != 0 and j != 0:
                    self.trav_time_between_districts[i,j,s] = (
                        self.distance_matrix[i-1,j-1] *
                        (1/(self.n_districts -1)) * 
                        dist *
                        np.random.uniform(
                            low = 0.9, 
                            high = 1.1, 
                            size = 1 )
                                  )
                else:
                    self.trav_time_between_districts[i,j,s] = (
                        5*
                        (1/(self.n_districts -1)) * 
                        dist *
                        np.random.uniform(
                            low = 0.9, 
                            high = 1.1, 
                            size = 1 )
                                  )    
        # Generate probabilites of scenarios
        self.probabilities = (1 / (self.n_scenarios) * 
                              np.ones(shape=(self.n_scenarios))
                              )
        
    def get_dict(self):
        data_dcap = {
            "instance_name": self.inst_name,
            "n_districts": self.n_districts,
            "n_vehicles": self.n_vehicles,
            "costs": self.costs,
            "cost_extra_capacity_kg": self.cost_extra_capacity_kg, # x j district |cost of ext capacity
            "capacities": self.capacities,
            "cost_driver_time_h": self.cost_driver_time_h, # x i vehicle | cost of working time 
            "probabilities": self.probabilities,
            "district_volume": self.district_volumes,
            "time_spent": self.time_spent, # j x s | distr x scenarios | time spend to serve each district
            "cost_move": self.cost_move,  # i x j x j | vehicle x distr x distr | cost to move between districts
            "cost_extra_time": self.cost_extra_time,  # cost of the time overworked
            "daily_working_time": self.daily_working_time,
            "trav_time_between_districts": self.trav_time_between_districts,    #j x j x s| distr x distr x scen | travel time between districts
            "complementary_set_arcs" : self.complementary_set_arcs  # [m,n] | list of the arcs that does NOT exist
        }
        
        return data_dcap


    def change_demand_homology_realistic(self):
        # Change the homology of the demand distribution to Monopolar
        # Before usage is reccomended to re-generate a separate instance
        self.demand_zones = self.demand_zones * np.max(
            self.district_graph.zone_distance)/2
        center_node = int(np.floor(len(self.district_graph.zones)/2))
        for z in self.district_graph.zones:
            if z != center_node:
                self.demand_zones[z] = (
                    self.demand_zones[z] *
                    1/(self.district_graph.zone_distance[center_node,z])
                    )
        # Pass to scenarios
        self._generate_dynamic_params()
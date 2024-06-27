# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from itertools import permutations
from solver.tsp import tsp_solve
from sklearn.cluster import KMeans
"""
District graph generation and presentation
"""
class DistrictNetwork():
    
    def __init__(self, n_districts):
        self.n_districts = n_districts
        
    def dummy_generator(self, zone_graph):
        # -> DistrictNetwork (initial solution)
        # Generator of districts from zone graph
        self.zone_graph = zone_graph
        self.zones = self.zone_graph.nodes
        self.zone_distance = self.zone_graph.distance_matrix
        if len(self.zones) < self.n_districts:
            print("Number of zones higher than number of districts")
            return
        self.zones_edges = zone_graph.edges
        self.zones_available = list(self.zones).copy()
        ratio = (len(self.zones)/self.n_districts + 1)
        self.districts = {}
        self.districts[0] = ['D']
        for d in range(1, self.n_districts + 1):
            core = np.random.choice(self.zones_available)
            self.zones_available.remove(core)
            self.districts[d] = [core]
        while self.zones_available:
            for d in self.districts:
                if d != 0:
                    if (random.uniform(0, 1)> 0.5 
                        and np.size(self.districts[d]) < ratio
                        ):
                        self.districts[d] = self._add_neighbor(
                            self.districts[d]
                            )
                    elif random.uniform(0, 1)> 0.9:
                        self.districts[d] = self._add_neighbor(
                            self.districts[d]
                            )
        return self._construct_graph()
        
    def _construct_graph(self, print_map = False):
        if print_map:
            print("District map: [n_district: [n_nodes]]")
            print(self.districts)
        self.nodes = [0]
        self.pos = {}
        self.pos[0] = (-2,4)
        for d in self.districts:
            if d != 0:
                self.nodes.append(d)
                coord_list = [
                    self.zone_graph.pos[i] 
                    for i in self.zone_graph.pos 
                    if i in self.districts[d]
                    ]
                coord = np.mean(coord_list, axis = 0)
                try:
                    self.pos[d] = (coord[0],coord[1])
                except:
                    self.pos[d] = (0.1,0.1)
        # Check connectivity and generate edges
        self.edges_list = []
        for d_1, d_2 in permutations(range(1,self.n_districts + 1),2):
            self.edges_list = self._check_connectivity(d_1, d_2, self.edges_list)
        # Add the edges for the depot node?
        for d in range(1, self.n_districts + 1):
            self.edges_list.append((0,d))
            self.edges_list.append((d,0))
        # Construct graph from edgelist
        self.G = nx.from_edgelist(self.edges_list).to_directed()
        # Distance matrix should be without 0 depot node
        self.distance_matrix = nx.floyd_warshall_numpy(
            self.G.subgraph(range(1, self.n_districts + 1))
            )
        self.edges = self.G.edges()
        self.edges_complement = nx.complement(self.G).edges()
        return self
    
    def _check_connectivity(self, d_1, d_2, edges):
            flag = False
            for n_1 in self.districts[d_1]:
                for n_2 in self.districts[d_2]:
                    if (n_1, n_2) in self.zones_edges:
                        edges.append((d_1,d_2))
                        flag = True
                    if flag == True:
                        return edges
            return edges
        
    def _add_neighbor(self, district_list):
        flag = False
        for dis in district_list:
            neighbours = [
                j for i,j in self.zones_edges
                if i == dis and j in self.zones_available
                ]
            if neighbours != []:
                add_node = np.random.choice(neighbours)
                district_list.append(add_node)
                self.zones_available.remove(add_node)
                flag = True
            if flag == True:
                return district_list
        return district_list

    def plot(self, file_name: str = None):
        # Draw base structure of the graph
        fig = plt.figure(figsize=(5,4))
        nx.draw_networkx(self.G.subgraph(range(1, self.n_districts + 1)), pos=self.pos)
        plt.axis("off")
        if file_name:
            pfp = os.path.join('.', 'results', 'fig', file_name)
            fig.savefig(pfp)
        else:
            fig.show()
            
    def plot_districts(self, file_name: str = None):
        labels = {}
        D = self.zone_graph.G
        posD = self.zone_graph.pos
        fig = plt.figure(figsize=(5,4))
        for key, zones in self.districts.items():
            if key != 0:
                # Labeling district
                for n in zones:
                    labels[n] = f"{key}"
                # Coloring
                color = self.random_color_generator()
                nx.draw_networkx_nodes(
                    D, 
                    posD, 
                    nodelist= zones, 
                    node_color= color
                    )
        nx.draw_networkx_labels(
            D, 
            posD, 
            labels
            )
        nx.draw_networkx_edges(
            D,
            posD
            )
        plt.axis("off")
        if file_name:
            pfp = os.path.join('.', 'results', 'fig', file_name)
            fig.savefig(pfp)
        else:
            fig.show()
            
    def random_color_generator(self):
        color = random.choice(list(mcolors.TABLEAU_COLORS.keys()))
        return color
    
    def plot_assignment(self, assignment, file_name: str = None):
        options = {"node_size": 500, "alpha": 0.9}
        labels = {}
        fig = plt.figure(figsize=(5,4))
        for v in range(np.shape(assignment)[0]):
            route = [x for x in np.where(assignment[v]>0.5)[0]]
            color = self.random_color_generator()
            nx.draw_networkx_nodes(
                self.G, 
                self.pos, 
                nodelist=route, 
                node_color= color, 
                **options
                )
            # Labeling the drivers in the district
            for n in route:
                if n in labels:
                    labels[n] = labels[n] + f", {v}"
                else:
                    labels[n] = f"{v}"      
        nx.draw_networkx_labels(
            self.G, 
            self.pos, 
            labels, 
            font_size= 12
            )
        fig.tight_layout()
        plt.axis("off")
        if file_name:
            pfp = os.path.join('.', 'results', 'fig', file_name)
            fig.savefig(pfp)
        else:
            fig.show()

    def get_zone_assignment(self, assignment):
        # Transfer districts to zones
        zone_assignment = {}
        for i in range(np.shape(assignment)[0]):
            for j in range(np.shape(assignment)[1]):
                if assignment[i,j] > 0.5:
                    if i in zone_assignment:
                        for d in self.districts[j]:
                            zone_assignment[i].append(d)
                    else:
                        zone_assignment[i] = ['D']
                        for d in self.districts[j]:
                            zone_assignment[i].append(d)
        return zone_assignment
    
    
    def simulate_pickup(self, list_zones, 
                        scale_param = 1000,
                        rand_perc = 30,
                        rand_base = 2):
        # Construct distance matrix between the zones
        distance_matrix = np.zeros(
            [len(list_zones), len(list_zones)], 
            dtype = int
        )
        for i,j in permutations(range(1, len(list_zones)),2):
            # With OR tools distance matrix should be integers
            distance_matrix[i,j] = int(
                np.ceil(
                self.zone_distance[
                    list_zones[i],
                    list_zones[j]
                    ] * 
                (
                    1/np.max(self.zone_distance)
                    ) *
                scale_param *
                random.uniform(
                    rand_base - rand_perc/100, 
                    rand_base + rand_perc/100
                    )
                )
                )
            if (list_zones[i], list_zones[j]) in self.zones_edges:
                pass
            else:
                distance_matrix[i,j] = distance_matrix[i,j]*2
        return {'distance_matrix': distance_matrix,
                'num_vehicles': 1,
                'depot': 0
                }
            
    def simulate_pickup_times(self, 
                              scale_param = 1000, 
                              rand_perc = 30,
                              rand_base = 2):
        # Generates routing times inside each district
        timings = []
        for district in self.nodes:
            list_zones = ['D']
            # Get the list of zones of the district considered
            for z in self.districts[district]:
                list_zones.append(z)
                
            data = self.simulate_pickup(
                list_zones,
                scale_param = scale_param,
                rand_perc = rand_perc,
                rand_base = rand_base
            )
            timings.append(
                tsp_solve(data,
                          0,
                          list_zones,
                          scale_param = scale_param,
                          return_sol = 'OF', 
                          print_sol = False
                          )
                )
        return timings
            
            
    def dummy_generator_deterministic(self, zone_graph): 
        # -> DistrictNetwork (initial solution)
        # Generator of districts from zone graph with less randomness
        self.zone_graph = zone_graph
        self.zones = self.zone_graph.nodes
        self.zone_distance = self.zone_graph.distance_matrix
        if len(self.zones) < self.n_districts:
            print("Number of zones higher than number of districts")
            return
        self.zones_edges = self.zone_graph.edges
        self.zones_available = list(self.zones).copy()
        ratio = (len(self.zones)/self.n_districts + 1)
        self.districts = {}
        self.districts[0] = ['D']
        count = 1
        for d in range(1, self.n_districts + 1):
            if count in self.zones_available:
                core = count
            else:
                core = np.random.choice(self.zones_available)
            self.zones_available.remove(core)
            self.districts[d] = [core]
            count += int(np.floor(ratio))
        while self.zones_available:
            for d in self.districts:
                if d != 0:
                    self.districts[d] = self._add_neighbor(
                        self.districts[d]
                        )
        return self._construct_graph()            
            
    def redefine_districts(self, districts):
        # Only after first generation
        self.districts = districts
        return self._construct_graph()
            
    
    def k_means_generator(self, zone_graph):
        # -> DistrictNetwork (initial solution)
        # Generator of districts with K Means clustering
        self.zone_graph = zone_graph
        self.zones = self.zone_graph.nodes
        self.zone_distance = self.zone_graph.distance_matrix
        self.zones_edges = self.zone_graph.edges
        if len(self.zones) < self.n_districts:
            print("Number of zones higher than number of districts")
            return
        self.districts = {}
        self.districts[0] = ['D']
        kmeans = KMeans(n_clusters=self.n_districts)
        coord = [[x,y] for x,y in zone_graph.pos.values()]
        kmeans.fit(coord)
        self.labels = kmeans.labels_
        for d in range(1, self.n_districts + 1):
            for zone in self.zones:
                if self.labels[zone] + 1 ==d:
                    if self.districts.get(d)==None:
                        self.districts[d] = [zone]
                    else:
                        self.districts[d].append(zone)
                                    
        return self._construct_graph()
            
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
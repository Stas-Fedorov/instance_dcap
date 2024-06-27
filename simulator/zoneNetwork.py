# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
"""
Zones graph generation and presentation 
"""

class ZoneNetwork():
        
    def __init__(self, n_nodes):
        # Generate 2d positions of the nodes
        self.nodes = []
        self.pos = {}
        count = 0
        for i in range(n_nodes):
            for j in range(n_nodes):
                self.nodes.append(count)
                self.pos[count] = (i,j)
                count += 1
        self.G = nx.convert_node_labels_to_integers(
                nx.grid_graph(dim=[n_nodes, n_nodes])
        ).to_directed()
        self.distance_matrix = nx.floyd_warshall_numpy(self.G)
        self.edges = self.G.edges()
        self.edges_complement = nx.complement(self.G).edges()
        
    def plot(self, file_name: str = None):
        # Draw base structure of the graph
        fig = plt.figure(figsize=(20,16))
        nx.draw_networkx(self.G, pos=self.pos)
        plt.axis("off")
        if file_name:
            pfp = os.path.join('.', 'results', 'fig', file_name)
            fig.savefig(pfp)
        else:
            fig.show()

    
    def random_color_generator(self):
        color = random.choice(list(mcolors.TABLEAU_COLORS.keys()))
        return color
    
    def plot_routing(self, sol_tsp, district_graph, file_name: str = None):
        # Draw routing
        options = {"node_size": 300, "alpha": 0.6}
        labels = {}
        # Add depot node
        D = self.G.copy()
        D.add_node('D')
        for j in D.nodes():
            if j != 'D':
                D.add_edge('D', j)
                D.add_edge(j, 'D')
        posD = self.pos.copy()
        posD['D'] = (-2,4)
        
        fig = plt.figure(figsize=(5,4))
        for key, zones in district_graph.districts.items():
            if key != 0:
                color = self.random_color_generator()
                nx.draw_networkx_nodes(
                    D, 
                    posD, 
                    nodelist= zones, 
                    node_color= color, 
                    **options
                    )
        for key, route in sol_tsp.items():
            color = self.random_color_generator()
            # Labeling the driver nums in the districts
            for n in route:
                if n in labels:
                    labels[n] = labels[n] + f", {key}"
                else:
                    labels[n] = f"{key}"
            tour = []
            for i in range(len(route[:-1])):
                tour.append((route[i],route[i+1]))                    
            nx.draw_networkx_edges(
                        D,
                        posD,
                        edgelist= tour,
                        width=5,
                        alpha=0.8,
                        edge_color= color
                        )
        nx.draw_networkx_nodes(
            D, 
            posD, 
            nodelist= ['D'], 
            node_color= 'Grey', 
            **options
            )
        labels['D'] = 'D'
        nx.draw_networkx_labels(
            D, 
            posD, 
            labels, 
            font_size= 12 
            )
        plt.axis("off")
        if file_name:
            pfp = os.path.join('.', 'results', 'fig', file_name)
            fig.savefig(pfp)
        else:
            fig.show()

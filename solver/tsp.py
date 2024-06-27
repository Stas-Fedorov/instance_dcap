# -*- coding: utf-8 -*-
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
"""Simple Travelling Salesperson Problem (TSP) between cities."""


def tsp_solve(data, vehicle, list_zones, scale_param = 1000, 
              return_sol = 'list', print_sol = True):
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']),
        data['num_vehicles'], 
        data['depot']
        )
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    
    distance_matrix = data['distance_matrix']
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
        )
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if print_sol:
        print_solution(
            manager, routing, solution, 
            vehicle, list_zones, scale_param
            )
    
    if return_sol == 'dict':
        # Pass the solution to dict
        return dict_solution(
            manager, routing, solution, 
            vehicle, list_zones, scale_param
            )
    elif return_sol == 'list':
        # Pass the solution to list
        return list_solution(
            manager, routing, solution, 
            list_zones
            )
        
    elif return_sol == 'OF':
        # Pass the solution to list
        return solution.ObjectiveValue()/scale_param
    else:
        print('Wrong param for return_sol')
        return
    
def print_solution(
        manager, routing, solution, 
        vehicle, list_zones, scale_param):
    """Prints solution on console."""
    index = routing.Start(0)
    plan_output = f'Route for vehicle {vehicle}:\n'
    print('Objective time: {} hours'.format(solution.ObjectiveValue()/scale_param))
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(list_zones[manager.IndexToNode(index)])
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(list_zones[manager.IndexToNode(index)])
    print(plan_output)
    plan_output += 'Route distance: {} hours \n'.format(route_distance)
    
def dict_solution(
        manager, routing, solution, 
        vehicle, list_zones, scale_param):
    """Returns solution as dict"""
    sol = {}
    index = routing.Start(0)
    sol['vehicle'] = vehicle
    sol['route'] = []
    route_distance = 0
    while not routing.IsEnd(index):
        sol['route'].append(list_zones[manager.IndexToNode(index)])
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    sol['route'].append(list_zones[manager.IndexToNode(index)])
    sol['objective'] = route_distance/scale_param
    return sol

def list_solution(
        manager, routing, solution, 
        list_zones):
    """Returns solution as list"""
    sol = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        sol.append(list_zones[manager.IndexToNode(index)])
        index = solution.Value(routing.NextVar(index))
    sol.append(list_zones[manager.IndexToNode(index)])
    return sol








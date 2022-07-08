from numpy import true_divide
from utils import *
from aima.utils import *
from aima import search 
from part1 import makeGraph
import os
import csv
import time



from multiprocessing import parent_process


def minKey(key, mstSet, node):
    min = float('inf')
    min_index = None
    for v in node.state.unvisited:
        # only deals with stuff that hasn't been put in the mst yet
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v
    return min_index

def MST(node, graph):
    cost = 0
    key = [float('inf')] * len(graph[0])
    key[node.state.curr_val] = 0
    mstSet = [False] * len(graph[0])
    parent = [None] * len(graph[0])
    parent[node.state.curr_val] = -1
    if node.state.unvisited != None:

        for cout in node.state.unvisited:
            u = minKey(key, mstSet, node)
            mstSet[u] = True

            for v in node.state.unvisited:
                if graph[u][v] > 0 and mstSet[v] == False and key[v] > graph[u][v]:
                    key[v] = graph[u][v]
                    parent[v] = u

        for i in node.state.unvisited:
            if i != node.state.curr_val:
                cost += graph[parent[i]][i]
    return cost

        




class State:
    def __init__(self, unvisited, path_cost=0, curr_val=0):
        self.unvisited = unvisited
        self.curr_val = curr_val
        self.path_cost = path_cost

    def __repr__(self):
        return "<State {}, {}>".format(self.curr_val, self.unvisited)


# Actions are unvisited nodes
# Initial is unvisited and the current value of start?- it's the initial state of everything that is left
class Problem_Custom(search.Problem):
    def __init__(self, initial, graph):
        self.initial = initial
        self.goal = 0
        self.graph = graph
        self.expanded = 0

    def actions(self, state):
        new_unvisited = state.unvisited.copy()
        new_unvisited.remove(state.curr_val)
        return new_unvisited

    def result(self, state, action):
        new_unvisited = state.unvisited.copy()
        new_unvisited.remove(state.curr_val)
        new_state = State(new_unvisited, (state.path_cost + self.graph[state.curr_val][action]), action)
        if len(new_state.unvisited) == 1:
            if new_state.curr_val == self.initial.curr_val:
                new_state.unvisited.remove(new_state.curr_val)
            else: 
                new_state.unvisited.append(self.initial.curr_val)
        return new_state

# I AM EDITTING THIS AT SOME POINT IN TIME
    def goal_test(self, state):
        if state.unvisited == None:
            return True
        else:
            return len(state.unvisited) == self.goal


    def path_cost(self, c, state1, action, state2):
        return c + self.graph[state1.curr_val][state2.curr_val]

    def h(self, node):
        self.expanded += 1
        return MST(node, self.graph)

    def value(self, state):
        return state.path_cost


class Node():
    #action = unvisited
    def __init__(self, state, parent=None, action = None, path_cost = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
        self.curr_val = state.curr_val
        self.path_cost = path_cost

    def __repr__(self):
        return "<Node {}>".format(self.state)
    
    def expand(self,problem):
        return [self.child_node(problem, action) 
            for action in problem.actions(self.state)]
    
    def child_node (self, problem, action):
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node
    
    def solution(self):
        return [node.action for node in self.path()[1:]]

    def path(self):
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state.curr_val == other.state.curr_val and self.state.unvisited == other.state.unvisited and self.path_cost == other.path_cost

    def __has__(self):
        return hash(self.state)


def RunProgram(graph_list):
    ave_cost = 0
    ave_cpu = 0
    ave_real = 0
    ave_exp = 0

    min_cost = float('inf')
    min_cpu = float('inf')
    min_real = float('inf')
    min_exp = float('inf')

    max_cost = 0
    max_cpu = 0
    max_real = 0
    max_exp = 0
    size = len(graph_list[0])
    unvisited = list(range(size))
    counter = 0
    for g in graph_list:
        counter += 1
        print ("Graph Size: {} \nGraph Number: {}".format(size, counter))
        start_state = State(unvisited)
        tsp = Problem_Custom(start_state, g)
        real_start = time.time()
        cpu_start = time.process_time()
        r = search.astar_search(tsp)
        cpu_time = time.process_time() - cpu_start
        real_time = time.time() - real_start
        if (r.path_cost < min_cost):
            min_cost = r.path_cost
        if cpu_time < min_cpu:
            min_cpu = cpu_time
        if real_time < min_real:
            min_real = real_time
        if tsp.expanded < min_exp:
            min_exp = tsp.expanded
        if (r.path_cost > max_cost):
            max_cost = r.path_cost
        if cpu_time > max_cpu:
            max_cpu = cpu_time
        if real_time > max_real:
            max_real = real_time
        if tsp.expanded > max_exp:
            max_exp = tsp.expanded
        ave_cost += r.path_cost
        ave_cpu += cpu_time
        ave_real += real_time
        ave_exp += tsp.expanded

    tot = len(graph_list)
    ave_cost = ave_cost/tot
    ave_cpu = ave_cpu/tot
    ave_real = ave_real/tot
    ave_exp = ave_exp/tot
    results = []
    results.append(size)
    results.append(ave_cost)
    results.append(min_cost)
    results.append(max_cost)
    results.append(ave_cpu)
    results.append(min_cpu)
    results.append(max_cpu)
    results.append(ave_real)
    results.append(min_real) 
    results.append(max_real)
    results.append(ave_exp)
    results.append(min_exp)
    results.append(max_exp)
    print(size)
    return results




        
'''
my_csv = open("Part1_data.csv", 'a', newline='')
write = csv.writer(my_csv)
my_list = os.listdir('infiles')
graph_list = []
write.writerow(['A* MST'])
headers = ['Nodes', 'Ave Cost', 'Min Cost', 'Max Cost','Ave CPU Time', 'Min CPU Time', 'Max CPU Time', 'Ave Real Time', 'Min Real Time', 'Max Real Time', 'Ave Expanded', 'Min Expanded', 'Max Expanded']
write.writerow(headers)
for folder in my_list:
    file_list = os.listdir("infiles/" + folder)
    for file in file_list:
        graph_list.append(makeGraph("infiles/" + folder + "/"+ file))
    
    write.writerow(RunProgram(graph_list))
    graph_list.clear()


'''
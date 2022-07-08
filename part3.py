from utils import *
from aima.utils import *
from aima import search 
import os
import csv
import time
from part1 import makeGraph
from part2 import Problem_Custom
from part2 import State
import numpy as np
import sys


#redefine problem and nodes and state so that each state is /a/ solution and neighboring solutions are switching around two cities


class State:
    def __init__(self, path, path_cost):
        self.path = path
        self.path_cost = path_cost

    def __repr__(self):
        return "<State {}, {}>".format(self.path, self.path_cost)

class Node():
    #action = unvisited
    def __init__(self, state, path_cost = 0, parent=None, action = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0

        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)
    
    def expand(self,problem):
        return [self.child_node(problem, action) 
            for action in problem.actions(self.state)]
    
    def child_node (self, problem, action):
        curr_cost = 0
        path = problem.result(self.state, action).path
        for i in range(problem.size - 1):
            curr_cost += problem.graph[path[i]][path[i+1]]

        curr_cost += problem.graph[path[i]][path[i+1]]
        state = State(path, curr_cost)
        next_node = Node(state, curr_cost)
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
        return isinstance(other, Node) and self.state == other.state

    def __has__(self):
        return hash(self.state)


class Problem:

    def __init__(self, initial, graph, size):
        self.initial = initial
        self.size = size
        self.graph = graph
        # self.goal = goal

    def actions(self, state):
        action_list = []
        for i in range(self.size):
           for j in range(i, self.size):
               action_list.append([state.path[i],state.path[j]])
        return action_list

    def result(self, state, action):
        path = state.path
        new_path = path.copy()
        new_path[action[0]] = state.path[action[1]]
        new_path[action[1]] = state.path[action[0]]
        cost = find_cost(self.graph, new_path, self.size)
        new_state = State(new_path, cost)
        return new_state

    def goal_test(self, state):
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1

    def value(self, state):
        return -(state.path_cost)


def hill_climbing(problem):
    """
    [Figure 4.2]
    From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better.
    """
    current = Node(problem.initial)
    while True:
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmax_random_tie(neighbors, key=lambda node: problem.value(node.state))
        if problem.value(neighbor.state) <= problem.value(current.state):
            break
        current = neighbor
    return current.state


def simulated_annealing(problem, schedule=search.exp_schedule()):
    """[Figure 4.5] CAUTION: This differs from the pseudocode as it
    returns a state instead of a Node."""
    current = Node(problem.initial)
    for t in range(sys.maxsize):
        T = schedule(t)
        if T == 0:
            return current.state
        neighbors = current.expand(problem)
        if not neighbors:
            return current.state
        next_choice = random.choice(neighbors)
        delta_e = problem.value(next_choice.state) - problem.value(current.state)
        if delta_e > 0 or probability(np.exp(delta_e / T)):
            current = next_choice

def find_cost(g, path, size):
    cost = 0
    for i in range(size - 1):
        cost += g[path[i]][path[i+1]]
    cost += g[path[0]][path[size - 1]]
    return cost



def run_hill(graph_list):
    size = len(graph_list[0])
    tot_results = []
    counter = 1
    for g in graph_list:
        curr_path = (np.random.permutation(size)).tolist()
        cost = find_cost(g, curr_path, size)
        start_state = State(curr_path, cost)
        tsp = Problem(start_state, g, size)

        real_start = time.time()
        r = hill_climbing(tsp)
        real_time = time.time() - real_start
        results = []
        results.append(counter)
        results.append(real_time)
        results.append(r.path_cost)
        counter += 1

        tot_results.append(results)
    return tot_results

def run_simulated_annealing(graph_list):
    size = len(graph_list[0])
    tot_results = []
    counter = 1
    for g in graph_list:
        curr_path = (np.random.permutation(size)).tolist()
        cost = find_cost(g, curr_path, size)
        start_state = State(curr_path, cost)
        tsp = Problem(start_state, g, size)

        real_start = time.time()
        r = simulated_annealing(tsp)
        real_time = time.time() - real_start
        results = []
        results.append(counter)
        results.append(real_time)
        results.append(r.path_cost)
        counter += 1

        tot_results.append(results)
    return tot_results

def run_genetic_algorithm(graph_list):
    size = len(graph_list[0])
    tot_results = []
    counter = 1
    for g in graph_list:
        cities = range(size)
        ''' matrix = []
        for i in range(size):
            for j in range(size):
                matrix[i,j] = g[i][j]'''
        np_matrix = np.array(g)
        real_start = time.time()
        best = genetic_algorithm(cities, np_matrix, verbose=False)
        real_time = time.time() - real_start
        path_cost = find_cost(g, best, size)
        results = []
        results.append(counter)
        results.append(real_time)
        results.append(path_cost)
        counter += 1

        tot_results.append(results)
    return tot_results

def swap(chromosome):
    a, b = np.random.choice(len(chromosome), 2)
    chromosome[a], chromosome[b] = (
        chromosome[b],
        chromosome[a],
    )
    return chromosome

class Population():
    def __init__(self, bag, adjacency_mat):
        self.bag = bag
        self.parents = []
        self.score = 0
        self.best = None
        self.adjacency_mat = adjacency_mat

    def init_population(cities, adjacency_mat, n_population):
        return Population(
            np.asarray([np.random.permutation(cities) for _ in range(n_population)]), 
            adjacency_mat
        )

    def fitness(self, chromosome):
        cost = 0
        for i in range(len(chromosome) -1):
            cost += self.adjacency_mat[chromosome[i]][chromosome[i+1]]

        cost += self.adjacency_mat[chromosome[0]][chromosome[len(chromosome) -1]]
        return cost

    def evaluate(self):
        distances = np.asarray(
            [self.fitness(chromosome) for chromosome in self.bag]
        )
        self.score = np.min(distances)
        self.best = self.bag[distances.tolist().index(self.score)]
        self.parents.append(self.best)
        if False in (distances[0] == distances):
            distances = np.max(distances) - distances
        return distances / np.sum(distances)

    def select(self, k=4):
        fit = self.evaluate()
        while len(self.parents) < k:
            idx = np.random.randint(0, len(fit))
            if fit[idx] > np.random.rand():
                self.parents.append(self.bag[idx])
        self.parents = np.asarray(self.parents)
    
    def crossover(self, p_cross=0.1):
        children = []
        count, size = self.parents.shape
        for _ in range(len(self.bag)):
            if np.random.rand() > p_cross:
                children.append(
                    list(self.parents[np.random.randint(count, size=1)[0]])
                )
            else:
                parent1, parent2 = self.parents[
                    np.random.randint(count, size=2), :
                ]
                idx = np.random.choice(range(size), size=2, replace=False)
                start, end = min(idx), max(idx)
                child = [None] * size
                for i in range(start, end + 1, 1):
                    child[i] = parent1[i]
                pointer = 0
                for i in range(size):
                    if child[i] is None:
                        while parent2[pointer] in child:
                            pointer += 1
                        child[i] = parent2[pointer]
                children.append(child)
        return children

    def mutate(self, p_cross=0.1, p_mut=0.1):
        next_bag = []
        children = self.crossover(p_cross)
        for child in children:
            if np.random.rand() < p_mut:
                next_bag.append(swap(child))
            else:
                next_bag.append(child)
        return next_bag


def genetic_algorithm(cities, adjacency_mat, n_population=5, n_iter=20, selectivity=0.15, p_cross=0.5, p_mut=0.1, print_interval=100, 
    return_history=False, verbose=False
):

    pop = Population.init_population(cities, adjacency_mat, n_population)
    best = pop.best
    score = float("inf")
    history = []
    for i in range(n_iter):
        pop.select(n_population * selectivity)
        history.append(pop.score)
        if verbose:
            print(f"Generation {i}: {pop.score}")
        elif i % print_interval == 0:
            print(f"Generation {i}: {pop.score}")
        if pop.score < score:
            best = pop.best
            score = pop.score
        children = pop.mutate(p_cross, p_mut)
        pop = Population(children, pop.adjacency_mat)
    if return_history:
        return best, history
    return best



my_csv = open("Part1_data.csv", 'a', newline='')
write = csv.writer(my_csv)
my_list = os.listdir('infiles/size_15')
graph_list = []
write.writerow([])

for file in my_list:
    graph_list.append(makeGraph("infiles/size_15/" + file))
    
headers = ['Graph Number', 'Real Time', 'Path Cost']
write.writerow(['Hill Climbing'])

write.writerow(headers)
hill = run_hill(graph_list)
for row in hill:
    write.writerow(row)

simuAnnealing = run_simulated_annealing(graph_list)
write.writerow(['Simulated Annealing'])
write.writerow(headers)
for row in simuAnnealing:
    write.writerow(row)

genAlgo = run_genetic_algorithm(graph_list)
write.writerow(['Genetic Algorithm'])
write.writerow(headers)
for row in genAlgo:
    write.writerow(row)

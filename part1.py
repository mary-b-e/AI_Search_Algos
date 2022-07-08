import random
import time
import os
import csv


def makeGraph(filename):
    file = open(filename, 'r')
    Lines = file.readlines()
    count = 0
    matrix = []
    size = int(Lines[0])
    for line in Lines:
        if count != 0:
            temp = [] 
            row = line.split(' ')
            for j in range(size):
                temp.append(int(row[j]))
            matrix.append(temp)
        count += 1
    return matrix



def runNN (graph, size):
    cpu_start = time.process_time()
    real_start = time.time()

    unvisited = list(range(size))
    curr_expanded = 0
    currNode = 0
    cost = 0
    path = []
    results = []

    unvisited.remove(currNode)
    path.append(currNode)

    while len(unvisited) != 0:
        minNode = -1
        minCost = float('inf')

        for node in unvisited:
            if graph[currNode][node] < minCost:
                minCost = graph[currNode][node]
                minNode = node
        curr_expanded += 1
        currNode = minNode
        unvisited.remove(currNode)
        path.append(currNode)
        cost += minCost
    
    path.append(0)
    cost += graph[currNode][0]
    real_time = time.time() - real_start
    cpu_time = time.process_time() - cpu_start
    results = {}
    results["cost"] = cost
    results["path"] = path
    results["cpu"] = cpu_time
    results["real"] = real_time
    results["expanded"] = curr_expanded
    return results

def runTwoOpt(graph, size):
    cpu_start = time.process_time()
    real_start = time.time()
    r = runNN(graph, size)
    newR = twoOpt(graph, size, r)
    cpu_time = time.process_time() - cpu_start
    real_time = time.time() - real_start
    newR["cpu"] = cpu_time
    newR["real"] = real_time
    return newR


def twoOpt (graph, size, results):
    
   
    path = results["path"]
    cost = results["cost"]
    expanded = results["expanded"]

    improved = True
    while improved:
        improved = False
        for i in range(size-3):
            for j in range(i+2, size - 1):
                beg1 = path[i]
                beg2 = path[j]
                end1 = path[i+1]
                end2 = path[j+1]

                if beg1 != end2:
                    tempCost = cost - graph[beg1][end1] - graph[beg2][end2]
                    tempCost = tempCost + graph[beg1][beg2] + graph[end1][end2]
                    if tempCost < cost:
                        for k in range(int((j-i)/2) + (j-i)%2):
                            temp = path[k + i + 1]
                            path[i + k + 1] = path[j - k]
                            path[j - k] = temp
                        improved = True
                        expanded += 1
                        cost = tempCost
                        
                        break
            if improved:
                break
    results = {}
    results["cost"] = cost
    results["path"] = path
    results["expanded"] = expanded
    
    return results

def randomizedNN(graph, size, start):
    unvisited = list(range(size))
    unvisited.remove(start)
    path = []
    path.append(start)
    currNode = start
    cost = 0
    expanded = 0
    while len(unvisited) != 0:
        nsmallest = [[-1, float('inf')], [-1,float('inf')]]
        if len(unvisited) > 2:
            for node in unvisited:
                currCost = graph[currNode][node]
                if currCost < nsmallest[1][1]:
                    nsmallest[1][0] = node
                    nsmallest[1][1] = currCost
                    if currCost < nsmallest[0][1]:
                        nsmallest[1][0] = nsmallest[0][0]
                        nsmallest[1][1] = nsmallest[0][1]
                        nsmallest[0][0] = node
                        nsmallest[0][1] = currCost
            nextNode = nsmallest[random.randint(0,1)][0]
        else:
            nextNode = unvisited[random.randint(0,len(unvisited) - 1)]
        expanded += 1
        cost += graph[currNode][nextNode]
        unvisited.remove(nextNode)
        path.append(nextNode)
        currNode = nextNode
    path.append(start)
    cost += graph[start][path[len(path) - 2]]

    results = {}
    results["expanded"] = expanded
    results["path"] = path
    results["cost"] = cost
    return results

def RNN2Opt (graph, size, start):
    results = randomizedNN(graph, size, start)
    return twoOpt(graph, size, results)

def runRRNN (graph, size):
    cpu_start = time.process_time()
    real_start = time.time()
    bestResults = RNN2Opt (graph, size, 0)
    expanded = 0

    for i in range(1, size):
        temp = RNN2Opt (graph, size, i)
        expanded += temp["expanded"]
        if (temp["cost"] < bestResults["cost"]):
            bestResults = temp
    
    cpu_time = time.process_time() - cpu_start
    real_time = time.time() - real_start
    bestResults["cpu"] = cpu_time
    bestResults["real"] = real_time
    bestResults["expanded"] = expanded
    return bestResults

def runAllNN(graph_list, size):
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

    for g in graph_list:
        r = runNN(g, size)
        if (r["cost"] < min_cost):
            min_cost = r["cost"]
        if r["cpu"] < min_cpu:
            min_cpu = r["cpu"]
        if r["real"] < min_real:
            min_real = r["real"]
        if r["expanded"] < min_exp:
            min_exp = r["expanded"]
        if (r["cost"] > max_cost):
            max_cost = r["cost"]
        if r["cpu"] > max_cpu:
            max_cpu = r["cpu"]
        if r["real"] > max_real:
            max_real = r["real"]
        if r["expanded"] > max_exp:
            max_exp = r["expanded"]
        ave_cost += r["cost"]
        ave_cpu += r["cpu"]
        ave_real += r["real"]
        ave_exp += r["expanded"]

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
    return results

def runAll2Opt(graph_list, size):
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

    for g in graph_list:
        r = runTwoOpt(g, size)
        if (r["cost"] < min_cost):
            min_cost = r["cost"]
        if r["cpu"] < min_cpu:
            min_cpu = r["cpu"]
        if r["real"] < min_real:
            min_real = r["real"]
        if r["expanded"] < min_exp:
            min_exp = r["expanded"]
        if (r["cost"] > max_cost):
            max_cost = r["cost"]
        if r["cpu"] > max_cpu:
            max_cpu = r["cpu"]
        if r["real"] > max_real:
            max_real = r["real"]
        if r["expanded"] > max_exp:
            max_exp = r["expanded"]
        ave_cost += r["cost"]
        ave_cpu += r["cpu"]
        ave_real += r["real"]
        ave_exp += r["expanded"]

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
    return results

def runAllRRNN(graph_list, size):
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
    for g in graph_list:
        r = runRRNN(g, size)
        if (r["cost"] < min_cost):
            min_cost = r["cost"]
        if r["cpu"] < min_cpu:
            min_cpu = r["cpu"]
        if r["real"] < min_real:
            min_real = r["real"]
        if r["expanded"] < min_exp:
            min_exp = r["expanded"]
        if (r["cost"] > max_cost):
            max_cost = r["cost"]
        if r["cpu"] > max_cpu:
            max_cpu = r["cpu"]
        if r["real"] > max_real:
            max_real = r["real"]
        if r["expanded"] > max_exp:
            max_exp = r["expanded"]
        ave_cost += r["cost"]
        ave_cpu += r["cpu"]
        ave_real += r["real"]
        ave_exp += r["expanded"]

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
    return results
        
'''
my_csv = open("Part1_data.csv", 'a', newline='')
write = csv.writer(my_csv)
my_list = os.listdir('infiles')
nn_data = []
twoOpt_data = []
rnn_data = []
graph_list = []
for folder in my_list:
    file_list = os.listdir("infiles/" + folder)
    for file in file_list:
        graph_list.append(makeGraph("infiles/" + folder + "/"+ file))
    r1 = runAllNN(graph_list, len(graph_list[0]))
    r2 = runAll2Opt(graph_list, len(graph_list[0]))
    r3 = runAllRRNN(graph_list, len(graph_list[0]))
    nn_data.append(r1)
    twoOpt_data.append(r2)
    rnn_data.append(r3)
    graph_list.clear()
headers_NN = ['Nodes NN', 'Ave Cost NN', 'Min Cost NN', 'Max Cost NN','Ave CPU Time NN', 'Min CPU Time NN', 'Max CPU Time NN', 'Ave Real Time NN', 'Min Real Time NN', 'Max Real Time NN', 'Ave Expanded NN', 'Min Expanded NN', 'Max Expanded NN']
write.writerow(['Nearest Neighbor'])
write.writerow(headers_NN)
write.writerows(nn_data)
headers_twoOpt = ['Nodes 2-Opt', 'Ave Cost 2-Opt', 'Min Cost 2-Opt', 'Max Cost 2-Opt','Ave CPU Time 2-Opt', 'Min CPU Time 2-Opt', 'Max CPU Time 2-Opt', 'Ave Real Time 2-Opt', 'Min Real Time 2-Opt', 'Max Real Time 2-Opt', 'Ave Expanded 2-Opt', 'Min Expanded 2-Opt', 'Max Expanded 2-Opt']
write.writerow(['Two Opt Nearest Neighbor'])
write.writerow(headers_twoOpt)
write.writerows(twoOpt_data)
headers_RNN = ['Nodes RNN', 'Ave Cost RNN', 'Min Cost RNN', 'Max Cost RNN','Ave CPU Time RNN', 'Min CPU Time RNN', 'Max CPU Time RNN', 'Ave Real Time RNN', 'Min Real Time RNN', 'Max Real Time RNN', 'Ave Expanded RNN', 'Min Expanded RNN', 'Max Expanded RNN']
write.writerow(['Randomized RNN'])
write.writerow(headers_RNN)
write.writerows(rnn_data)
'''
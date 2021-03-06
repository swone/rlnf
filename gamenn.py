import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import start

class flow_graph():
    def __init__(self, graph):
        self.graph = graph
        for i in range(len(self.graph[0])):
            for j in range(len(self.graph[0])):
                if j == len(self.graph[0])-1:
                    self.graph[i][j] = 1
                else:
                    if self.graph[i][j] == 0:
                        self.graph[i][j] = -1
                    else:
                        self.graph[i][j] = 0
        self.net_flow = [0]*len(graph[0])
        self.source = 0
        self.sink = len(graph[0])-1
        self.current = source
        self.score = np.infty

    def getState(self):
        return self.graph

    def getMove(self, action):
        if self.current == self.sink:
            return self.score
        move = int(action*len(self.graph[0]))
        if self.graph[self.current][move] == 0:
            return -100
        else:
            self.score = min(self.score, self.graph[self.current][move])
            self.current = move
            return -1    

class agent():
    def __init__(self, lr, gamma, graph):
        self.gamma = gamma
        self.reward_graph = graph
        self.q = np.zeros([10, 10])
        self.current = 0
        self.sink = 9

    def update(self):
        end = False
        v = 0
        while not end:
            valid = False
            check = 0
            for j in range(10):
                if self.reward_graph[self.current][j] > -1:
                    check+=1
            if check == 0:
                return -1
            while not valid:
                i = np.random.randint(0, 10)
                if self.reward_graph[self.current][i] > -1:
                    valid = True
            maxq = 0
            v+=1
            for j in range(10):
                if self.reward_graph[i][j] > -1:
                    if self.q[i][j] > maxq:
                        maxq = self.q[i][j]
            self.q[self.current][i] = self.reward_graph[self.current][i]+self.gamma*maxq
            if i == self.sink:
                end = True
            else:
                self.current = i
            if v > 10:
                end = True

episodes = 500
ex_graph_1 = start.generate_random_graph(10, 0.4, 15)
print(start.edmonds_karp(ex_graph_1))
done = False
maxf = 0
while not done:
    graph = np.zeros([10, 10])
    for i in range(len(graph[0])):
        for j in range(len(graph[0])):
            if ex_graph_1[i][j] > 0:
                if j == 9:
                    graph[i][j] = 1
                else:
                    graph[i][j] = 0
            else:
                graph[i][j] = -1
    bond = agent(1, 0.9, graph)
    for i in range(episodes):
        bond.update()
    current = 0
    maxq = 0
    maxi = 0
    fin = False
    path = [0]
    count = 0
    for k in range(10):
        if bond.q[k][9] != 0:
            count+=1
    if count == 0:
        break
    while not fin:
        for i in range(10):
            if(bond.q[current][i] > maxq):
                maxq = bond.q[current][i]
                maxi = i
        path.append(maxi)
        if maxi == 9:
            fin = True
        else:
            current = maxi
    minf = 15
    for x in range(len(path)-1):
        minf = min(minf, ex_graph_1[path[x]][path[x+1]])
    maxf+=minf
    for x in range(len(path)-1):
        ex_graph_1[path[x]][path[x+1]] -= minf
        ex_graph_1[path[x+1]][path[x]] += minf
print(maxf)
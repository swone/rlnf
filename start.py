import numpy as np

def generate_random_graph(numV, p, max_flow):
    graph = np.zeros((numV, numV))
    for i in range(numV):
        for j in range(i+1, numV):
            random = np.random.rand()
            if random > p:
                graph[i][j] = np.random.randint(1, max_flow)
                graph[j][i] = graph[i][j]
    source = np.random.randint(0, numV)
    sink = np.random.randint(0, numV-1)
    return graph, source, sink

def depth_first_search(graph, source, sink, discovered, max_flow):
    discovered.append(source)
    if source == sink:
        return max_flow, discovered
    for i in range(len(graph[0])):
        if graph[source][i] > 0:
            if i not in discovered:
                found, discovered = depth_first_search(graph, i, sink, discovered, max_flow)
                if found != -1:
                    return min(found, graph[source][i]), discovered
    discovered.pop()
    return -1, discovered

def ford_fulkerson(graph, source, sink, max_flow):
    update = graph
    flow = 0
    maxed = False
    while not maxed:
        discovered = []
        add, path = depth_first_search(update, source, sink, discovered, max_flow)
        if add > 0:
            flow += add
            for i in range(len(path)-1):
                update[path[i]][path[i+1]] -= add
                update[path[i+1]][path[i]] += add
        else:
            maxed = True
    return flow       

ex_graph = [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
source = 1
sink = 3
max_flow = 10

print(ford_fulkerson(ex_graph, source, sink, 10))
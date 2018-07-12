import numpy as np

def generate_random_graph(numV, p, max_flow):
    graph = np.zeros((numV, numV))
    for i in range(numV):
        for j in range(numV):
            if i != j:
                random = np.random.rand()
                if random < p:
                    graph[i][j] = np.random.randint(1, max_flow)
    return graph

def depth_first_search(graph, source, sink, discovered):
    discovered.append(source)
    if source == sink:
        return np.infty, discovered
    for i in range(len(graph[0])):
        if graph[source][i] > 0:
            if i not in discovered:
                found, discovered = depth_first_search(graph, i, sink, discovered)
                if found != -1:
                    return min(found, graph[source][i]), discovered
    discovered.pop()
    return -1, discovered

def ford_fulkerson(graph):
    source = 0
    sink = len(graph[0])-1
    update = [element.copy() for element in graph]
    flow = 0
    maxed = False
    while not maxed:
        discovered = []
        add, path = depth_first_search(update, source, sink, discovered)
        if add > 0:
            flow += add
            for i in range(len(path)-1):
                update[path[i]][path[i+1]] -= add
                update[path[i+1]][path[i]] += add
        else:
            maxed = True
    return flow       

def edmonds_karp(graph):
    source = 0
    sink = len(graph[0])-1
    update = [element.copy() for element in graph]
    end = False
    flow = 0
    while not end:
        queue = [source]
        path = [None] * len(update[0])
        while len(queue) > 0:
            current = queue.pop(0)
            for i in range(0, len(update[0])):
                if ((path[i] == None) and (i != source) and (update[current][i] > 0)):
                    path[i] = current
                    queue.append(i)
        if path[sink] != None:
            df = np.infty
            t = sink
            while t != source:
                df = min(df, update[path[t]][t])
                t = path[t]
            flow += df
            t = sink
            while t != source:
                update[path[t]][t] -= df
                update[t][path[t]] += df
                t = path[t]
        else:
            end = True
    return flow

if __name__ == "__main__":
    ex_graph = [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
    source = 1
    sink = 3

    print(ford_fulkerson(ex_graph))
    print(ex_graph)
    print(edmonds_karp(ex_graph))
    print(ex_graph)
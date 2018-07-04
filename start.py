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
    current = source
    if source == sink:
        return max_flow, discovered
    for all i in range(current, numV):
        if graph[current][i] > 0:
            if graph[current][i] not in discovered:
                current = i
                found, discovered = depth_first_search(graph, current, sink, discovered, max_flow)
                return min(found, graph[source][current]), discovered
    return 0, discovered

def ford_fulkerson(graph, source, sink):
    discovered = []
    update = graph
    flow = 0
    add, path = depth_first_search(graph, source, sink, discovered, max_flow)
    if add > 0:
        flow += add
        for i in range(len(path)-1):
            update[path[i]][path[i+1]] -= add
            update[path[i]][path[i+1]] += add
    return flow       

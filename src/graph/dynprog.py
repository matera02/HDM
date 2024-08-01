from queue import PriorityQueue

import networkx as nx
import graph

"""
Creo una tabella offline di cost_to_goal(n) per ogni nodo n
Questa tabella contiene il costo totale per raggiungere un nodo goal
Posso utilizzare questo valore come valore dell'euristica su un nodo cioè heuristic(n)
E poi applico l'A Star

Per implementare la dpsearch uso come implementazione quella della LCFS con
multiple path pruning
"""
from queue import PriorityQueue

# Deve prendere in input il grafo al contrario
def dpSearch(graph, goal):
    print("DP SEARCH: ")
    frontier = PriorityQueue()
    visited = set()
    cost_to_goal = {}
    cost_to_goal[goal] = 0

    # Inizializzo a infinito il valore di cost_to_goal per ciascun nodo
    for node in graph.nodes:
        if node != goal:
            cost_to_goal[node] = float('inf')

    # Aggiungo il nodo di partenza alla coda di priorità
    frontier.put((0, (goal, [goal])))

    while not frontier.empty():
        priority, (current, path) = frontier.get()
        print(priority, path)
        print(current)

        if current not in visited:
            visited.add(current)

            for adj in graph.neighbors(current):
                peso = graph.get_edge_data(current, adj)['peso']
                new_cost = cost_to_goal[current] + peso
                if new_cost < cost_to_goal[adj]:
                    cost_to_goal[adj] = new_cost
                    frontier.put((new_cost, (adj, path + [adj])))

    return cost_to_goal

if __name__ == '__main__':
    G, node_labels, edge_labels = graph.generate_labeled_directed_graph_h()

    #dpsearch funziona con il grafo invertito
    GReverse = G.reverse()

    print(dpSearch(GReverse, 'G'))

    # ORA PER OTTENERE IL VALORE PER COST_TO_GOAL DOVE IL GOAL E' CIASCUN NODO BASTA CICLARE IL METODO SU
    # QUALSIASI NODO SETTANDOLO COME GOAL E IN QUESTA MANIERA OTTENERE IL DB PATTERN CHE MI SERVE

from queue import PriorityQueue
from pyswip import Prolog
from src.graph.provaProlog2 import GrafoStanze

def dpSearch(graph, goal):
    print("DP SEARCH: ")
    frontier = PriorityQueue()
    visited = set()
    cost_to_goal = {}

    for node in graph.nodes:
        cost_to_goal[node] = {goal: float('inf')}

    cost_to_goal[goal][goal] = 0
    frontier.put((0, (goal, [goal])))

    while not frontier.empty():
        priority, (current, path) = frontier.get()
        print(priority, path)
        print(current)

        if current not in visited:
            visited.add(current)

            for adj in graph.neighbors(current):
                peso = graph.get_edge_data(current, adj)['weight']
                new_cost = cost_to_goal[current][goal] + peso
                if new_cost < cost_to_goal[adj][goal]:
                    cost_to_goal[adj][goal] = new_cost
                    frontier.put((new_cost, (adj, path + [adj])))

    return cost_to_goal

def add_cost_to_goal_to_prolog(cost_to_goal, prolog_file):
    with open(prolog_file, 'a') as f:
        f.write("\n% Cost to goal values\n")
        for start, goals in cost_to_goal.items():
            for goal, cost in goals.items():
                f.write(f"cost_to_goal({start}, {goal}, {cost}).\n")

def get_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        current = path[i]
        next = path[i + 1]
        if graph.has_edge(current, next):
            peso = graph.get_edge_data(current, next)['weight']
            cost += peso
        else:
            return -1
    return cost

def get_heuristic(prolog, node, goal):
    query = list(prolog.query(f"cost_to_goal({node}, {goal}, Cost)"))
    if query:
        cost = query[0]['Cost']
        if cost == 'inf':
            return float('inf')
        try:
            return float(cost)
        except ValueError:
            print(f"Warning: Invalid heuristic value for node {node} to goal {goal}")
            return float('inf')
    return float('inf')  # Ritorna infinito se non trova un valore

def get_f(prolog, graph, path, goal):
    costo = get_cost(graph, path)
    if costo < 0:
        return None
    heuristic = get_heuristic(prolog, path[-1], goal)
    
    print("Euristica:", heuristic)
    print("Nodo: ", path[-1])
    
    if costo == float('inf') or heuristic == float('inf'):
        return float('inf')
    
    print("Costo: ", costo)
    print("F: ", costo + heuristic)
    
    return costo + heuristic




def AStarSearch(prolog, graph, start, goal):
    print('AStar Search: ')
    frontier = PriorityQueue()
    best_costs = {}  # Dizionario per tenere traccia del miglior costo per ogni nodo
    start_priority = get_f(prolog, graph, [start], goal)
    if start_priority == float('inf'):
        print(f"Il nodo di partenza {start} non è raggiungibile dal goal {goal}")
        return None
    frontier.put((start_priority, (start, [start])))
    best_costs[start] = start_priority

    while not frontier.empty():
        priority, (current, path) = frontier.get()
        current = path[-1]
        print(priority, path)
        print(current)

        if current == goal:
            return path

        if current in graph:
            for adj in graph.neighbors(current):
                new_path = path + [adj]
                new_priority = get_f(prolog, graph, new_path, goal)
                if new_priority is not None and new_priority != float('inf'):
                    if adj not in best_costs or new_priority < best_costs[adj]:
                        best_costs[adj] = new_priority
                        frontier.put((new_priority, (adj, new_path)))
        else:
            print(f"Il nodo {current} non è presente nel grafo.")

    print(f"Nessun percorso trovato da {start} a {goal}")
    return None


if __name__ == '__main__':
    filename = "src/graph/prova.pl"
    prolog = Prolog()
    prolog.consult(filename)
    G = GrafoStanze.build_graph_from_prolog(filename)

    # Cost_to_goal_dp

    ## Calcola cost_to_goal per ogni nodo come goal
    #for goal_node in G.nodes:
    #    GReverse = G.reverse()
    #    cost_to_goal = dpSearch(GReverse, goal_node)
    #    add_cost_to_goal_to_prolog(cost_to_goal, filename)

    # Esegui A* Search
    start = 101  # Nodo di partenza
    goal = 124   # Nodo di arrivo

    path = AStarSearch(prolog, G, start, goal)
    if path:
        print(f"Percorso trovato: ", path)
    else:
        print("Nessun percorso trovato.")
from queue import Queue, PriorityQueue
from src.graph.hospital import Hospital
from pyswip import Prolog
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import time

FILENAME = 'src/graph/data/hospital.pl'

class PathFinder:

    @staticmethod
    # dfs con cycle pruning e multiple path pruning
    def dfs(graph, start, goal):
        print('DFS: ')
        start_time = time.time()
        frontier = [(start, [start])]
        visited = set()  # Per il cycle pruning
        paths_explored = 0  # Numero di percorsi esplorati
        nodes_visited = 0  # Numero di nodi visitati

        while frontier:
            current, path = frontier.pop()
            nodes_visited += 1

            if current == goal:
                t = time.time() - start_time
                return path, paths_explored, nodes_visited, t

            if current not in visited:
                visited.add(current)

                for adj in graph.neighbors(current):
                    if adj not in path:  # Multiple path pruning
                        new_path = path + [adj]
                        paths_explored += 1
                        print((adj, new_path))
                        frontier.append((adj, new_path))

        return None
    
    @staticmethod
    # in questo caso non era necessario implementare
    # un meccanismo di pruning
    def bfs(graph, start, goal):
        print('BFS: ')
        start_time = time.time()
        frontier = Queue()
        visited = set()
        frontier.put((start, [start]))
        
        paths_explored = 0  # Numero di percorsi esplorati
        nodes_visited = 0  # Numero di nodi visitati
        
        while not frontier.empty():
            current, path = frontier.get()
            nodes_visited += 1
            
            if current == goal:
                t = time.time() - start_time
                return path, paths_explored, nodes_visited, t
            
            if current not in visited: #cp
                visited.add(current)
            
                for adj in graph.neighbors(current):
                    if adj not in path: #mpp
                        paths_explored += 1
                        print((adj, path + [adj]))
                        frontier.put((adj, path + [adj]))
        
        return None
    

    @staticmethod
    def IterativeDeepening(graph, start, goal, max_bound = None):
        print('Iterative Deepening: ')
        start_time = time.time()
    
        def DepthLimitedSearch(graph, node, goal, bound, path, paths_explored, nodes_visited, visited):
            nodes_visited += 1
            
            if node == goal:
                return path, paths_explored, nodes_visited
            
            if bound > 0:
                visited.add(node)  # Aggiungi il nodo corrente ai visitati
                
                for adj in graph.neighbors(node):
                    if adj not in path and adj not in visited:  # Multiple path pruning e cycle pruning
                        new_path = path + [adj]
                        paths_explored += 1
                        result, pe, nv = DepthLimitedSearch(graph, adj, goal, bound - 1, new_path, paths_explored, nodes_visited, visited)
                        if result is not None:
                            return result, pe, nv
    
                visited.remove(node)  # Rimuovi il nodo corrente dai visitati dopo l'esplorazione
                
            return None, paths_explored, nodes_visited
    
        paths_explored = 0
        nodes_visited = 0
        bound = 0
        
        if max_bound is None:
            max_bound = len(graph.nodes())  # Limite massimo della profondità basato sul numero di nodi del grafo
        
        while bound <= max_bound:
            visited = set()  # Inizializza l'insieme dei nodi visitati per ogni livello di profondità
            result, pe, nv = DepthLimitedSearch(graph, start, goal, bound, [start], paths_explored, nodes_visited, visited)
            paths_explored = pe
            nodes_visited = nv
            
            if result is not None:
                t = time.time() - start_time
                return result, paths_explored, nodes_visited, t
            
            bound += 1
    
        print(f"Nessun percorso trovato fino a una profondità di {max_bound}.")
        return None

    @staticmethod
    def lowestCostSearch(graph, start, goal):
        print('Lowest Cost Search: ')
        start_time = time.time()
        frontier = PriorityQueue()
        visited = set()
        best_costs = {}  # Aggiunta di una struttura per tenere traccia del miglior costo per ogni nodo
        paths_explored = 0
        nodes_visited = 0

        frontier.put((0, (start, [start])))
        best_costs[start] = 0  # Inizializza il miglior costo per il nodo di partenza

        while not frontier.empty():
            priority, (current, path) = frontier.get()
            nodes_visited += 1
            print(priority, path)

            if current == goal:
                t = time.time() - start_time
                return path, paths_explored, nodes_visited, t

            if current not in visited:
                visited.add(current)
                if current in graph:
                    for adj in graph.neighbors(current):
                        weight = graph.get_edge_data(current, adj)['weight']
                        new_cost = priority + weight
                        if adj not in best_costs or new_cost < best_costs[adj]:  # Multiple path pruning
                            best_costs[adj] = new_cost
                            frontier.put((new_cost, (adj, path + [adj])))
                            paths_explored += 1
                else:
                    print(f"Il nodo {current} non è presente nel grafo.")

        t = time.time() - start_time
        return None

    """
    Creo una tabella offline di cost_to_goal(n) per ogni nodo n
    Questa tabella contiene il costo totale per raggiungere un nodo goal
    Posso utilizzare questo valore come valore dell'euristica su un nodo cioè heuristic(n)
    E poi applico l'A Star

    Per implementare la dpsearch uso come implementazione quella della LCFS con
    multiple path pruning
    """
    @staticmethod
    def __dpSearch(graph, goal):
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
    
    @staticmethod
    def __add_cost_to_goal_to_prolog(cost_to_goal, prolog_file):
        with open(prolog_file, 'a') as f:
            f.write("\n% Cost to goal values\n")
            for start, goals in cost_to_goal.items():
                for goal, cost in goals.items():
                    f.write(f"cost_to_goal({start}, {goal}, {cost}).\n")

    # in questo modo quando chiamo il metodo posso calcolare le euristiche
    # necessarie per AStar, l'ideale è precalcolarlo e poi sfruttare
    # cost_to_goal per l'AStar
    @staticmethod
    def make_graph_heuristics(G=Hospital.get_hospital(), filename=FILENAME):
        for goal_node in G.nodes:
            GReverse = G.reverse()
            cost_to_goal = PathFinder.__dpSearch(GReverse, goal_node)
            PathFinder.__add_cost_to_goal_to_prolog(cost_to_goal, filename)

    # costo totale del percorso nel grafo
    @staticmethod
    def __get_cost(graph, path):
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

    @staticmethod
    # ottengio h(nodo)
    def __get_heuristic(prolog, node, goal):
        query = list(prolog.query(f"cost_to_goal({node}, {goal}, Cost)"))
        if query:
            cost = query[0]['Cost']
            if cost == 'inf':
                return float('inf')
            try:
                return float(cost)
            except ValueError:
                print(f"Attenzione: valore euristica non valido da start {node} a goal {goal}")
                return float('inf')
        return float('inf')  # Ritorna infinito se non trova un valore

    @staticmethod
    # f = cost + h per AStar
    def __get_f(prolog, graph, path, goal):
        costo = PathFinder.__get_cost(graph, path)
        if costo < 0:
            return None
        heuristic = PathFinder.__get_heuristic(prolog, path[-1], goal)    
        #print("Euristica:", heuristic)
        #print("Nodo: ", path[-1])
        if costo == float('inf') or heuristic == float('inf'):
            return float('inf')
        #print("Costo: ", costo)
        #print("F: ", costo + heuristic)
        return costo + heuristic
    
    @staticmethod
    def AStarSearch(graph, start, goal, filename=FILENAME):
        print('AStar Search: ')
        start_time = time.time()
        prolog = Prolog()
        prolog.consult(filename=filename)
        frontier = PriorityQueue()
        best_costs = {}
        start_priority = PathFinder.__get_f(prolog, graph, [start], goal)
        paths_explored = 0
        nodes_visited = 0

        if start_priority == float('inf'):
            print(f"Il nodo di partenza {start} non è raggiungibile dal goal {goal}")
            return None

        frontier.put((start_priority, (start, [start])))
        best_costs[start] = start_priority  # Inizializza il miglior costo per il nodo di partenza

        while not frontier.empty():
            priority, (current, path) = frontier.get()
            current = path[-1]
            nodes_visited += 1

            if current == goal:
                t = time.time() - start_time
                return path, paths_explored, nodes_visited, t

            if current in graph:
                for adj in graph.neighbors(current):
                    new_path = path + [adj]
                    new_priority = PathFinder.__get_f(prolog, graph, new_path, goal)
                    if new_priority is not None and new_priority != float('inf'):
                        if adj not in best_costs or new_priority < best_costs[adj]:  # Multiple path pruning
                            best_costs[adj] = new_priority
                            frontier.put((new_priority, (adj, new_path)))
                            paths_explored += 1
            else:
                print(f"Il nodo {current} non è presente nel grafo.")

        print(f"Nessun percorso trovato da {start} a {goal}")
        return None

    @staticmethod
    # aggiunto meccanismo di cycle pruning e multiple path pruning
    def DF_branch_and_bound(graph, start, goal, filename=FILENAME):
        print("DF Branch and Bound Search:")
        start_time = time.time()
        prolog = Prolog()
        prolog.consult(filename=filename)
        paths_explored = 0
        nodes_visited = 0

        def cbsearch(graph, path, goal, bound, frontier, visited, paths_explored, nodes_visited):
            current = path[-1]
            nodes_visited += 1
            f_value = PathFinder.__get_f(prolog, graph, path, goal)

            if f_value < bound:
                if current == goal:
                    return PathFinder.__get_cost(graph, path), paths_explored, nodes_visited
                else:
                    for adj in graph.neighbors(current):
                        if adj not in path:
                            new_path = path + [adj]
                            new_f = PathFinder.__get_f(prolog, graph, new_path, goal)
                            paths_explored += 1
                            if adj not in visited or new_f < visited[adj]:
                                visited[adj] = new_f
                                frontier.append((new_f, (adj, new_path)))
            return bound, paths_explored, nodes_visited

        frontier = []
        visited = defaultdict(lambda: sys.float_info.max)
        initial_cost = PathFinder.__get_f(prolog, graph, [start], goal)
        frontier.append((initial_cost, (start, [start])))
        visited[start] = initial_cost
        bound = sys.float_info.max

        best_path = None
        while frontier:
            print(frontier)
            cost, (current, path) = min(frontier, key=lambda x: x[0])
            frontier.remove((cost, (current, path)))

            if current == goal:
                if cost < bound:
                    bound = cost
                    best_path = path
            elif cost < bound:
                new_bound, pe, nv = cbsearch(graph, path, goal, bound, frontier, visited, paths_explored, nodes_visited)
                paths_explored = pe
                nodes_visited = nv
                if new_bound < bound:
                    bound = new_bound
        
        t = time.time() - start_time
        if best_path is None:
            return None
        return best_path, paths_explored, nodes_visited, t

if __name__ == '__main__':
    G = Hospital.get_hospital()

    path_bfs, paths_explored_bfs, nodes_visited_bfs, _ = PathFinder.bfs(G, 101, 320)
    path_dfs, paths_explored_dfs, nodes_visited_dfs, _ = PathFinder.dfs(G, 101, 320)
    path_id , paths_explored_id, nodes_visited_id, _ = PathFinder.IterativeDeepening(G, 101, 320)
    path_lcfs, paths_explored_lcfs, nodes_visited_lcfs, _ = PathFinder.lowestCostSearch(G, 101, 320)

    # QUESTO METODO È DA SISTEMARE PERCHÉ AGGIUNGE
    # INDIPENDENTEMENTE DAL FATTO SE SONO PRESENTI O MENO
    
    #PathFinder.make_graph_heuristics()

    path_astar, paths_explored_astar, nodes_visited_astar, _ = PathFinder.AStarSearch(G, 101, 320)
    path_dfbb, paths_explored_dfbb, nodes_visited_dfbb, _ = PathFinder.DF_branch_and_bound(G, 101, 320)

    print("Path trovato bfs: ", path_bfs, "\n")
    print("Path trovato dfs: ", path_dfs, "\n")
    print("Path trovato id: ", path_id, "\n")
    print("Path trovato lcfs: ", path_lcfs, "\n")
    print("Path trovato da AStar: ", path_astar, "\n")
    print("Path trovato da DFBB: ", path_dfbb, "\n")

    nodes_visited = {
        "BFS": nodes_visited_bfs,
        "DFS": nodes_visited_dfs,
        "ID": nodes_visited_id,
        "LCFS": nodes_visited_lcfs,
        "A*": nodes_visited_astar,
        "DFBB": nodes_visited_dfbb
    }

    print("Nodi visitati: \n", nodes_visited)

    paths_explored = {
        "BFS": paths_explored_bfs,
        "DFS": paths_explored_dfs,
        "ID": paths_explored_id,
        "LCFS": paths_explored_lcfs,
        "A*": paths_explored_astar,
        "DFBB": paths_explored_dfbb
    }

    print("Percorsi esplorati: \n", paths_explored)

    ## Creazione del grafico
    #fig, ax = plt.subplots()
    ## Aggiunta dei punti al grafico
    #for algorithm in nodes_visited:
    #    ax.scatter(nodes_visited[algorithm], paths_explored[algorithm], label=algorithm)
    ## Impostazione dei titoli degli assi e del grafico
    #ax.set_xlabel("Nodi Incontrati nel Percorso")
    #ax.set_ylabel("Percorsi Esplorati")
    #ax.set_title("Confronto Algoritmi di Ricerca")
    ## Aggiunta della legenda
    #ax.legend()
    ## Mostra il grafico
    #plt.show()
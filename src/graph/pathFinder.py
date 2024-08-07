from queue import Queue, PriorityQueue
from src.graph.hospital import Hospital
from pyswip import Prolog
from collections import defaultdict
import sys

FILENAME = 'src/graph/data/hospital.pl'

class PathFinder:

    @staticmethod
    # dfs con cycle pruning e multiple path pruning
    def dfs(graph, start, goal):
        print('DFS: ')
        frontier = [(start, [start])]
        visited = set()  # Per il cycle pruning

        while frontier:
            current, path = frontier.pop()

            if current == goal:
                return path

            if current not in visited:
                visited.add(current)

                for adj in graph.neighbors(current):
                    if adj not in path:  # Multiple path pruning
                        new_path = path + [adj]
                        print((adj, new_path))
                        frontier.append((adj, new_path))

        return None
    
    @staticmethod
    # in questo caso non era necessario implementare
    # un meccanismo di pruning
    def bfs(graph, start, goal):
        print('BFS: ')
        frontier = Queue()
        frontier.put((start, [start]))

        while not frontier.empty():
            current, path = frontier.get()
            if current == goal:
                return path
            for adj in graph.neighbors(current):
                print((adj, path + [adj]))
                frontier.put((adj, path + [adj]))
        return None
    

    @staticmethod
    def IterativeDeepening(graph, start, goal):
        print('Iterative Deepening: ')
        def DepthLimitedSearch(graph, node, goal, bound, path):
            if bound > 0:
                if node == goal:
                    return path
                for adj in graph.neighbors(node):
                    if adj not in path:
                        new_path = path + [adj]
                        print(new_path)
                        result = DepthLimitedSearch(graph, adj, goal, bound - 1, new_path)
                        if result is not None:
                            return result
            return None
        bound = 0
        while True:
            result = DepthLimitedSearch(graph, start, goal, bound, [start])
            if result is not None:
                return result
            bound += 1

    @staticmethod
    def lowestCostSearch(graph, start, goal):
        print('Lowest Cost Search: ')
        frontier = PriorityQueue()
        visited = set()
        frontier.put((0, (start, [start])))
        while not frontier.empty():
            priority, (current, path) = frontier.get()
            print(priority, path)
            print(current)
            if current == goal:
                return path
            if current not in visited:
                visited.add(current)
                if current in graph:
                    for adj in graph.neighbors(current):
                        try:
                            peso = graph.get_edge_data(current, adj)['weight']
                            frontier.put((priority + peso, (adj, path + [adj])))
                        except KeyError:
                            pass
                else:
                    print(f"Il nodo {current} non è presente nel grafo.")
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
    def AStarSearch(prolog, graph, start, goal):
        print('AStar Search: ')
        frontier = PriorityQueue()
        best_costs = {}  # Dizionario per tenere traccia del miglior costo per ogni nodo
        start_priority = PathFinder.__get_f(prolog, graph, [start], goal)
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
                    new_priority = PathFinder.__get_f(prolog, graph, new_path, goal)
                    if new_priority is not None and new_priority != float('inf'):
                        if adj not in best_costs or new_priority < best_costs[adj]:
                            best_costs[adj] = new_priority
                            frontier.put((new_priority, (adj, new_path)))
            else:
                print(f"Il nodo {current} non è presente nel grafo.")
        print(f"Nessun percorso trovato da {start} a {goal}")
        return None
    
    @staticmethod
    # aggiunto meccanismo di cycle pruning e multiple path pruning
    def DF_branch_and_bound(prolog, graph, start, goal):
        print("DF Branch and Bound Search:")

        def cbsearch(graph, path, goal, bound, frontier, visited):
            current = path[-1]
            f_value = PathFinder.__get_f(prolog, graph, path, goal)

            if f_value < bound:
                if current == goal:
                    return PathFinder.__get_cost(graph, path)
                else:
                    for adj in graph.neighbors(current):
                        if adj not in path:  # Cycle pruning
                            new_path = path + [adj]
                            new_f = PathFinder.__get_f(prolog, graph, new_path, goal)

                            # Multiple path pruning
                            if adj not in visited or new_f < visited[adj]:
                                visited[adj] = new_f
                                frontier.append((new_f, (adj, new_path)))

            return bound

        frontier = []
        visited = defaultdict(lambda: sys.float_info.max)

        initial_cost = PathFinder.__get_f(prolog, graph, [start], goal)
        frontier.append((initial_cost, (start, [start])))
        visited[start] = initial_cost

        bound = sys.float_info.max  # Usiamo il massimo valore float rappresentabile

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
                new_bound = cbsearch(graph, path, goal, bound, frontier, visited)
                if new_bound < bound:
                    bound = new_bound

        return best_path

if __name__ == '__main__':
    G = Hospital.get_hospital()

    path_bfs = PathFinder.bfs(G, 101, 320)
    path_dfs = PathFinder.dfs(G, 101, 320)
    path_id = PathFinder.IterativeDeepening(G, 101, 320)
    path_lcfs = PathFinder.lowestCostSearch(G, 101, 320)
    print("Path trovato bfs: ", path_bfs, "\n")
    print("Path trovato dfs: ", path_dfs, "\n")
    print("Path trovato id: ", path_id, "\n")
    print("Path trovato lcfs: ", path_lcfs, "\n")

    # QUESTO METODO È DA SISTEMARE PERCHÉ AGGIUNGE
    # INDIPENDENTEMENTE DAL FATTO SE SONO PRESENTI O MENO
    
    #PathFinder.make_graph_heuristics()

    prolog = Prolog()
    prolog.consult(filename=FILENAME)

    path_astar = PathFinder.AStarSearch(prolog, G, 101, 320)
    print("Path trovato da AStar: ", path_astar, "\n")

    path_dfbb = PathFinder.DF_branch_and_bound(prolog, G, 101, 320)
    print("Path trovato da DFBB", path_dfbb, "\n")
    

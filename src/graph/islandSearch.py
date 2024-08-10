# classe per ricerca basata su isole
from src.graph.hospital import Hospital
from collections import deque
from src.util.utility import Utility as util
from src.graph.pathFinder import PathFinder
import itertools
from pyswip import Prolog
import time

FILENAME = 'src/graph/data/hospital.pl'

class IDGS:
    # i nodi goal delle isole con cui abbiamo a che fare sono quelli che 
    # rappresentano le scale per salire di piano in piano
    def __init__(self, graph=Hospital.get_hospital(), islands_goals={1:124,2:202,3:320}):
        self.graph = graph
        # faccio il partizionamento in inizializzazione per risparmiare poi in esecuzione
        # cioè quando chiamo island_driven_graph_search
        self.islands = self.partition_graph(self.graph)
        self.islands_goals = islands_goals

    def island_driven_graph_search(self, start, goal, search_algorithm='bfs'):
        start_time = time.time()
        local_paths = []
        paths_explored = 0
        nodes_visited = 0
        current_start = start
        visited_islands = set()

        while current_start != goal:
            found_path = False
            for island in self.islands:
                island_id = id(island)  # Uso l'id dell'oggetto come identificatore unico
                if current_start in island and island_id not in visited_islands:
                    visited_islands.add(island_id)
                    current_goal = goal if goal in island else self.islands_goals[util.get_room_from_graph(self.graph, current_start).floor]
                    result = self.search(island, current_start, current_goal, search_algorithm)

                    if result is not None:
                        local_path, is_same_floor, end_node, local_paths_explored, local_nodes_visited = result
                        local_paths.append(local_path)
                        paths_explored += local_paths_explored
                        nodes_visited += local_nodes_visited

                        if is_same_floor and end_node == goal:
                            return self.__integrate_local_paths(local_paths), paths_explored, nodes_visited, time.time() - start_time
                        else:
                            # Aggiorno il nodo di partenza per la prossima iterazione
                            neighbors = list(self.graph.neighbors(end_node))
                            if neighbors:
                                current_start = neighbors[-1]  # le scale sono collegate all'entrata di un piano
                                found_path = True
                                break
                            else:
                                print(f"Il nodo {end_node} non ha adiacenti")
                                return None

            if not found_path:
                print(f"Nessun percorso trovato da {current_start} a {goal}")
                return None
        final_path = self.__integrate_local_paths(local_paths)
        return final_path, paths_explored, nodes_visited, time.time() - start_time
    
    # Criterio basato sul piano dei nodi, direttamente da oggetti Room
    def __floor_criterion(self, node, neighbor):
        node_floor = util.get_room_from_graph(self.graph, node).floor
        neighbor_floor = util.get_room_from_graph(self.graph, neighbor).floor
        return node_floor == neighbor_floor

    def partition_graph(self, graph):
        # Implementazione della funzione di partizionamento del grafo (bfs)
        def divide_into_islands(graph, criterion):
            islands = []
            visited_nodes = set()
            for node in graph.nodes:
                if node not in visited_nodes:
                    current_island = []
                    queue = deque([node])
                    while queue:
                        current_node = queue.popleft()
                        if current_node not in visited_nodes:
                            visited_nodes.add(current_node)
                            current_island.append(current_node)
                            for neighbor in graph.neighbors(current_node):
                                if criterion(current_node, neighbor) and neighbor not in visited_nodes:
                                    queue.append(neighbor)
                    islands.append(current_island)
            return islands
        
        # Conversione in grafi delle isole trovate
        def get_subgraphs_from_islands(graph, islands):
            subgraphs = []
            for island in islands:
                subgraph = graph.subgraph(island)
                subgraphs.append(subgraph)
            return subgraphs
        
        islands = divide_into_islands(self.graph, self.__floor_criterion)
        return get_subgraphs_from_islands(self.graph, islands) #restituisco i sottografi isole

    def search(self, island, start, goal, search_algorithm):
        # Implementazione della ricerca locale all'interno di un'isola
        
        # controllo se i nodi si trovano sullo stesso piano
        # se non lo sono aggiorno il nodo goal con il goal dell'isola 
        # e restituisco il percorso trovato con l'algoritmo selezionato
        is_same_floor = self.__floor_criterion(start, goal)
        if not is_same_floor:
            # in questo caso devo considerare tutto il grafo per il controllo
            goal = self.islands_goals[util.get_room_from_graph(self.graph, start).floor]

        # ulteriore controllo se la ricerca dovesse andare nel verso opposto
        if start not in island:
            print(f"Il nodo {start} non è presente nell'isola corrente.")
            return None

        # devo considerare l'isola ora
        print("goal=", goal)
        match search_algorithm:
            # il tempo di esecuzione viene gestito in isldgs
            case 'bfs':
                result = PathFinder.bfs(island, start, goal)
            case 'dfs':
                result = PathFinder.dfs(island, start, goal)
            # iterative deepening
            case 'id':
                result = PathFinder.IterativeDeepening(island, start, goal)
            case 'lcfs':
                result = PathFinder.lowestCostSearch(island, start, goal)
            case 'astar':
                result = PathFinder.AStarSearch(island, start, goal)
            case 'dfbb':
                result = PathFinder.DF_branch_and_bound(island, start, goal)
            case _:
                path = None
        if result is None:
            return None
        path, paths_explored, nodes_visited, _ = result
        return path, is_same_floor, goal, paths_explored, nodes_visited

    
    def __integrate_local_paths(self, local_paths):
        # Concateno i percorsi trovati in ordine
        return list(itertools.chain.from_iterable(local_paths))


if __name__ == '__main__':
    isl = IDGS()
    start = 101
    goal = 320
    path, _, _, _ = isl.island_driven_graph_search(start, goal)
    print("Path trovato: ", path)






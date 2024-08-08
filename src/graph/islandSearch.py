# classe per ricerca basata su isole
from src.graph.hospital import Hospital
from collections import deque
from src.util.utility import Utility as util
from src.graph.pathFinder import PathFinder
import itertools
from pyswip import Prolog

FILENAME = 'src/graph/data/hospital.pl'

class IDGS:
    # i nodi goal delle isole con cui abbiamo a che fare sono quelli che 
    # rappresentano le scale per salire di piano in piano
    def __init__(self, graph=Hospital.get_hospital(), search_algorithm = 'bfs', islands_goals={1:124,2:202,3:320}):
        self.graph = graph
        self.search_algorithm = search_algorithm
        self.islands_goals = islands_goals

    def island_driven_graph_search(self, start, goal):
        # Inizializzazione
        islands = self.partition_graph(self.graph)
        
        #i = 1
        #for isl in islands:
        #    print(f"{i}\t", isl)
        #    i += 1

        local_paths = []

        # Ricerca nelle isole
        for island in islands:
            print("start=", start)
            print("goal=",goal)
            local_path, is_same_floor, end_node = self.search(island, start, goal)
            local_paths.append(local_path)
            # se non sono sullo stesso piano vuol dire che il goal restituito è quello
            # del piano su cui ho fatto la ricerca, quindi devo aggiornare il nodo
            # di partenza goal è sempre lo stesso
            if not is_same_floor:
                # il vicino è uno come scelta progettuale
                try:
                    print("end_node=", end_node)
                    #print("Vicini: ", )
                    start = list(self.graph.neighbors(end_node))[-1] # le scale sono collegate all'entrata di un piano
                    print("start=",start)
                except StopIteration:
                    print("Il nodo goal non ha adiacenti")
            else:
                # se sono sullo stesso piano ha trovato il percorso
                break
        
        # Unione dei risultati
        final_path = self.__integrate_local_paths(local_paths)

        return final_path
    
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

    def search(self, island, start, goal):
        # Implementazione della ricerca locale all'interno di un'isola
        
        # controllo se i nodi si trovano sullo stesso piano
        # se non lo sono aggiorno il nodo goal con il goal dell'isola 
        # e restituisco il percorso trovato con l'algoritmo selezionato
        is_same_floor = self.__floor_criterion(start, goal)
        if not is_same_floor:
            # in questo caso devo considerare tutto il grafo per il controllo
            goal = self.islands_goals[util.get_room_from_graph(self.graph, start).floor]
        
        # devo considerare l'isola ora
        print("goal=", goal)
        match self.search_algorithm:
            case 'bfs':
                path = PathFinder.bfs(island, start, goal)
            case 'dfs':
                path = PathFinder.dfs(island, start, goal)
            # iterative deepening
            case 'id':
                path = PathFinder.IterativeDeepening(island, start, goal)
            case 'lcfs':
                path = PathFinder.lowestCostSearch(island, start, goal)
            case 'astar':
                path = PathFinder.AStarSearch(island, start, goal)
            case 'dfbb':
                path = PathFinder.DF_branch_and_bound(island, start, goal)
            case _:
                path = None
        return path, is_same_floor, goal

    
    def __integrate_local_paths(self, local_paths):
        # Concateno i percorsi trovati in ordine
        return list(itertools.chain.from_iterable(local_paths))


if __name__ == '__main__':
    isl = IDGS(search_algorithm='dfbb')
    start = 101
    goal = 320
    path = isl.island_driven_graph_search(start, goal)
    print("Path trovato: ", path)






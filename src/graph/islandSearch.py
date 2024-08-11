from src.graph.hospital import Hospital
from collections import deque
from src.util.utility import Utility as util
from src.graph.pathFinder import PathFinder
import itertools
import time
import networkx as nx

class IDGS:
    def __init__(self, graph=Hospital.get_hospital(), stairs_nodes={1:{'up': 124, 'down': None}, 2:{'up': 202, 'down': 201}, 3:{'up': None, 'down': 301}}):
        self.graph = graph
        self.islands = self.partition_graph(self.graph)
        self.stairs_nodes = stairs_nodes

    def island_driven_graph_search(self, start, goal, search_algorithm='bfs', filename_heuristic=None):
        start_time = time.time()
        local_paths = []
        paths_explored = 0
        nodes_visited = 0
        current_floor = util.get_room_from_graph(self.graph, start).floor
        goal_floor = util.get_room_from_graph(self.graph, goal).floor

        while current_floor != goal_floor:
            island = self.islands[current_floor - 1]  # Assumendo che gli indici delle isole inizino da 0
            if current_floor < goal_floor:
                local_goal = self.stairs_nodes[current_floor]['up']
                direction = 'up'
            else:
                local_goal = self.stairs_nodes[current_floor]['down']
                direction = 'down'

            result = self.search(island, start, local_goal, search_algorithm, filename_heuristic=filename_heuristic)
            if result is None:
                return None
            
            local_path, _, _, local_paths_explored, local_nodes_visited = result
            local_paths.append(local_path)
            paths_explored += local_paths_explored
            nodes_visited += local_nodes_visited

            next_start = self.get_next_floor_start(local_goal, direction)
            if next_start is None:
                print(f"Impossibile trovare un percorso verso il piano successivo da {local_goal}")
                return None
        
            start = next_start
            current_floor += 1 if direction == 'up' else -1
        
        # Ricerca finale sul piano del goal
        result = self.search(self.islands[goal_floor - 1], start, goal, search_algorithm, filename_heuristic=filename_heuristic)
        if result is None:
            return None
        
        final_local_path, _, _, local_paths_explored, local_nodes_visited = result
        local_paths.append(final_local_path)
        paths_explored += local_paths_explored
        nodes_visited += local_nodes_visited

        final_path = self.__integrate_local_paths(local_paths)
        return final_path, paths_explored, nodes_visited, time.time() - start_time
    

    def get_next_floor_start(self, current_node, direction):
        current_floor = util.get_room_from_graph(self.graph, current_node).floor
        target_floor = current_floor + 1 if direction == 'up' else current_floor - 1

        for neighbor in self.graph.neighbors(current_node):
            neighbor_floor = util.get_room_from_graph(self.graph, neighbor).floor
            if neighbor_floor == target_floor:
                return neighbor

        print(f"Nessun nodo trovato al piano {target_floor} partendo da {current_node}")
        return None
    
    def __floor_criterion(self, node, neighbor):
        node_floor = util.get_room_from_graph(self.graph, node).floor
        neighbor_floor = util.get_room_from_graph(self.graph, neighbor).floor
        return node_floor == neighbor_floor

    def partition_graph(self, graph):
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
        
        def get_subgraphs_from_islands(graph, islands):
            subgraphs = []
            for island in islands:
                subgraph = graph.subgraph(island)
                subgraphs.append(subgraph)
            return subgraphs
        
        islands = divide_into_islands(self.graph, self.__floor_criterion)
        return get_subgraphs_from_islands(self.graph, islands)
    
    # ho aggiunto filename_heuristic nel caso in cui mi servisse l'euristica con il 
    # grafo fortemente connesso
    def search(self, island, start, goal, search_algorithm, filename_heuristic):
        if start not in island:
            print(f"Il nodo {start} non è presente nell'isola corrente.")
            return None

        match search_algorithm:
            case 'bfs':
                result = PathFinder.bfs(island, start, goal)
            case 'dfs':
                result = PathFinder.dfs(island, start, goal)
            case 'id':
                result = PathFinder.IterativeDeepening(island, start, goal)
            case 'lcfs':
                result = PathFinder.lowestCostSearch(island, start, goal)
            case 'astar':
                result = PathFinder.AStarSearch(island, start, goal) if filename_heuristic is None else PathFinder.AStarSearch(island, start, goal, filename=filename_heuristic)
            case 'dfbb':
                result = PathFinder.DF_branch_and_bound(island, start, goal) if filename_heuristic is None else PathFinder.DF_branch_and_bound(island, start, goal, filename=filename_heuristic)
            case _:
                return None

        if result is None:
            return None
        path, paths_explored, nodes_visited, _ = result
        return path, True, goal, paths_explored, nodes_visited

    def __integrate_local_paths(self, local_paths):
        return list(itertools.chain.from_iterable(local_paths))

if __name__ == '__main__':
    G = Hospital.get_hospital()
    G_reverse = G.reverse()
    G_combined = nx.compose(G, G_reverse)
    isl = IDGS(G_combined)
    start = 320
    goal = 101
    result = isl.island_driven_graph_search(start, goal)
    if result:
        path, paths_explored, nodes_visited, execution_time = result
        print("Path trovato:", path)
        print("Numero di percorsi esplorati:", paths_explored)
        print("Numero di nodi visitati:", nodes_visited)
        print("Tempo di esecuzione:", execution_time, "secondi")
    else:
        print("Nessun percorso trovato")
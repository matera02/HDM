from src.graph.hospital import Hospital
from src.graph.pathFinder import PathFinder
import networkx as nx
from functools import partial
from src.graph.islandSearch import IDGS

FILENAME = 'src/graph/data/hospital_strongly_connected.pl'
OUTPUT_FILE = 'src/graph/data/unfound.txt'

# tento la mia ricerca in una direzione se non trovo un risultato in quella direzione
# cerco nella direzione inversa
def get_evaluation_params(graph, algorithm):
    paths_explored = []
    nodes_visited = []
    times = []
    paths_found = 0
    unfound = []
    for node_start in graph.nodes():
        for node_goal in graph.nodes():
            if node_start != node_goal:
                result = algorithm(graph, node_start, node_goal)
                if result is not None:
                    _, local_paths_explored, local_nodes_visited, local_time = result
                    paths_explored.append(local_paths_explored)
                    nodes_visited.append(local_nodes_visited)
                    times.append(local_time)
                    paths_found += 1
                else:
                    unfound.append((node_start, node_goal))
                    print(f"Nessun percorso trovato da {node_start} a {node_goal} con {algorithm.__name__}")
    
    return paths_found, paths_explored, nodes_visited, times, unfound

def get_evaluation_params_idgs(idgs:IDGS, algorithm='bfs', filename_heuristic=None):
    paths_explored = []
    nodes_visited = []
    times = []
    paths_found = 0
    unfound = []
    for node_start in idgs.graph.nodes():
        for node_goal in idgs.graph.nodes():
            if node_start != node_goal:
                result = idgs.island_driven_graph_search(node_start, node_goal, search_algorithm=algorithm, filename_heuristic=filename_heuristic)
                if result is not None:
                    _, local_paths_explored, local_nodes_visited, local_time = result
                    paths_explored.append(local_paths_explored)
                    nodes_visited.append(local_nodes_visited)
                    times.append(local_time)
                    paths_found += 1
                else:
                    unfound.append((node_start, node_goal))
                    print(f"Nessun percorso trovato da {node_start} a {node_goal} con {algorithm}")
    
    return paths_found, paths_explored, nodes_visited, times, unfound

if __name__ == '__main__':
    G = Hospital.get_hospital()
    G_reverse = G.reverse()
    G_combined = nx.compose(G, G_reverse)

    #n_paths_bfs, paths_explored_bfs, nodes_visited_bfs, times_bfs, unfound = get_evaluation_params(G_combined, PathFinder.bfs)
    
    # Ottengo l'euristica che mi serve nella ricerca
    #PathFinder.make_graph_heuristics(G_combined, filename=FILENAME)

    #aStarSearch = partial(PathFinder.AStarSearch, filename=FILENAME)
    #n_paths_astar, paths_explored_astar, nodes_visited_astar, times_astar, unfound = get_evaluation_params(G_combined, aStarSearch)
    
    #dfbbSearch = partial(PathFinder.DF_branch_and_bound, filename=FILENAME)
    #n_paths_dfbb, paths_explored_dfbb, nodes_visited_dfbb, times_dfbb, unfound = get_evaluation_params(G_combined, dfbbSearch)
    
    idgs = IDGS(G_combined)
    #n_paths_isl_bfs, paths_explored_isl_bfs, nodes_visited_isl_bfs, times_isl_bfs, unfound = get_evaluation_params_idgs(idgs)
    #n_paths_isl_dfs, paths_explored_isl_dfs, nodes_visited_isl_dfs, times_isl_dfs, unfound = get_evaluation_params_idgs(idgs, algorithm='dfs')
    #n_paths_isl_lcfs, paths_explored_isl_lcfs, nodes_visited_isl_lcfs, times_isl_lcfs, unfound = get_evaluation_params_idgs(idgs, algorithm='lcfs')
    #n_paths_isl_astar, paths_explored_isl_astar, nodes_visited_isl_astar, times_isl_astar, unfound = get_evaluation_params_idgs(idgs, algorithm='astar', filename_heuristic=FILENAME)
    n_paths_isl_dfbb, paths_explored_isl_dfbb, nodes_visited_isl_dfbb, times_isl_dfbb, unfound = get_evaluation_params_idgs(idgs, algorithm='dfbb', filename_heuristic=FILENAME)
    
    paths_required = 0
    for start in G.nodes():
        for goal in G.nodes():
            if start != goal:
                paths_required += 1

    # Scrivo i percorsi non trovati nel file di output
    if unfound != []:
        with open(OUTPUT_FILE, 'w') as f:
            f.write("Percorsi non trovati:\n")
            for start, goal in unfound:
                f.write(f"Start={start}, Goal={goal}\n")
                print(f"Start={start}, Goal={goal}")

    print(f"Percorsi non trovati salvati nel file {OUTPUT_FILE}")

    print("Path richiesti: ", paths_required)
    #print("Path trovati da bfs: ", n_paths_bfs, "\n")
    #print("Path trovati da AStar: ", n_paths_astar, "\n")
    #print("Path trovati da DFBB: ", n_paths_dfbb)
    #print("Path trovati da ISL-BFS: ", n_paths_isl_bfs)
    #print("Path trovati da ISL-DFS: ", n_paths_isl_dfs)
    #print("Path trovati da ISL-LCFS: ", n_paths_isl_lcfs)
    #print("Path trovati da ISL-A*: ", n_paths_isl_astar)
    print("Path trovati da ISL-DFBB: ", n_paths_isl_dfbb)


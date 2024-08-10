from src.graph.hospital import Hospital
from src.graph.pathFinder import PathFinder
import networkx as nx
from functools import partial

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

if __name__ == '__main__':
    G = Hospital.get_hospital()
    G_reverse = G.reverse()
    G_combined = nx.compose(G, G_reverse)

    #n_paths_bfs, paths_explored_bfs, nodes_visited_bfs, times_bfs, unfound = get_evaluation_params(G_combined, PathFinder.bfs)
    
    # Ottengo l'euristica che mi serve nella ricerca
    #PathFinder.make_graph_heuristics(G_combined, filename=FILENAME)

    #aStarSearch = partial(PathFinder.AStarSearch, filename=FILENAME)
    #n_paths_astar, paths_explored_astar, nodes_visited_astar, times_astar, unfound = get_evaluation_params(G_combined, aStarSearch)
    
    dfbbSearch = partial(PathFinder.DF_branch_and_bound, filename=FILENAME)
    n_paths_dfbb, paths_explored_dfbb, nodes_visited_dfbb, times_dfbb, unfound = get_evaluation_params(G_combined, dfbbSearch)
    
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
    print("Path trovati da DFBB: ", n_paths_dfbb)
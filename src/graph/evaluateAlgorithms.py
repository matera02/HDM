from src.graph.hospital import Hospital
from src.graph.pathFinder import PathFinder
from src.graph.islandSearch import IDGS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Funzione per salvare i dati in un unico file
def save_all_data_to_pickle(data_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Tutti i dati sono stati salvati in {filename}")

# Funzione per caricare i dati da un unico file
def load_all_data_from_pickle(filename):
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"Tutti i dati sono stati caricati da {filename}")
    return data_dict



def get_evaluation_params(graph, algorithm):
    paths_explored = []
    nodes_visited = []
    times = []
    paths_found = 0
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
                    print(f"Nessun percorso trovato da {node_start} a {node_goal} con {algorithm.__name__}")
    
    return paths_found, paths_explored, nodes_visited, times

def get_evaluation_params_idgs(idgs:IDGS, algorithm='bfs'):
    paths_explored = []
    nodes_visited = []
    times = []
    paths_found = 0
    for node_start in idgs.graph.nodes():
        for node_goal in idgs.graph.nodes():
            if node_start != node_goal:
                result = idgs.island_driven_graph_search(node_start, node_goal, search_algorithm=algorithm)
                if result is not None:
                    _, local_paths_explored, local_nodes_visited, local_time = result
                    paths_explored.append(local_paths_explored)
                    nodes_visited.append(local_nodes_visited)
                    times.append(local_time)
                    paths_found += 1
                else:
                    print(f"Nessun percorso trovato da {node_start} a {node_goal} con {algorithm}")
    
    return paths_found, paths_explored, nodes_visited, times


#aggrego i dati in base al numero di nodi esplorati 
# e calcolato la media dei percorsi esplorati all'interno di ciascun intervallo
def plot_aggregated_data(nodes_visited, paths_explored, savefig='src/graph/data/agg_search_alg.png'):
    
    def calculate_aggregated_data(nodes_visited, paths_explored, bin_size=5):
        aggregated_nodes = {}
        aggregated_paths = {}

        for algorithm in nodes_visited:
            nodes = np.array(nodes_visited[algorithm])
            paths = np.array(paths_explored[algorithm])

            max_nodes = nodes.max()
            bins = range(0, max_nodes + bin_size, bin_size)

            digitized = np.digitize(nodes, bins)
            aggregated_nodes[algorithm] = [nodes[digitized == i].mean() for i in range(1, len(bins))]
            aggregated_paths[algorithm] = [paths[digitized == i].mean() for i in range(1, len(bins))]

        return aggregated_nodes, aggregated_paths
    
    ## Calcolo aggregato dei dati
    agg_nodes_visited, agg_paths_explored = calculate_aggregated_data(nodes_visited, paths_explored)
    # Creazione del grafico a linee aggregato
    fig, ax = plt.subplots()
    for algorithm in agg_nodes_visited:
        ax.plot(agg_nodes_visited[algorithm], agg_paths_explored[algorithm], label=algorithm)
    # Impostazione dei titoli degli assi e del grafico
    ax.set_xlabel("Nodi Incontrati nel Percorso (aggregati)")
    ax.set_ylabel("Percorsi Esplorati (media per intervallo)")
    ax.set_title("Confronto Algoritmi di Ricerca (aggregato)")
    # Aggiunta della legenda
    ax.legend()

    plt.savefig(savefig)

    # Mostra il grafico
    plt.grid(True)
    plt.show()
    

    


# DA SPOSTARE IN UTIL PLOT_CUMULATIVE_EXECUTION_TIMES
# E GET_EXECUTION_TIME_STATS, DATO CHE SONO STATI GIÀ UTILIZZATI PER CSP

def plot_cumulative_execution_times(times_dict, savefig='src/graph/data/cumulative_execution_times.png'):
    # Creo il grafico
    plt.figure(figsize=(10, 6))

    # Itero sui dati, calcolo la somma cumulativa e plotto ciascuna lista di tempi di esecuzione
    for label, times in times_dict.items():
        cumulative_times = np.cumsum(times)
        esempi = list(range(1, len(cumulative_times) + 1))
        plt.plot(esempi, cumulative_times, marker='o', label=label)

    # Aggiungo le etichette e la leggenda
    plt.title('Somma Cumulativa dei Tempi di Esecuzione per Numero di Esempio')
    plt.xlabel('Numero di Esempio')
    plt.ylabel('Somma Cumulativa del Tempo di Esecuzione (s)')
    plt.legend()

    # Salvo il plot come immagine
    plt.savefig(savefig)

    # Mostro il grafico
    plt.show()

def get_execution_time_stats(times_dict, savefig='src/graph/data/time_stats.png'):
    # Creo una lista per memorizzare i dati delle statistiche
    data = []
    
    # Itero sui dati per calcolare le statistiche
    for label, times in times_dict.items():
        mean_time = round(np.mean(times), 3)
        min_time = round(np.min(times), 3)
        max_time = round(np.max(times), 3)
        data.append([label, mean_time, min_time, max_time])
    
    # Creo un DataFrame con i dati
    df = pd.DataFrame(data, columns=['Algoritmo', 'Tempo Medio (s)', 'Tempo Minimo (s)', 'Tempo Massimo (s)'])

    fig, ax = plt.subplots(figsize=(10, 4))  # Imposto la dimensione della figura
    ax.axis('tight')
    ax.axis('off')
    
    # Creo la tabella
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Formatto la tabella
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.title('Statistiche dei Tempi di Esecuzione')

    # Salvo la tabella
    plt.savefig(savefig)

    plt.show()

# controllo che ciascun metodo chiamato abbia prodotto lo stesso numero di soluzioni
def check_solutions_found(*n_paths_list):
    # Controlla se tutti i valori in n_paths_list sono uguali
    if all(n_paths == n_paths_list[0] for n_paths in n_paths_list):
        print("Tutti gli algoritmi hanno trovato lo stesso numero di soluzioni.")
    else:
        print("Attenzione: Gli algoritmi non hanno trovato lo stesso numero di soluzioni!")
        for i, n_paths in enumerate(n_paths_list):
            print(f"Algoritmo {i + 1}: {n_paths} percorsi trovati")

    


if __name__ == '__main__':
    #G = Hospital.get_hospital()
    #
    ## Eseguire la valutazione per ogni algoritmo
    #n_paths_bfs, paths_explored_bfs, nodes_visited_bfs, times_bfs = get_evaluation_params(G, PathFinder.bfs)
    #n_paths_dfs, paths_explored_dfs, nodes_visited_dfs, times_dfs = get_evaluation_params(G, PathFinder.dfs)
    #n_paths_id, paths_explored_id, nodes_visited_id, times_id = get_evaluation_params(G, PathFinder.IterativeDeepening)
    #n_paths_lcfs, paths_explored_lcfs, nodes_visited_lcfs, times_lcfs = get_evaluation_params(G, PathFinder.lowestCostSearch)
    #n_paths_astar, paths_explored_astar, nodes_visited_astar, times_astar = get_evaluation_params(G, PathFinder.AStarSearch)
    #n_paths_dfbb, paths_explored_dfbb, nodes_visited_dfbb, times_dfbb = get_evaluation_params(G, PathFinder.DF_branch_and_bound)
#
    ## Organizzare i risultati per ogni algoritmo
    #nodes_visited = {
    #    "BFS": nodes_visited_bfs,
    #    "DFS": nodes_visited_dfs,
    #    "ID": nodes_visited_id,
    #    "LCFS": nodes_visited_lcfs,
    #    "A*": nodes_visited_astar,
    #    "DFBB": nodes_visited_dfbb
    #}
#
    #paths_explored = {
    #    "BFS": paths_explored_bfs,
    #    "DFS": paths_explored_dfs,
    #    "ID": paths_explored_id,
    #    "LCFS": paths_explored_lcfs,
    #    "A*": paths_explored_astar,
    #    "DFBB": paths_explored_dfbb
    #}
    #
    #plot_aggregated_data(nodes_visited, paths_explored)
    #
    #times_dict = {
    #    'BFS':times_bfs,
    #    'DFS':times_dfs,
    #    'ID':times_id,
    #    'LCFS':times_lcfs,
    #    'A*':times_astar,
    #    'DFBB':times_dfbb
    #}
#
    #plot_cumulative_execution_times(times_dict)
    #get_execution_time_stats(times_dict)
#
    #idgs = IDGS(G)
    #n_paths_isl_bfs, paths_explored_isl_bfs, nodes_visited_isl_bfs, times_isl_bfs = get_evaluation_params_idgs(idgs)
    #n_paths_isl_dfs, paths_explored_isl_dfs, nodes_visited_isl_dfs, times_isl_dfs = get_evaluation_params_idgs(idgs, algorithm='dfs')
    #n_paths_isl_id, paths_explored_isl_id, nodes_visited_isl_id, times_isl_id = get_evaluation_params_idgs(idgs, algorithm='id')
    #n_paths_isl_lcfs, paths_explored_isl_lcfs, nodes_visited_isl_lcfs, times_isl_lcfs = get_evaluation_params_idgs(idgs, algorithm='lcfs')
    #n_paths_isl_astar, paths_explored_isl_astar, nodes_visited_isl_astar, times_isl_astar = get_evaluation_params_idgs(idgs, algorithm='astar')
    #n_paths_isl_dfbb, paths_explored_isl_dfbb, nodes_visited_isl_dfbb, times_isl_dfbb = get_evaluation_params_idgs(idgs, algorithm='dfbb')
#
    #nodes_visited_isl = {
    #    'ISL-BFS':nodes_visited_isl_bfs,
    #    'ISL-DFS':nodes_visited_isl_dfs,
    #    'ISL-ID':nodes_visited_isl_id,
    #    'ISL-LCFS':nodes_visited_isl_lcfs,
    #    'ISL-A*':nodes_visited_isl_astar,
    #    'ISL-DFBB':nodes_visited_isl_dfbb
    #}
#
    #paths_explored_isl = {
    #    'ISL-BFS': paths_explored_isl_bfs,
    #    'ISL-DFS': paths_explored_isl_dfs,
    #    'ISL-ID': paths_explored_isl_id,
    #    'ISL-LCFS':paths_explored_isl_lcfs,
    #    'ISL-A*':paths_explored_isl_astar,
    #    'ISL-DFBB':paths_explored_isl_dfbb
    #}
#
    #plot_aggregated_data(nodes_visited_isl, paths_explored_isl, savefig='src/graph/data/agg_search_alg_isl.png')
#
    #times_dict_isl = {
    #    'ISL-BFS':times_isl_bfs,
    #    'ISL-DFS':times_isl_dfs,
    #    'ISL-ID':times_isl_id,
    #    'ISL-LCFS':times_isl_lcfs, 
    #    'ISL-A*':times_isl_astar ,
    #    'ISL-DFBB':times_isl_dfbb 
    #}
#
    #plot_cumulative_execution_times(times_dict_isl, savefig='src/graph/data/cumulative_execution_times_isl.png')
    #get_execution_time_stats(times_dict_isl, savefig='src/graph/data/time_stats_isl.png')
#
#
    #paths_required = 0
    #for start in G.nodes():
    #    for goal in G.nodes():
    #        paths_required += 1
    #print("Path richiesti: ", paths_required)
#
    #check_solutions_found(
    #    n_paths_bfs,
    #    n_paths_dfs, 
    #    n_paths_id, 
    #    n_paths_lcfs, 
    #    n_paths_astar, 
    #    n_paths_dfbb,
    #    n_paths_isl_bfs, 
    #    n_paths_isl_dfs, 
    #    n_paths_isl_id,
    #    n_paths_isl_lcfs,
    #    n_paths_isl_astar,
    #    n_paths_isl_dfbb,
    #)
#
    #all_data = {
    #    "nodes_visited":nodes_visited,
    #    "paths_explored":paths_explored,
    #    "times_dict":times_dict,
    #    "nodes_visited_isl":nodes_visited_isl,
    #    "paths_explored_isl":paths_explored_isl,
    #    "times_dict_isl":times_dict_isl
    #}
#
    ## Utilizzo
    #save_all_data_to_pickle(all_data, 'src/graph/data/all_search_data.pkl')

    # Carico i dati
    loaded_data = load_all_data_from_pickle('src/graph/data/all_search_data.pkl')

    loaded_nodes_visited = loaded_data['nodes_visited']
    loaded_paths_explored = loaded_data['paths_explored']
    loaded_times_dict = loaded_data['times_dict']

    loaded_nodes_visited_isl = loaded_data['nodes_visited_isl']
    loaded_paths_explored_isl = loaded_data['paths_explored_isl']
    loaded_times_dict_isl = loaded_data['times_dict_isl']

    # Li sfrutto per le statistiche
    plot_aggregated_data(loaded_nodes_visited, loaded_paths_explored)
    #plot_cumulative_execution_times(loaded_times_dict)
    #get_execution_time_stats(loaded_times_dict)

    plot_aggregated_data(loaded_nodes_visited_isl, loaded_paths_explored_isl, savefig='src/graph/data/agg_search_alg_isl.png')
    




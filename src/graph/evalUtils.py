import matplotlib.pyplot as plt
import numpy as np
from src.graph.islandSearch import IDGS
from src.util.utility import Utility as util
import networkx as nx

class EvaluationUtils:

    @staticmethod
    def plot_aggregated_data(nodes_visited, paths_explored, savefig):
        # Aggrego i dati in bin (intervalli) di dimensione specificata (default 5)
        # Calcolo la media dei nodi visitati e dei percorsi esplorati per ogni bin
        # Restituisco i dati aggregati per nodi e percorsi
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

    # controllo che ciascun metodo chiamato abbia prodotto lo stesso numero di soluzioni
    @staticmethod
    def check_solutions_found(*n_paths_list):
        # Controlla se tutti i valori in n_paths_list sono uguali
        if all(n_paths == n_paths_list[0] for n_paths in n_paths_list):
            print("Tutti gli algoritmi hanno trovato lo stesso numero di soluzioni: ", n_paths_list[0])
        else:
            print("Attenzione: Gli algoritmi non hanno trovato lo stesso numero di soluzioni!")
            for i, n_paths in enumerate(n_paths_list):
                print(f"Algoritmo {i + 1}: {n_paths} percorsi trovati")

    @staticmethod
    # questo è valido per gli algoritmi in cui la ricerca non è mediante isole
    def get_evaluation_params(graph, algorithm):
        print(f'{algorithm.__name__}')
        paths_explored = []
        nodes_visited = []
        times = []
        paths_found = 0
        unfound = []
        iter = 0
        for node_start in graph.nodes():
            for node_goal in graph.nodes():
                print(f'Esempio: {iter} - Algoritmo: {algorithm.__name__} - Start={node_start} - Goal={node_goal}')
                if node_start != node_goal and nx.has_path(G=graph, source=node_start, target=node_goal):
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
                    iter+=1

        return paths_found, paths_explored, nodes_visited, times, unfound
    
    @staticmethod
    # per ricerca mediante isole
    def get_evaluation_params_idgs(idgs:IDGS, algorithm='bfs', filename_heuristic=None):
        print(algorithm)
        paths_explored = []
        nodes_visited = []
        times = []
        paths_found = 0
        unfound = []
        iter = 0
        for node_start in idgs.graph.nodes():
            for node_goal in idgs.graph.nodes():
                if node_start != node_goal and nx.has_path(G=idgs.graph, source=node_start, target=node_goal):
                    print(f'Esempio: {iter} - Algoritmo: {algorithm} - Start={node_start} - Goal={node_goal}')
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
                    iter+=1

        return paths_found, paths_explored, nodes_visited, times, unfound
    
    @staticmethod
    # metodo scritto per debug, con bigraph si trovano tutti i percorsi
    def write_unfound_paths(unfound, alg_name, output_file):
        if unfound != []:
            with open(output_file, 'w') as f:
                f.write(f"Percorsi non trovati - {alg_name}:\n")
                for start, goal in unfound:
                    f.write(f"Start={start}, Goal={goal}\n")
                    print(f"Start={start}, Goal={goal}")
        print(f"Percorsi non trovati salvati nel file {output_file}")

    @staticmethod
    def save_evaluation_params(graph, algorithm, filename):
        n_paths, paths_explored, nodes_visited, times, _ = EvaluationUtils.get_evaluation_params(graph, algorithm)
        util.save_params_to_pickle(
            n_paths=n_paths,
            nodes_visited=nodes_visited,
            paths_explored=paths_explored,
            times=times,
            filename=filename
        )

    @staticmethod
    def save_evaluation_params_idgs(graph, algorithm, save, heuristic_filename=None):
        idgs = IDGS(graph)
        n_paths, paths_explored, nodes_visited, times, _ = EvaluationUtils.get_evaluation_params_idgs(idgs, algorithm, heuristic_filename)
        util.save_params_to_pickle(
            n_paths=n_paths,
            nodes_visited=nodes_visited,
            paths_explored=paths_explored,
            times=times,
            filename=save
        )

    @staticmethod
    def eval(bfs_filename, dfs_filename, id_filename, lcfs_filename, astar_filename, dfbb_filename, savefig_agg_data, savefig_cet, savefig_ets, save_all_data):

        n_paths_bfs, paths_explored_bfs, nodes_visited_bfs, times_bfs = util.load_params_from_pickle(filename=bfs_filename)
        n_paths_dfs, paths_explored_dfs, nodes_visited_dfs, times_dfs = util.load_params_from_pickle(filename=dfs_filename)
        n_paths_id, paths_explored_id, nodes_visited_id, times_id = util.load_params_from_pickle(filename=id_filename)
        n_paths_lcfs, paths_explored_lcfs, nodes_visited_lcfs, times_lcfs = util.load_params_from_pickle(filename=lcfs_filename)
        n_paths_astar, paths_explored_astar, nodes_visited_astar, times_astar = util.load_params_from_pickle(filename=astar_filename)
        n_paths_dfbb, paths_explored_dfbb, nodes_visited_dfbb, times_dfbb = util.load_params_from_pickle(filename=dfbb_filename)

        nodes_visited = {
            "BFS": nodes_visited_bfs,
            "DFS": nodes_visited_dfs,
            "ID": nodes_visited_id,
            "LCFS": nodes_visited_lcfs,
            "A*": nodes_visited_astar,
            "DFBB": nodes_visited_dfbb
        }

        paths_explored = {
            "BFS": paths_explored_bfs,
            "DFS": paths_explored_dfs,
            "ID": paths_explored_id,
            "LCFS": paths_explored_lcfs,
            "A*": paths_explored_astar,
            "DFBB": paths_explored_dfbb
        }

        EvaluationUtils.plot_aggregated_data(nodes_visited, paths_explored, savefig=savefig_agg_data)

        times_dict = {
            'BFS':times_bfs,
            'DFS':times_dfs,
            'ID':times_id,
            'LCFS':times_lcfs,
            'A*':times_astar,
            'DFBB':times_dfbb
        }

        util.plot_cumulative_execution_times(times_dict, savefig=savefig_cet)
        util.get_execution_time_stats(times_dict, savefig=savefig_ets)

        n_paths = {
            "BFS":n_paths_bfs,
            "DFS":n_paths_dfs, 
            "ID":n_paths_id, 
            'LCFS':n_paths_lcfs, 
            'A*':n_paths_astar, 
            'DFBB':n_paths_dfbb
        }

        EvaluationUtils.check_solutions_found(
            n_paths["BFS"],
            n_paths["DFS"], 
            n_paths["ID"], 
            n_paths["LCFS"], 
            n_paths["A*"], 
            n_paths["DFBB"]
        )

        util.save_params_to_pickle(
            n_paths=n_paths,
            nodes_visited=nodes_visited,
            paths_explored=paths_explored,
            times=times_dict,
            filename=save_all_data
        )

    @staticmethod
    def compare_algorithms(alg1_name, alg1_filename, alg2_name, alg2_filename, savefig_agg_data, savefig_cet, savefig_time_stats):
        _, paths_explored_alg1, nodes_visited_alg1, times_alg1 = util.load_params_from_pickle(filename=alg1_filename)
        _, paths_explored_alg2, nodes_visited_alg2, times_alg2 = util.load_params_from_pickle(filename=alg2_filename)

        nodes_visited = {
            alg1_name:nodes_visited_alg1,
            alg2_name:nodes_visited_alg2
        }

        paths_explored = {
            alg1_name:paths_explored_alg1,
            alg2_name:paths_explored_alg2
        }

        EvaluationUtils.plot_aggregated_data(nodes_visited, paths_explored, savefig=savefig_agg_data)

        times_dict = {
            alg1_name:times_alg1,
            alg2_name:times_alg2
        }

        util.plot_cumulative_execution_times(times_dict, savefig=savefig_cet)
        util.get_execution_time_stats(times_dict, savefig=savefig_time_stats)

if __name__ == '__main__':
    pass







        







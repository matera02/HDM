import pickle
import time
import optuna
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# UTILITÀ PER CSP
NTRIALS = 1000
DIRECTION = 'minimize'
TABU = 'TabuSearch'
class Utility:
    @staticmethod
    # best_value si può riferire o alla fitness dei modelli per l'nsp
    # oppure per i modelli di machine learning e deep learning
    def salva_parametri_modello(algorithm, params, fitness, filename='best_params.pkl'):
        try:
            with open(filename, 'rb') as f:
                best_data = pickle.load(f)
        except FileNotFoundError:
            best_data = {}

        best_data[algorithm] = {'params': params, 'fitness': fitness}

        with open(filename, 'wb') as f:
            pickle.dump(best_data, f)

    @staticmethod
    def carica_parametri_modello(algorithm, filename='best_params.pkl'):
        with open(filename, 'rb') as f:
            best_data = pickle.load(f)
        return best_data.get(algorithm)


    @staticmethod
    def get_optimized_params(study_name, optimization_function, direction=DIRECTION,n_trials=NTRIALS, n_jobs=8):
        study = optuna.create_study(study_name=study_name, direction=direction)
        study.optimize(optimization_function, n_trials=n_trials, n_jobs=n_jobs)
        return study.best_params, study.best_value



    @staticmethod
        #n thread per n file su cui scrivere i risultati
    # i file sono numerati da 1 a n
    def write_file_result(input_dir, output_dir,start_file, end_file, max_workers, function):
        files = [f"{input_dir}/{i}.nsp" for i in range(start_file, end_file + 1)] #ultimo estremo escluso
        best_fitness_values = []

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # FAI ATTENZIONE ALLA FUNZIONE IN INPUT
            futures = [executor.submit(function, file, output_dir) for file in files]
            for future in futures:
                best_fitness_values.append(future.result())
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("All file processed")
        print("Best fitness values: ", best_fitness_values)
        print("Elapsed time: ", elapsed_time)


    @staticmethod
        # output_directory è quella dove stanno scritti i risultati per ciascun algoritmo
    def load_results(output_directory, num_files):
        results = []
        for i in range(1, num_files + 1):
            file = f"{output_directory}/{i}.pkl"
            try:
                with open(file, 'rb') as f:
                    result = pickle.load(f)
                    results.append(result)
                    print(f"Loaded {file}: Best Fitness = {result['best_fitness']}, Execution Time = {result['execution_time']} seconds")
                    #print("Hospital coverage:", result['hospital_coverage'])
                    #print("Nurses preferences:", result['nurse_preferences'])
                    #print("Best schedule:", result['best_schedule'])
                    print("\n")
            except FileNotFoundError:
                print(f"File {file} not found.")
        return results
    

    @staticmethod
    def create_pkl_from_nsp(filename, output_directory, result):
        base_filename = filename.split('/')[-1].replace('.nsp', '.pkl')
        output_filename = f"{output_directory}/{base_filename}"
        with open(output_filename, 'wb') as f:
            pickle.dump(result, f)

    @staticmethod
    def get_items_from_results(source, key, num_files=100):
        results = Utility.load_results(source, num_files)
        return [result[key] for result in results]

# UTILITÀ PER GRAPH

    @staticmethod
    def plot_and_interact_by_floor(G, filename):
        floors = set(room['room'].floor for n, room in G.nodes(data=True))
        color_map = plt.get_cmap('tab20')
        unique_types = list(set(room['room'].name for n, room in G.nodes(data=True)))
        color_dict = {t: color_map(i / len(unique_types)) for i, t in enumerate(unique_types)}

        def draw_floor(floor):
            subG = G.subgraph([n for n, d in G.nodes(data=True) if d['room'].floor == floor])
            pos = {n: (d['room'].x, d['room'].y) for n, d in subG.nodes(data=True)}

            fig, ax = plt.subplots(figsize=(18, 10))
            plt.subplots_adjust(left=0.05, right=0.75)  # Aggiungere margini per la legenda
            nodes = nx.draw_networkx_nodes(subG, pos, node_color=[color_dict[d['room'].name] for n, d in subG.nodes(data=True)],
                                           ax=ax, node_size=500)
            nx.draw_networkx_labels(subG, pos, labels={n: n for n in subG.nodes()}, font_size=8, ax=ax)
            nx.draw_networkx_edges(subG, pos, ax=ax)
            edge_labels = nx.get_edge_attributes(subG, 'weight')
            edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=8, ax=ax)

            # Modifiche alla legenda
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{d["room"].number} - {d["room"].name}', markerfacecolor=color_dict[d["room"].name], markersize=10)
                               for n, d in sorted(subG.nodes(data=True), key=lambda item: item[0])]
            ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Legenda")

            ax.set_title(f"Piano {floor}")
            ax.axis('on')

            dragged_node = [None]

            def on_press(event):
                if event.inaxes != ax: return
                mouse_pos = np.array([event.xdata, event.ydata])
                distances = np.linalg.norm(np.array(list(pos.values())) - mouse_pos, axis=1)
                if np.min(distances) < 0.5:
                    dragged_node[0] = list(pos.keys())[np.argmin(distances)]
                    print(f"Node selected: {dragged_node[0]}")

            def on_release(event):
                if dragged_node[0] is not None:
                    G.nodes[dragged_node[0]]['room'].x = event.xdata
                    G.nodes[dragged_node[0]]['room'].y = event.ydata
                    print(f"Updated position of node {dragged_node[0]} to ({event.xdata}, {event.ydata})")
                    Utility.__update_positions_in_prolog(G, filename)
                    dragged_node[0] = None

            def on_motion(event):
                if dragged_node[0] is not None:
                    pos[dragged_node[0]] = (event.xdata, event.ydata)
                    ax.clear()
                    nodes = nx.draw_networkx_nodes(subG, pos, node_color=[color_dict[d['room'].name] for n, d in subG.nodes(data=True)],
                                                   ax=ax, node_size=500)
                    nx.draw_networkx_labels(subG, pos, labels={n: n for n in subG.nodes()}, font_size=8, ax=ax)
                    nx.draw_networkx_edges(subG, pos, ax=ax)
                    nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=8, ax=ax)
                    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), title="Legenda")
                    fig.canvas.draw()

            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.mpl_connect('button_release_event', on_release)
            fig.canvas.mpl_connect('motion_notify_event', on_motion)
            plt.show()

        for floor in floors:
            draw_floor(floor)

    @staticmethod
    def __update_positions_in_prolog(G, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        with open(filename, 'w') as file:
            for line in lines:
                if line.startswith('x(') or line.startswith('y('):
                    node_id = int(line.split(',')[0][2:])
                    if line.startswith('x('):
                        line = f"x({node_id}, {G.nodes[node_id]['room'].x:.2f}).\n"
                    if line.startswith('y('):
                        line = f"y({node_id}, {G.nodes[node_id]['room'].y:.2f}).\n"
                file.write(line)

    @staticmethod
    def get_room_from_graph(G, room_number):
        if room_number in G.nodes:
            return G.nodes[room_number]['room']
        else:
            print(f"Stanza con numero {room_number} non trovata nel grafo.")
            return None

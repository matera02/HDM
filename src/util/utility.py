import pickle
import time
import optuna
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
import matplotlib.pyplot as plt
from pyswip import Prolog
from src.graph.stanza import Stanza


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

# DA AGGIUNGERE QUELLE FATTE CON PROLOG

    @staticmethod
    def build_graph_from_prolog(filename="src/graph/prova.pl"):
        prolog = Prolog()
        prolog.consult(filename)
        G = nx.DiGraph()
        stanze = list(prolog.query("stanza(X)"))
        for stanza in stanze:
            numero = stanza['X']
            tipo = list(prolog.query(f"tipo({numero}, T)"))[0]['T']
            x = float(list(prolog.query(f"x({numero}, X)"))[0]['X'])
            y = float(list(prolog.query(f"y({numero}, Y)"))[0]['Y'])
            piano = int(list(prolog.query(f"piano({numero}, P)"))[0]['P'])
            stanza_obj = Stanza(numero, tipo, x, y, piano)
            G.add_node(numero, stanza=stanza_obj)
        
        connessioni = list(prolog.query("connessione(A, B, Peso), Peso \= '_', A \= B"))
        for conn in connessioni:
            peso = float(conn['Peso'])
            G.add_edge(conn['A'], conn['B'], weight=peso)
        
        return G
    
    @staticmethod
    def __save_positions_to_prolog_file(G, pos, filename="src/graph/prova.pl"):
        with open(filename, "r") as f:
            lines = f.readlines()
        
        for node, (x, y) in pos.items():
            x_line = next((i for i, line in enumerate(lines) if line.startswith(f"x({node},")) or None)
            y_line = next((i for i, line in enumerate(lines) if line.startswith(f"y({node},")) or None)
            
            # con aggiunta controllo della posizione del nodo
            if x_line is not None:
                lines[x_line] = f"x({node}, {max(0, x):.2f}).\n"  # Assicura che x non sia negativo
            if y_line is not None:
                lines[y_line] = f"y({node}, {max(0, y):.2f}).\n"  # Assicura che y non sia negativo
        
        with open(filename, "w") as f:
            f.writelines(lines)

    @staticmethod
    def plot_and_interact_graph(G, title="Ospedale"):
        pos = {node: (data['stanza'].x, data['stanza'].y) for node, data in G.nodes(data=True)}
        fig, ax = plt.subplots(figsize=(20, 20))
        
        def draw_graph():
            ax.clear()
            nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", 
                    font_size=10, font_weight="bold", edge_color="gray", arrows=True, ax=ax)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
            plt.title(title)
            # Controllo sul posizionamento dei nodi
            ax.set_xlim(left=0)  # Imposta il limite sinistro a 0
            ax.set_ylim(bottom=0)  # Imposta il limite inferiore a 0

            fig.canvas.draw()

        draw_graph()

        dragged_node = None

        def on_press(event):
            nonlocal dragged_node
            if event.inaxes == ax:
                dragged_node = min(pos, key=lambda n: ((pos[n][0]-event.xdata)**2 + (pos[n][1]-event.ydata)**2))

        def on_release(event):
            nonlocal dragged_node
            if dragged_node:
                G.nodes[dragged_node]['stanza'].x = pos[dragged_node][0]
                G.nodes[dragged_node]['stanza'].y = pos[dragged_node][1]
            dragged_node = None

        def on_motion(event):
            if dragged_node is not None:
                #controllo sul posizionamento del nodo
                new_x = max(0, event.xdata)  # Assicura che x non sia negativo
                new_y = max(0, event.ydata)  # Assicura che y non sia negativo
                
                pos[dragged_node] = (event.xdata, event.ydata)
                draw_graph()

        def on_close(event):
            Utility.__save_positions_to_prolog_file(G, pos)

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('close_event', on_close)
        plt.show()

import networkx as nx
import matplotlib.pyplot as plt
from pyswip import Prolog
import numpy as np
import src.graph.graph as gr
from src.graph.stanza import Stanza

# TENTATIVO PIÙ INTELLIGENTE E FUNZIONANTE DI RAPPRESENTARE IL GRAFO CON PIANI E LEGENDA

class GrafoStanze:
    @staticmethod
    def build_graph_from_prolog(filename):
        prolog = Prolog()
        prolog.consult(filename)
        G = nx.DiGraph()
        stanze = list(prolog.query("stanza(X)"))
        for stanza in stanze:
            numero = stanza['X']
            details = next(prolog.query(f"tipo({numero}, T), x({numero}, X), y({numero}, Y), piano({numero}, P)"))
            stanza_obj = Stanza(numero, details['T'], float(details['X']), float(details['Y']), int(details['P']))
            G.add_node(numero, stanza=stanza_obj)
        
        connessioni = list(prolog.query("connessione(A, B, Peso)"))
        for conn in connessioni:
            G.add_edge(conn['A'], conn['B'], weight=float(conn['Peso']))
        
        return G

    @staticmethod
    def plot_and_interact_by_floor(G, filename):
        floors = set(stanza['stanza'].piano for n, stanza in G.nodes(data=True))
        color_map = plt.get_cmap('tab20')
        unique_types = list(set(stanza['stanza'].tipo for n, stanza in G.nodes(data=True)))
        color_dict = {t: color_map(i / len(unique_types)) for i, t in enumerate(unique_types)}

        def draw_floor(floor):
            subG = G.subgraph([n for n, d in G.nodes(data=True) if d['stanza'].piano == floor])
            pos = {n: (d['stanza'].x, d['stanza'].y) for n, d in subG.nodes(data=True)}

            fig, ax = plt.subplots(figsize=(18, 10))
            plt.subplots_adjust(left=0.05, right=0.75)  # Aggiungere margini per la legenda
            nodes = nx.draw_networkx_nodes(subG, pos, node_color=[color_dict[d['stanza'].tipo] for n, d in subG.nodes(data=True)],
                                           ax=ax, node_size=500)
            nx.draw_networkx_labels(subG, pos, labels={n: n for n in subG.nodes()}, font_size=8, ax=ax)
            nx.draw_networkx_edges(subG, pos, ax=ax)
            edge_labels = nx.get_edge_attributes(subG, 'weight')
            edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
            nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_size=8, ax=ax)

            # Modifiche alla legenda
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'{d["stanza"].numero} - {d["stanza"].tipo}', markerfacecolor=color_dict[d["stanza"].tipo], markersize=10)
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
                    G.nodes[dragged_node[0]]['stanza'].x = event.xdata
                    G.nodes[dragged_node[0]]['stanza'].y = event.ydata
                    print(f"Updated position of node {dragged_node[0]} to ({event.xdata}, {event.ydata})")
                    GrafoStanze.update_positions_in_prolog(G, filename)
                    dragged_node[0] = None

            def on_motion(event):
                if dragged_node[0] is not None:
                    pos[dragged_node[0]] = (event.xdata, event.ydata)
                    ax.clear()
                    nodes = nx.draw_networkx_nodes(subG, pos, node_color=[color_dict[d['stanza'].tipo] for n, d in subG.nodes(data=True)],
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
    def update_positions_in_prolog(G, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        with open(filename, 'w') as file:
            for line in lines:
                if line.startswith('x(') or line.startswith('y('):
                    node_id = int(line.split(',')[0][2:])
                    if line.startswith('x('):
                        line = f"x({node_id}, {G.nodes[node_id]['stanza'].x:.2f}).\n"
                    if line.startswith('y('):
                        line = f"y({node_id}, {G.nodes[node_id]['stanza'].y:.2f}).\n"
                file.write(line)

    @staticmethod
    def get_stanza_from_graph(G, numero_stanza):
        if numero_stanza in G.nodes:
            return G.nodes[numero_stanza]['stanza']
        else:
            print(f"Stanza con numero {numero_stanza} non trovata nel grafo.")
            return None
        

# dfs con cycle pruning e multiple path pruning
def dfs_pruning(graph, start, goal):
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

from queue import PriorityQueue

def AStarSearch_MPP(graph, start, goal, node_labels):
    print('AStar Search with Multiple Path Pruning: ')
    frontier = PriorityQueue()
    visited = {}  # Dizionario per tenere traccia dei nodi visitati e dei loro costi
    g_costs = {start: 0}  # Dizionario per i costi g
    priority = get_f(graph, start, node_labels)
    frontier.put((priority, start))
    came_from = {}  # Dizionario per ricostruire il percorso

    while not frontier.empty():
        current_f, current = frontier.get()

        if current == goal:
            return reconstruct_path(came_from, start, goal)

        if current in visited and visited[current] <= current_f:
            continue

        visited[current] = current_f

        if current in graph:
            for adj in graph.neighbors(current):
                new_g = g_costs[current] + graph[current][adj]['weight']
                if adj not in g_costs or new_g < g_costs[adj]:
                    g_costs[adj] = new_g
                    f = new_g + get_h(adj, goal, node_labels)
                    frontier.put((f, adj))
                    came_from[adj] = current
        else:
            print(f"Il nodo {current} non è presente nel grafo.")

    return None

def get_f(graph, node, node_labels):
    # Implementa questa funzione per calcolare f = g + h
    pass

def get_h(node, goal, node_labels):
    # Implementa questa funzione per calcolare l'euristica h
    pass

def reconstruct_path(came_from, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path



if __name__ == "__main__":
    filename = "src/graph/prova.pl"
    G = GrafoStanze.build_graph_from_prolog(filename)
    GrafoStanze.plot_and_interact_by_floor(G, filename)

    #nodi = G.nodes()
    #for node in nodi:
    #    print(type(node))

    numero_stanza = 101  # Sostituisci con il numero della stanza che ti interessa

    # Verifica se il nodo esiste nel grafo
    if numero_stanza in G.nodes:
        stanza = G.nodes[numero_stanza]['stanza']
        print(stanza)
    else:
        print(f"Stanza con numero {numero_stanza} non trovata nel grafo.")

    # HO RISOLTO IL PROBLEMA CHE AVEVO IERI CON I NODI CHE ERANO OGGETTO
    # ORA PER FARE RIFERIMENTO AD UN NODO MI BASTA IL NUMERO DEL NODO
    path_bfs = gr.bfs(G, 101, 320)
    #path_dfs = gr.dfs(G, 101, 202)
    path_id = gr.IterativeDeepening(G, 101, 320)
    path_lcfs = gr.lowestCostSearch(G, 101, 202)

    print("Path trovato bfs: ", path_bfs, "\n")

    #print("Path trovato dfs: ", path_dfs, "\n")

    print("Path trovato id: ", path_id, "\n")
    print("Path trovato lcfs: ", path_lcfs, "\n")


    # DEVO PREVEDERE UNA STRATEGIA DI PATH PRUNING e CYCLE PRUNING PER DFS
    path_dfs = dfs_pruning(G, 101, 320)
    print("Path trovato dfs: ", path_dfs, "\n")

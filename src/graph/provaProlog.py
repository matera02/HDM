from pyswip import Prolog
import networkx as nx
import matplotlib.pyplot as plt
from src.util.utility import Utility as util
from src.graph.stanza import Stanza

# TENTATIVO DISPERATO DI RAPPRESENTARE L'INTERO GRAFO DELL'OSPEDALE

class GrafoStanze:
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
    def plot_and_interact_graph(G, title="Grafo delle Stanze"):
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

            # MODIFICHE
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
                #MODIFICHE
                new_x = max(0, event.xdata)  # Mi assicuro che x non sia negativo
                new_y = max(0, event.ydata)  # Mi assicuro che y non sia negativo
                pos[dragged_node] = (event.xdata, event.ydata)
                draw_graph()

        def on_close(event):
            GrafoStanze.__save_positions_to_prolog_file(G, pos)

        fig.canvas.mpl_connect('button_press_event', on_press)
        fig.canvas.mpl_connect('button_release_event', on_release)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('close_event', on_close)
        plt.show()

    @staticmethod
    def __save_positions_to_prolog_file(G, pos, filename="src/graph/prova.pl"):
        with open(filename, "r") as f:
            lines = f.readlines()
        
        for node, (x, y) in pos.items():
            x_line = next((i for i, line in enumerate(lines) if line.startswith(f"x({node},")) or None)
            y_line = next((i for i, line in enumerate(lines) if line.startswith(f"y({node},")) or None)
            
            if x_line is not None:
                lines[x_line] = f"x({node}, {max(0, x):.2f}).\n"  # Mi assicuro che x non sia negativo
            if y_line is not None:
                lines[y_line] = f"y({node}, {max(0, y):.2f}).\n"  # Mi assicuro che y non sia negativo
        
        
        with open(filename, "w") as f:
            f.writelines(lines)



if __name__ == "__main__":
    #graph = GrafoStanze.build_graph_from_prolog()
    #GrafoStanze.plot_and_interact_graph(graph)
    g = util.build_graph_from_prolog()
    util.plot_and_interact_graph(g)

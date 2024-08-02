import matplotlib.pyplot as plt
import networkx as nx
from src.graph.stanza import Stanza
import src.graph.graph as gr
from queue import PriorityQueue
import sys
import src.graph.dynprog as dp
import mplcursors


def get_node(na, nodes):
    for node in nodes:
        # Controllo se il valore di 'na' è un intero e il nodo ha l'attributo 'numero'
        if isinstance(na, int) and hasattr(node, 'numero'):
            if node.numero == na:
                return node
        # Controllo se il valore di 'na' è una stringa e il nodo ha l'attributo 'tipo'
        elif isinstance(na, str) and hasattr(node, 'tipo'):
            if node.tipo == na:
                return node
    return None

# non dovrebbe esserci nessun cambiamento
def get_cost(graph, path):
    cost = 0
    for i in range(len(path) - 1):
        current = path[i]
        next = path[i + 1]
        if graph.has_edge(current, next):
            peso = graph.get_edge_data(current, next)['peso']
            cost += peso
        else:
            return -1
    return cost

def get_heuristic(node):
    if isinstance(node, Stanza):
        return node.heuristic
        
def get_f(graph, path):
    costo = get_cost(graph, path)
    if costo < 0:
        return None
    return costo + get_heuristic(path[-1])



def get_positions(nodes):
    positions = {}
    for node in nodes:
        positions[node] = node.get_pos()
    return positions

def get_node_by_position(pos, x, y):
    for node, (px, py) in pos.items():
        if abs(px - x) < 0.1 and abs(py - y) < 0.1:
            return node
    return None

def save_positions_to_file(positions, filename="node_positions.txt"):
    with open(filename, "w") as f:
        for node, (x, y) in positions.items():
            f.write(f"{node.get_id()} {node.tipo} {x} {y}\n")

def plot_and_interact_graph(G, pos, title, filename="node_positions.txt"):
    fig, ax = plt.subplots(figsize=(20, 20))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", arrows=True, ax=ax)
    plt.title(title)

    dragged_node = None

    def on_press(event):
        nonlocal dragged_node
        if event.inaxes == ax:
            dragged_node = get_node_by_position(pos, event.xdata, event.ydata)

    def on_release(event):
        nonlocal dragged_node
        dragged_node = None

    def on_motion(event):
        if dragged_node is not None:
            pos[dragged_node] = (event.xdata, event.ydata)
            ax.clear()
            nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", arrows=True, ax=ax)
            plt.title(title)
            fig.canvas.draw()

    def on_close(event):
        save_positions_to_file(pos, filename)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()



# assumo che sia diretto da a a b 
def get_edge(na, nb, nodes):
    a = get_node(na, nodes)
    b = get_node(nb, nodes)
    return (a, b)

def get_weighted_edge(na, nb, w, nodes):
    a = get_node(na, nodes)
    b = get_node(nb, nodes)
    return (a, b, {'peso': w})

def get_positions(nodes):
    positions = {}
    for node in nodes:
        positions[node] = node.get_pos()
    return positions


# 1° PIANO
def get_piano1():

    # Definisco i nodi
    nodes = [
        Stanza(101, "Entry", 5, 10, 1),
        Stanza(102, "Reception Room New Patient", 4, 9, 1),
        Stanza(103, "Reception Room Hygiene Patient", 6, 9, 1),
        Stanza(104, "Business Office", 5, 8, 1),
        Stanza(105, "Consult", 5, 7, 1),
        Stanza(106, "Operatory 1", 4, 7, 1),
        Stanza(107, "Operatory 2", 4, 6, 1),
        Stanza(108, "Operatory 3", 4, 5, 1),
        Stanza(109, "Operatory 4", 3, 5, 1),
        Stanza(110, "Operatory 5", 3, 4, 1),
        Stanza(111, "Operatory 6", 3, 3, 1),
        Stanza(112, "Operatory 7", 3, 2, 1),
        Stanza(113, "Operatory 8", 3, 1, 1),
        Stanza(114, "Operatory 9", 4, 1, 1),
        Stanza(115, "Operatory 10", 5, 1, 1),
        Stanza(116, "Darkroom", 6, 1, 1),
        Stanza(117, "Lab", 6, 2, 1),
        Stanza(118, "X-ray", 6, 3, 1),
        Stanza(119, "Central Tray Prep", 5, 3, 1),
        Stanza(120, "Elev. Equip", 5, 4, 1),
        Stanza(121, "Storage", 5, 5, 1),
        Stanza(122, "Women's", 6, 5, 1),
        Stanza(123, "Men's", 6, 6, 1),
        Stanza(124, "Stairs", 6, 7, 1),
        Stanza(125, "Nit. 2", 7, 7, 1),
        Stanza(126, "Nit. 1", 7, 6, 1),
        Stanza(127, "Store 1", 8, 6, 1),
        Stanza(128, "Store 2", 7, 5, 1),
        Stanza(129, "Break Room", 6, 8, 1)
    ]

    # Definisco gli archi
    edges = [
        get_edge("Entry", "Reception Room New Patient", nodes),
        get_edge("Entry", "Reception Room Hygiene Patient", nodes),
        get_edge("Reception Room New Patient", "Business Office", nodes),
        get_edge("Reception Room Hygiene Patient", "Business Office", nodes),
        get_edge("Business Office", "Consult", nodes),
        get_edge("Consult", "Operatory 1", nodes),
        get_edge("Consult", "Operatory 2", nodes),
        get_edge("Consult", "Operatory 3", nodes),
        get_edge("Operatory 1", "Operatory 2", nodes),
        get_edge("Operatory 2", "Operatory 3", nodes),
        get_edge("Operatory 3", "Operatory 4", nodes),
        get_edge("Operatory 4", "Operatory 5", nodes),
        get_edge("Operatory 5", "Operatory 6", nodes),
        get_edge("Operatory 6", "Operatory 7", nodes),
        get_edge("Operatory 7", "Operatory 8", nodes),
        get_edge("Operatory 8", "Operatory 9", nodes),
        get_edge("Operatory 9", "Operatory 10", nodes),
        get_edge("Operatory 10", "Central Tray Prep", nodes),
        get_edge("Darkroom", "Lab", nodes),
        get_edge("Lab", "X-ray", nodes),
        get_edge("X-ray", "Central Tray Prep", nodes),
        get_edge("Central Tray Prep", "Elev. Equip", nodes),
        get_edge("Elev. Equip", "Storage", nodes),
        get_edge("Storage", "Stairs", nodes),
        get_edge("Stairs", "Women's", nodes),
        get_edge("Stairs", "Men's", nodes),
        get_edge("Storage", "Break Room", nodes),
        get_edge("Break Room", "Nit. 1", nodes),
        get_edge("Nit. 1", "Nit. 2", nodes),
        get_edge("Nit. 1", "Store 1", nodes),
        get_edge("Nit. 2", "Store 2", nodes),
        get_edge("Darkroom", "X-ray", nodes),
        #aggiungo come collegamento da 115 a 116 per non avere un'isola
        get_edge("Operatory 10", "Darkroom", nodes)

        # fai attenzione al grafico
        #get_edge("Operatory 4", "Storage", nodes)
    ]

    # Posiziono manualmente le posizioni dei nodi del grafo
    pos = get_positions(nodes)

    #print(pos)

    # Diretto
    # Converto il grafo di prima in diretto
    G = nx.DiGraph()

    # Aggiungo i nodi al grafo diretto
    for node in nodes:
        G.add_node(node)

    # Aggiungo gli archi al grafo diretto
    G.add_edges_from(edges)

    # Plot del grafo diretto
    plot_and_interact_graph(G, pos, "1° piano", "src/graph/data/node_positions_P1.txt")

    return G, nodes, edges

def get_piano2():
    # 2° PIANO
    G2 = nx.DiGraph()

    # Definisco i nodi del secondo piano
    nodes2 = [
        Stanza(201, "Entry", 5, 11, 2),
        Stanza(202, "Stairs", 8, 0, 2),
        Stanza(203, "Patient Room 1", 0, 10, 2), 
        Stanza(204, "Patient Room 2", 1, 10, 2),
        Stanza(205, "Patient Room 3", 2, 10, 2),
        Stanza(206, "Patient Room 4", 3, 10, 2),
        Stanza(207, "Patient Room 5", 4, 10, 2),
        Stanza(208, "Patient Room 6", 5, 10, 2),
        Stanza(209, "Patient Room 7", 6, 10, 2),
        Stanza(210, "Patient Room 8", 7, 10, 2),
        Stanza(211, "Patient Room 9", 8, 10, 2),
        Stanza(212, "Patient Room 10", 9, 10, 2),
        Stanza(213, "Patient Room 11", 10, 10, 2),
        Stanza(214, "Patient Room 12", 11, 10, 2),
        Stanza(215, "Patient Room 13", 12, 10, 2),
        Stanza(216, "Patient Room 14", 13, 10, 2),
        Stanza(217, "Single Patient Room 1", 0, 9, 2),
        Stanza(218, "Single Patient Room 2", 1, 9, 2),
        Stanza(219, "Single Patient Room 3", 2, 9, 2),
        Stanza(220, "Toilet Room 1", 4, 9, 2),
        Stanza(221, "Toilet Room 2", 5, 9, 2),
        Stanza(222, "Toilet Room 3", 6, 9, 2),
        Stanza(223, "Shower Room 1", 4, 8, 2),
        Stanza(224, "Shower Room 2", 5, 8, 2),
        Stanza(225, "Guest Room", 2, 8, 2),
        Stanza(226, "Mech Room", 6, 8, 2),
        Stanza(227, "Office 1", 8, 9, 2),
        Stanza(228, "Office 2", 9, 9, 2),
        Stanza(229, "Counseling", 10, 9, 2),
        Stanza(230, "Intake", 11, 9, 2),
        Stanza(231, "Medical Storage", 9, 8, 2),
        Stanza(232, "Records", 10, 8, 2),
        Stanza(233, "Employee Room", 9, 7, 2),
        Stanza(234, "Study Library", 12, 9, 2),
        Stanza(235, "Group Patient Room", 13, 9, 2),
        Stanza(236, "Activity Room", 14, 9, 2),
        Stanza(237, "Storage Room", 15, 9, 2)
    ]

    # Aggiungo i nodi al grafo
    for node in nodes2:
        G2.add_node(node)

    # Definisco gli archi del secondo piano
    edges2 = [
        get_edge("Patient Room 1", "Patient Room 2", nodes2),
        get_edge("Patient Room 2", "Patient Room 3", nodes2), 
        get_edge("Patient Room 3", "Patient Room 4", nodes2), 
        get_edge("Patient Room 4", "Patient Room 5", nodes2), 
        get_edge("Patient Room 5", "Patient Room 6", nodes2), 
        get_edge("Patient Room 6", "Patient Room 7", nodes2), 
        get_edge("Patient Room 7", "Patient Room 8", nodes2), 
        get_edge("Patient Room 8", "Patient Room 9", nodes2), 
        get_edge("Patient Room 9", "Patient Room 10", nodes2), 
        get_edge("Patient Room 10", "Patient Room 11", nodes2), 
        get_edge("Patient Room 11", "Patient Room 12", nodes2), 
        get_edge("Patient Room 12", "Patient Room 13", nodes2), 
        get_edge("Patient Room 13", "Patient Room 14", nodes2),
        get_edge("Single Patient Room 1", "Single Patient Room 2", nodes2), 
        get_edge("Single Patient Room 2", "Single Patient Room 3", nodes2),
        get_edge("Toilet Room 1", "Toilet Room 2", nodes2), 
        get_edge("Toilet Room 2", "Toilet Room 3", nodes2),
        get_edge("Shower Room 1", "Shower Room 2", nodes2),
        get_edge("Guest Room", "Toilet Room 1", nodes2), 
        get_edge("Mech Room", "Shower Room 2", nodes2), 
        get_edge("Office 1", "Office 2", nodes2), 
        get_edge("Office 2", "Employee Room", nodes2),
        get_edge("Counseling", "Intake", nodes2), 
        get_edge("Medical Storage", "Records", nodes2), 
        get_edge("Study Library", "Group Patient Room", nodes2), 
        get_edge("Group Patient Room", "Activity Room", nodes2), 
        get_edge("Activity Room", "Storage Room", nodes2),
        get_edge("Entry", "Patient Room 1", nodes2), 
        get_edge("Entry", "Patient Room 5", nodes2), 
        get_edge("Entry", "Guest Room", nodes2),
        get_edge("Entry", "Office 1", nodes2), 
        get_edge("Entry", "Study Library", nodes2),
        get_edge("Storage Room", "Stairs", nodes2),
        # Le seguenti sono aggiunte per rimuovere altre isole
        get_edge(203, 217, nodes2), 
        get_edge(204, 218, nodes2),
        get_edge(205, 219, nodes2),
        get_edge(219, 225, nodes2),
        get_edge(220, 223, nodes2),
        get_edge(221, 224, nodes2),
        get_edge(222, 226, nodes2),
        get_edge(228, 229, nodes2),
        get_edge(230, 232, nodes2),
        get_edge(227, 231, nodes2) 
    ]

    # Aggiungo gli archi al grafo
    G2.add_edges_from(edges2)

    # Posizionamento manuale dei nodi del grafo
    pos2 = get_positions(nodes2)

    # Plot del grafo
    plot_and_interact_graph(G2, pos2, "2° Piano", "src/graph/data/node_positions_P2.txt")
    return G2, nodes2, edges2

def get_piano3():
    # 3° PIANO

    # Definisco tutti i nodi e le connessioni
    nodes3 = [
        Stanza(301, "Entry", 1, 0, 3), 
        Stanza(302, "Minor Procedures", 5, -0.5, 3),
        Stanza(303, "Exam 1", 6, 0.3, 3),
        Stanza(304, "Exam 2", 7, 0.3, 3),
        Stanza(305, "Exam 3", 7, -0.3, 3),
        Stanza(306, "Drug/Supply", 4, -0.5, 3),
        Stanza(307, "Doctor Office 1", 3, -0.5, 3),
        Stanza(308, "Waiting Area", 2, 0, 3),
        Stanza(309, "Business Office", 3, 0, 3),
        Stanza(310, "Lab 1", 4, 0.5, 3),
        Stanza(311, "Lab 2", 5, 0.5, 3),
        Stanza(312, "Nurses Station", 4.5, 0, 3), 
        Stanza(313, "Break Room", 4, -1, 3),
        Stanza(314, "Staff Room", 3, -1, 3),
        Stanza(315, "Doctor Office 2", 5, -1, 3),
        Stanza(316, "Multi-Purpose", 6, -1, 3),
        Stanza(317, "X-ray Suite", 7, -1, 3),
        Stanza(318, "Control Area", 8, -1, 3),
        Stanza(319, "Supply Closet", 6, 0.7, 3),
        Stanza(320, "Stairs", 9, -1, 3)
    ]

    edges3 = [
        get_edge("Entry", "Waiting Area", nodes3),
        get_edge("Waiting Area", "Business Office", nodes3),
        get_edge("Business Office", "Lab 1", nodes3),
        get_edge("Lab 1", "Lab 2", nodes3),
        get_edge("Lab 2", "Nurses Station", nodes3),
        get_edge("Nurses Station", "Exam 1", nodes3),
        get_edge("Exam 1", "Exam 2", nodes3),
        get_edge("Exam 2", "Exam 3", nodes3),
        get_edge("Exam 3", "Minor Procedures", nodes3),
        get_edge("Minor Procedures", "Drug/Supply", nodes3),
        get_edge("Drug/Supply", "Doctor Office 1", nodes3),
        get_edge("Doctor Office 1", "Staff Room", nodes3),
        get_edge("Staff Room", "Break Room", nodes3),
        get_edge("Break Room", "Doctor Office 2", nodes3),
        get_edge("Doctor Office 2", "Multi-Purpose", nodes3),
        get_edge("Multi-Purpose", "X-ray Suite", nodes3),
        get_edge("X-ray Suite", "Control Area", nodes3),
        get_edge("Control Area", "Stairs", nodes3),
        get_edge("Supply Closet", "Exam 1", nodes3),
        # Aggiunta nodi per rimozioni isole
        get_edge(311, 319, nodes3),
        get_edge(302, 316, nodes3),
        # Ulteriori aggiunte
        get_edge(309, 307, nodes3),
        get_edge(318, 302, nodes3),
        get_edge(304, 319, nodes3),
        get_edge(312, 319, nodes3),
        get_edge(313, 306, nodes3),
        get_edge(305, 312, nodes3)
    ]

    pos3 = get_positions(nodes3)

    # Creating the directed graph
    G3 = nx.DiGraph()
    G3.add_nodes_from(nodes3)
    G3.add_edges_from(edges3)

    # Plotting the graph with accurate node positions
    plot_and_interact_graph(G3, pos3, "3° Piano", "src/graph/data/node_positions_P3.txt")
    return G3, nodes3, edges3

def get_piano1_pesato():
    # 1° PIANO
    # Definisco i nodi
    nodes = [
        # bisognava assegnarli in maniera più sensata ma si tratta di un tentativo
        Stanza(101, "Entry", 5, 10, 1, heuristic=27),
        Stanza(102, "Reception Room New Patient", 4, 9, 1, heuristic=26),
        Stanza(103, "Reception Room Hygiene Patient", 6, 9, 1, heuristic=25),
        Stanza(104, "Business Office", 5, 8, 1, heuristic=24),
        Stanza(105, "Consult", 5, 7, 1, heuristic=23),
        Stanza(106, "Operatory 1", 4, 7, 1, heuristic=23),
        Stanza(107, "Operatory 2", 4, 6, 1, heuristic=22),
        Stanza(108, "Operatory 3", 4, 5, 1, heuristic=21),
        Stanza(109, "Operatory 4", 3, 5, 1, heuristic=20),
        Stanza(110, "Operatory 5", 3, 4, 1, heuristic=19),
        Stanza(111, "Operatory 6", 3, 3, 1, heuristic=18),
        Stanza(112, "Operatory 7", 3, 2, 1, heuristic=17),
        Stanza(113, "Operatory 8", 3, 1, 1, heuristic=16),
        Stanza(114, "Operatory 9", 4, 1, 1, heuristic=15),
        Stanza(115, "Operatory 10", 5, 1, 1 ,heuristic=14),
        Stanza(116, "Darkroom", 6, 1, 1, heuristic=13),
        Stanza(117, "Lab", 6, 2, 1, heuristic=12),
        Stanza(118, "X-ray", 6, 3, 1, heuristic=11),
        Stanza(119, "Central Tray Prep", 5, 3, 1, heuristic=10),
        Stanza(120, "Elev. Equip", 5, 4, 1, heuristic=9),
        Stanza(121, "Storage", 5, 5, 1, heuristic=8),
        Stanza(122, "Women's", 6, 5, 1, heuristic=7),
        Stanza(123, "Men's", 6, 6, 1, heuristic=6),
        Stanza(124, "Stairs", 6, 7, 1, heuristic=5),
        Stanza(125, "Nit. 2", 7, 7, 1, heuristic=4),
        Stanza(126, "Nit. 1", 7, 6, 1, heuristic=3),
        Stanza(127, "Store 1", 8, 6, 1, heuristic=0),
        Stanza(128, "Store 2", 7, 5, 1, heuristic=2),
        Stanza(129, "Break Room", 6, 8, 1, heuristic=1)
    ]

    # Definisco gli archi
    edges = [
        get_weighted_edge("Entry", "Reception Room New Patient", 1 ,nodes),
        get_weighted_edge("Entry", "Reception Room Hygiene Patient", 1, nodes),
        get_weighted_edge("Reception Room New Patient", "Business Office", 1, nodes),
        get_weighted_edge("Reception Room Hygiene Patient", "Business Office", 1, nodes),
        get_weighted_edge("Business Office", "Consult", 1, nodes),
        get_weighted_edge("Consult", "Operatory 1", 1, nodes),
        get_weighted_edge("Consult", "Operatory 2", 1, nodes),
        get_weighted_edge("Consult", "Operatory 3", 1, nodes),
        get_weighted_edge("Operatory 1", "Operatory 2", 1, nodes),
        get_weighted_edge("Operatory 2", "Operatory 3", 1, nodes),
        get_weighted_edge("Operatory 3", "Operatory 4", 1, nodes),
        get_weighted_edge("Operatory 4", "Operatory 5", 1, nodes),
        get_weighted_edge("Operatory 5", "Operatory 6", 1, nodes),
        get_weighted_edge("Operatory 6", "Operatory 7", 1, nodes),
        get_weighted_edge("Operatory 7", "Operatory 8", 1, nodes),
        get_weighted_edge("Operatory 8", "Operatory 9", 1, nodes),
        get_weighted_edge("Operatory 9", "Operatory 10", 1, nodes),
        get_weighted_edge("Operatory 10", "Central Tray Prep", 1, nodes),
        get_weighted_edge("Darkroom", "Lab", 1, nodes),
        get_weighted_edge("Lab", "X-ray", 1, nodes),
        get_weighted_edge("X-ray", "Central Tray Prep", 1, nodes),
        get_weighted_edge("Central Tray Prep", "Elev. Equip", 1, nodes),
        get_weighted_edge("Elev. Equip", "Storage", 1,  nodes),
        get_weighted_edge("Storage", "Stairs", 1, nodes),
        get_weighted_edge("Stairs", "Women's", 1, nodes),
        get_weighted_edge("Stairs", "Men's", 1, nodes),
        get_weighted_edge("Storage", "Break Room", 1, nodes),
        get_weighted_edge("Break Room", "Nit. 1", 1, nodes),
        get_weighted_edge("Nit. 1", "Nit. 2", 1, nodes),
        get_weighted_edge("Nit. 1", "Store 1", 1, nodes),
        get_weighted_edge("Nit. 2", "Store 2", 1, nodes),
        get_weighted_edge("Darkroom", "X-ray", 1, nodes),
        # fai attenzione al grafico
        #get_edge("Operatory 4", "Storage", nodes)
        get_edge("Operatory 10", "Darkroom", nodes)
    ]

    # Posiziono manualmente le posizioni dei nodi del grafo
    pos = get_positions(nodes)

    #print(pos)

    # Diretto
    # Converto il grafo di prima in diretto
    G = nx.DiGraph()

    # Aggiungo i nodi al grafo diretto
    for node in nodes:
        G.add_node(node)

    # Aggiungo gli archi al grafo diretto
    G.add_edges_from(edges)

    # Plot del grafo diretto
    #plt.figure(figsize=(20, 20))
    #nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", arrows=True)
    #plt.title("1° piano")
    #plt.show()

    return G, nodes, edges

#prova con algoritmi con strategia uninformed
def prova1():
    g1, n1, e1 = get_piano1()
    g2, n2, e2 = get_piano2()
    g3, n3, e3 = get_piano3()

    start = get_node(101, n1)
    goal = get_node(127, n1)

    print("Path trovato: ", gr.bfs(g1, start, goal), "\n")
    print("Path trovato: ", gr.dfs(g1, start, goal), "\n")
    print("Path trovato: ", gr.IterativeDeepening(g1, start, goal), "\n")

# ho dovuto riadattare la lowestcostsearch perché non andava con le stanze
def lowestCostSearch(graph, start, goal):
    print('Lowest Cost Search: ')
    frontier = PriorityQueue()
    visited = set()
    frontier.put((0, start.get_id(), [start.get_id()]))

    while not frontier.empty():
        priority, current_id, path = frontier.get()
        print(priority, path)
        print(current_id)

        if current_id == goal.get_id():
            return path

        if current_id not in visited:
            visited.add(current_id)
            current = get_node(current_id, graph.nodes)
            if current in graph:
                for adj in graph.neighbors(current):
                    try:
                        peso = graph.get_edge_data(current, adj)['peso']
                        print(peso)
                        frontier.put((priority + peso, adj.get_id(), path + [adj.get_id()]))
                    except KeyError:
                        pass
            else:
                print(f"Il nodo {current_id} non è presente nel grafo.")
    return None

# anche l'A-Star deve essere riadattato per funzionare con oggetti Stanza
def AStarSearch(graph, start, goal):
    print('AStar Search: ')
    frontier = PriorityQueue()
    visited = set()
    start_id = start.get_id()
    goal_id = goal.get_id()

    # Calcola il valore f iniziale per il nodo di partenza
    priority = get_f(graph, [start])
    frontier.put((priority, start_id, [start_id]))

    while not frontier.empty():
        priority, current_id, path = frontier.get()

        print(priority, path)
        print(current_id)

        if current_id == goal_id:
            return path  # Restituisce il percorso come lista di ID

        if current_id not in visited:
            visited.add(current_id)
            current = get_node(current_id, graph.nodes)  # Recupera l'oggetto stanza usando l'ID

            for adj in graph.neighbors(current):
                adj_id = adj.get_id()
                if adj_id not in visited:
                    new_path = path + [adj_id]
                    new_priority = get_f(graph, [get_node(node_id, graph.nodes) for node_id in new_path])
                    if new_priority is not None:  # Assicurarsi che new_priority non sia None
                        frontier.put((new_priority, adj_id, new_path))

    return None


# da usare questo
def DF_branch_and_bound(graph, start, goal):

    def cbsearch(graph, path, goal, bound, frontier):
        current = path[-1]
        if get_f(graph, path) < bound:
            if current == goal:
                bound = get_cost(graph, path)
            else:
                for adj in graph.neighbors(current):
                    new_bound = get_f(graph, path + [adj])
                    frontier.append((new_bound, (adj, path + [adj])))
    frontier = []
    cost = get_f(graph, [start])
    frontier.append((cost, (start, [start])))
    bound = sys.maxsize  ##inizialmente bound = massimo numero rappresentabile
    while True:
        print(frontier)
        cost, (current, path) = frontier.pop()
        if current == goal:
            return path
        if cost < bound:
            cbsearch(graph, path, goal, bound,frontier)


def provalcfs():
    g1, n1, e1 = get_piano1_pesato()
    start = get_node(101, n1)
    boh = get_node(102, n1)
    goal = get_node(127, n1)
    #print(g1.get_edge_data(start, boh)['peso'])
    print("Path trovato: ", lowestCostSearch(g1, start, goal), "\n")


def prova_informed():
    g1, n1, e1 = get_piano1_pesato()
    start = get_node(101, n1)
    goal = get_node(127, n1)
    print("Path trovato: ", AStarSearch(g1, start, goal), "\n")

    #l'unica modifica a df branch and bround dovrebbe essere quella di togliere nodelabels
    print("Path trovato: ", DF_branch_and_bound(g1, start, goal))

# va modificata pure dpSearch per funzionare con oggetti Stanza
def dpSearch(graph, goal):
    print("DP SEARCH: ")
    frontier = PriorityQueue()
    visited = set()
    cost_to_goal = {}
    goal_id = goal.get_id()  # Assicurati che 'goal' sia un oggetto Stanza

    # Inizializzo a infinito il valore di cost_to_goal per ciascun nodo
    for node in graph.nodes:
        node_id = node.get_id()  # Ottiene l'ID dell'oggetto Stanza
        if node_id != goal_id:
            cost_to_goal[node_id] = float('inf')
    cost_to_goal[goal_id] = 0

    # Aggiungo il nodo di partenza alla coda di priorità
    frontier.put((0, goal_id, [goal_id]))  # Cambiato per gestire ID invece di oggetti

    while not frontier.empty():
        priority, current_id, path = frontier.get()
        print(priority, path)
        print(current_id)

        if current_id not in visited:
            visited.add(current_id)
            current = get_node(current_id, graph.nodes)  # Recupera l'oggetto Stanza usando l'ID

            for adj in graph.neighbors(current):
                adj_id = adj.get_id()  # Ottiene l'ID del nodo adiacente
                peso = graph.get_edge_data(current, adj)['peso']
                new_cost = cost_to_goal[current_id] + peso
                if new_cost < cost_to_goal[adj_id]:
                    cost_to_goal[adj_id] = new_cost
                    frontier.put((new_cost, adj_id, path + [adj_id]))  # Usa ID nel percorso
    return cost_to_goal

def prova_dp():
    g1, n1, e1 = get_piano1_pesato()
    goal = get_node(127, n1)
    gRev = g1.reverse()
    print(dpSearch(gRev, goal))




if __name__ == '__main__':
    prova1()
    #provalcfs()
    #prova_informed()
    #prova_dp()
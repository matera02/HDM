import networkx as nx
import matplotlib.pyplot as plt
from queue import Queue, PriorityQueue
import sys

def generate_labeled_directed_graph():
    G = nx.DiGraph()

    # Aggiungi nodi con etichette
    nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J']
    node_labels = {node: f' {node}' for node in nodes}
    G.add_nodes_from(nodes)

    # Aggiungi archi con etichette
    edges = [('A', 'B', {'peso': 2}),
             ('A', 'C', {'peso': 3}),
             ('A', 'D', {'peso': 4}),
             ('B', 'E', {'peso': 2}),
             ('B', 'F', {'peso': 3}),
             ('F', 'D', {'peso': 2}),
             ('D', 'H', {'peso': 4}),
             ('H', 'G', {'peso': 3}),
             ('C', 'J', {'peso': 7}),
             ('J', 'G', {'peso': 4})]

    edge_labels = {edge[:2]: f'{edge[0]}-{edge[1]}: {edge[2]["peso"]}' for edge in edges}
    G.add_edges_from(edges)

    return G, node_labels, edge_labels


def draw_labeled_directed_graph(graph, node_labels=None, edge_labels=None, pos=None):
    """
    Draw a labeled directed graph using NetworkX and Matplotlib.

    Parameters:
        - graph: NetworkX graph object
        - node_labels: Dictionary containing node labels (default: None)
        - edge_labels: Dictionary containing edge labels (default: None)
        - pos: Dictionary containing node positions (default: None)
    """
    if pos is None:
        pos = nx.spring_layout(graph)

    # Disegna i nodi con le etichette
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=12, font_weight='bold',
            arrows=True)

    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Labeled Directed Graph")
    plt.show()


def bfs(graph, start, goal):
    print('BFS: ')
    frontier = Queue()
    frontier.put((start, [start]))

    while not frontier.empty():
        current, path = frontier.get()
        if current == goal:
            return path
        for adj in graph.neighbors(current):
            print((adj, path + [adj]))
            frontier.put((adj, path + [adj]))
    return None


def dfs(graph, start, goal):
    print('DFS: ')
    frontier = [(start, [start])]

    while not frontier == []:
        current, path = frontier.pop()
        if current == goal:
            return path
        for adj in graph.neighbors(current):
            print((adj, path + [adj]))
            frontier.append((adj, path + [adj]))
    return None


def IterativeDeepening(graph, start, goal):
    print('Iterative Deepening: ')

    def DepthLimitedSearch(graph, node, goal, bound, path):
        if bound > 0:
            if node == goal:
                return path
            for adj in graph.neighbors(node):
                if adj not in path:
                    new_path = path + [adj]
                    print(new_path)
                    result = DepthLimitedSearch(graph, adj, goal, bound - 1, new_path)
                    if result is not None:
                        return result
        return None

    bound = 0
    while True:
        result = DepthLimitedSearch(graph, start, goal, bound, [start])
        if result is not None:
            return result
        bound += 1


def lowestCostSearch(graph, start, goal):
    print('Lowest Cost Search: ')
    frontier = PriorityQueue()
    visited = set()
    frontier.put((0, (start, [start])))

    while not frontier.empty():
        priority, (current, path) = frontier.get()
        print(priority, path)
        print(current)

        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            if current in graph:
                for adj in graph.neighbors(current):
                    try:
                        # cambiato da peso a weight
                        peso = graph.get_edge_data(current, adj)['weight']
                        frontier.put((priority + peso, (adj, path + [adj])))
                    except KeyError:
                        pass
            else:
                print(f"Il nodo {current} non è presente nel grafo.")
    return None

def prova_uninfomed():
    G, node_labels, edge_labels = generate_labeled_directed_graph()
    node_positions = {'A': (0, 0), 'B': (0, 1), 'C': (-1, 0), 'D': (2, 0),
                      'E': (0.5, 2),
                      'F': (2, 1),
                      'G': (3, 3), 'H': (3, 1), 'J': (-0.5, 2)}
    draw_labeled_directed_graph(G, node_labels=node_labels, edge_labels=edge_labels, pos=node_positions)

    print("Path trovato: ", bfs(G, 'A', 'G'), "\n")

    print("Path trovato: ", dfs(G, 'A', 'G'), "\n")

    print("Path trovato: ", IterativeDeepening(G, 'A', 'G'), "\n")

    print("Path trovato: ", lowestCostSearch(G, 'A', 'G'))

"""
DA QUI IN POI ABBIAMO LE VARIE IMPLEMENTAZIONI CHE SERVONO PER LA RICERCA INFORMATA
"""
def generate_labeled_directed_graph_h():
    G = nx.DiGraph()
    nodes = {'A': 7, 'B': 5, 'C': 9, 'D': 6, 'E': 3, 'F': 5, 'G': 0, 'H': 3, 'J': 4}
    node_labels = {key: f'{key}:{value}' for key, value in nodes.items()}
    G.add_nodes_from(node_labels.keys())

    edges = [('A', 'B', {'peso': 2}),
             ('A', 'C', {'peso': 3}),
             ('A', 'D', {'peso': 4}),
             ('B', 'E', {'peso': 2}),
             ('B', 'F', {'peso': 3}),
             ('F', 'D', {'peso': 2}),
             ('D', 'H', {'peso': 4}),
             ('H', 'G', {'peso': 3}),
             ('C', 'J', {'peso': 7}),
             ('J', 'G', {'peso': 4})]

    edge_labels = {edge[:2]: f'{edge[0]}-{edge[1]}: {edge[2]["peso"]}' for edge in edges}
    G.add_edges_from(edges)

    return G, node_labels, edge_labels


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


def get_heuristic(node, node_labels):
    label = node_labels.get(node)
    if label:
        return int(label.split(':')[1])  # Estrai il valore numerico dopo il carattere ':'
    else:
        return 0  # Ritorna 0 se l'etichetta del nodo non è presente




def get_f(graph, path, node_labels):
    costo = get_cost(graph, path)
    if costo < 0:
        return None
    return costo + get_heuristic(path[-1], node_labels)


def draw_labeled_directed_graph_h(graph, node_labels=None, edge_labels=None, pos=None):
    if pos is None:
        pos = nx.spring_layout(graph)

    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=12, font_weight='bold')
    nx.draw_networkx_edges(graph, pos, arrows=True)

    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Labeled Directed Graph")
    plt.show()

"""
L'A Star combina la lowest cost first search dove l'ordinamento era dato da cost(path)
La Greedy Best First Search ordinava per heuristic(ultimo_nodo_nel_path)
Quindi qui il criterio di ordinamento è dato da f(p) = cost(p) + heuristic(p)
"""
def AStarSearch(graph, start, goal, node_labels):
    print('AStar Search: ')
    frontier = PriorityQueue()
    visited = set()
    priority = get_f(graph, start, node_labels)

    frontier.put((priority, (start, [start])))

    while not frontier.empty():
        priority, (current, path) = frontier.get()
        current = path[-1]

        print(priority, path)
        print(current)


        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            if current in graph:
                for adj in graph.neighbors(current):
                    try:
                        new_path = path + [adj]
                        priority = get_f(graph, new_path, node_labels)
                        frontier.put((priority, (adj, path + [adj])))
                    except KeyError:
                        pass
            else:
                print(f"Il nodo {current} non è presente nel grafo.")
    return None

def DF_branch_and_bound(graph, start, goal, node_labels):

    def cbsearch(graph, path, goal, bound, frontier, node_labels):
        current = path[-1]
        if get_f(graph, path, node_labels) < bound:
            if current == goal:
                bound = get_cost(graph, path)
            else:
                for adj in graph.neighbors(current):
                    new_bound = get_f(graph, path + [adj], node_labels)
                    frontier.append((new_bound, (adj, path + [adj])))
    frontier = []
    cost = get_f(graph, start, node_labels)
    frontier.append((cost, (start, [start])))
    bound = sys.maxsize  ##inizialmente bound = massimo numero rappresentabile
    while True:
        print(frontier)
        cost, (current, path) = frontier.pop()
        if current == goal:
            return path
        if cost < bound:
            cbsearch(graph, path, goal, bound,frontier, node_labels)



def prova_informed():
    G, node_labels, edge_labels = generate_labeled_directed_graph_h()
    print(node_labels)
    path = ['A', 'B', 'E']
    costo_percorso = get_cost(G, path)
    euristica = get_heuristic(path[-1], node_labels)
    f = get_f(G, path, node_labels)

    print("Costo del percorso:", costo_percorso)
    print("Euristica:", euristica)
    print("F:", f)

    node_positions = {'A': (0, 0), 'B': (0, 1), 'C': (-1, 0), 'D': (2, 0),
                          'E': (0.5, 2),
                          'F': (2, 1),
                          'G': (3, 3), 'H': (3, 1), 'J': (-0.5, 2)}

    draw_labeled_directed_graph_h(G, node_labels=node_labels, edge_labels=edge_labels, pos = node_positions)

    print(AStarSearch(G, 'A', 'G', node_labels))
    print(DF_branch_and_bound(G, 'A', 'G', node_labels))
    print("finito")

if __name__ == '__main__':
    prova_informed()
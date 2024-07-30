import matplotlib.pyplot as plt
import networkx as nx

# 1° PIANO
def get_piano1():
    # Creo il grafo
    G = nx.Graph()

    # Definisco i nodi
    nodes = [
        "Entry", "Reception Room New Patient", "Reception Room Hygiene Patient", 
        "Business Office", "Consult", "Operatory 1", "Operatory 2", "Operatory 3", 
        "Operatory 4", "Operatory 5", "Operatory 6", "Operatory 7", "Operatory 8", 
        "Operatory 9", "Operatory 10", "Darkroom", "Lab", "X-ray", "Central Tray Prep",
        "Elev. Equip", "Storage", "Women's", "Men's", "Stairs", "Nit. 2", "Nit. 1", "Store 1", "Store 2", "Break Room"
    ]

    # Li aggiungo al grafo
    for node in nodes:
        G.add_node(node)

    # Definisco gli archi
    edges = [
        ("Entry", "Reception Room New Patient"),
        ("Entry", "Reception Room Hygiene Patient"),
        ("Reception Room New Patient", "Business Office"),
        ("Reception Room Hygiene Patient", "Business Office"),
        ("Business Office", "Consult"),
        ("Consult", "Operatory 1"),
        ("Consult", "Operatory 2"),
        ("Consult", "Operatory 3"),
        ("Operatory 1", "Operatory 2"),
        ("Operatory 2", "Operatory 3"),
        ("Operatory 3", "Operatory 4"),
        ("Operatory 4", "Operatory 5"),
        ("Operatory 5", "Operatory 6"),
        ("Operatory 6", "Operatory 7"),
        ("Operatory 7", "Operatory 8"),
        ("Operatory 8", "Operatory 9"),
        ("Operatory 9", "Operatory 10"),
        ("Operatory 10", "Central Tray Prep"),
        ("Darkroom", "Lab"),
        ("Lab", "X-ray"),
        ("X-ray", "Central Tray Prep"),
        ("Central Tray Prep", "Elev. Equip"),
        ("Elev. Equip", "Storage"),
        ("Storage", "Stairs"),
        ("Stairs", "Women's"),
        ("Stairs", "Men's"),
        ("Storage", "Break Room"),
        ("Break Room", "Nit. 1"),
        ("Nit. 1", "Nit. 2"),
        ("Nit. 1", "Store 1"),
        ("Nit. 2", "Store 2"),
        ("Darkroom", "X-ray"),
        ("Operatory 4", "Storage"),
    ]

    # Aggiungo gli archi al grafo
    G.add_edges_from(edges)

    # Posiziono manualmente le posizioni dei nodi del grafo
    pos = {
        "Entry": (5, 10),
        "Reception Room New Patient": (4, 9),
        "Reception Room Hygiene Patient": (6, 9),
        "Business Office": (5, 8),
        "Consult": (5, 7),
        "Operatory 1": (4, 7),
        "Operatory 2": (4, 6),
        "Operatory 3": (4, 5),
        "Operatory 4": (3, 5),
        "Operatory 5": (3, 4),
        "Operatory 6": (3, 3),
        "Operatory 7": (3, 2),
        "Operatory 8": (3, 1),
        "Operatory 9": (4, 1),
        "Operatory 10": (5, 1),
        "Darkroom": (6, 1),
        "Lab": (6, 2),
        "X-ray": (6, 3),
        "Central Tray Prep": (5, 3),
        "Elev. Equip": (5, 4),
        "Storage": (5, 5),
        "Women's": (6, 5),
        "Men's": (6, 6),
        "Stairs": (6, 7),
        "Nit. 2": (7, 7),
        "Nit. 1": (7, 6),
        "Store 1": (8, 6),
        "Store 2": (7, 5),
        "Break Room": (6, 8),
    }


    # Diretto
    # Converto il grafo di prima in diretto
    G_directed = nx.DiGraph()

    # Aggiungo i nodi al grafo diretto
    for node in nodes:
        G_directed.add_node(node)

    # Aggiungo gli archi al grafo diretto
    G_directed.add_edges_from(edges)

    # Plot del grafo diretto
    plt.figure(figsize=(20, 20))
    nx.draw(G_directed, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", arrows=True)

    plt.title("1° piano")
    plt.show()

def get_piano2():
    # 2° PIANO
    G2 = nx.DiGraph()

    # Definisco i nodi del secondo piano
    nodes2 = [
        "Entry", "Stairs", "Patient Room 1", "Patient Room 2", "Patient Room 3", "Patient Room 4", 
        "Patient Room 5", "Patient Room 6", "Patient Room 7", "Patient Room 8", "Patient Room 9", 
        "Patient Room 10", "Patient Room 11", "Patient Room 12", "Patient Room 13", "Patient Room 14", 
        "Single Patient Room 1", "Single Patient Room 2", "Single Patient Room 3", 
        "Toilet Room 1", "Toilet Room 2", "Toilet Room 3", "Shower Room 1", "Shower Room 2", 
        "Guest Room", "Mech Room", "Office 1", "Office 2", "Counseling", "Intake", "Medical Storage", 
        "Records", "Employee Room", "Study Library", "Group Patient Room", "Activity Room", "Storage Room"
    ]

    # Aggiungo i nodi al grafo
    for node in nodes2:
        G2.add_node(node)

    # Definisco gli archi del secondo piano
    edges2 = [
        ("Patient Room 1", "Patient Room 2"), ("Patient Room 2", "Patient Room 3"), ("Patient Room 3", "Patient Room 4"), 
        ("Patient Room 4", "Patient Room 5"), ("Patient Room 5", "Patient Room 6"), ("Patient Room 6", "Patient Room 7"), 
        ("Patient Room 7", "Patient Room 8"), ("Patient Room 8", "Patient Room 9"), ("Patient Room 9", "Patient Room 10"), 
        ("Patient Room 10", "Patient Room 11"), ("Patient Room 11", "Patient Room 12"), ("Patient Room 12", "Patient Room 13"), 
        ("Patient Room 13", "Patient Room 14"),
    
        ("Single Patient Room 1", "Single Patient Room 2"), ("Single Patient Room 2", "Single Patient Room 3"),
    
        ("Toilet Room 1", "Toilet Room 2"), ("Toilet Room 2", "Toilet Room 3"),
        ("Shower Room 1", "Shower Room 2"),
    
        ("Guest Room", "Toilet Room 1"), ("Mech Room", "Shower Room 2"), 
        ("Office 1", "Office 2"), ("Office 2", "Employee Room"),
        ("Counseling", "Intake"), ("Medical Storage", "Records"), 
        ("Study Library", "Group Patient Room"), ("Group Patient Room", "Activity Room"), 
        ("Activity Room", "Storage Room"),
    
        ("Entry", "Patient Room 1"), ("Entry", "Patient Room 5"), ("Entry", "Guest Room"),
        ("Entry", "Office 1"), ("Entry", "Study Library"),
        ("Storage Room", "Stairs")
    ]

    # Aggiungo gli archi al grafo
    G2.add_edges_from(edges2)

    # Posizionamento manuale dei nodi del grafo
    pos2 = {
        "Entry": (5, 11), "Stairs": (8, 0), 
        "Patient Room 1": (0, 10), "Patient Room 2": (1, 10), "Patient Room 3": (2, 10), "Patient Room 4": (3, 10), 
        "Patient Room 5": (4, 10), "Patient Room 6": (5, 10), "Patient Room 7": (6, 10), "Patient Room 8": (7, 10), 
        "Patient Room 9": (8, 10), "Patient Room 10": (9, 10), "Patient Room 11": (10, 10), "Patient Room 12": (11, 10), 
        "Patient Room 13": (12, 10), "Patient Room 14": (13, 10),
    
        "Single Patient Room 1": (0, 9), "Single Patient Room 2": (1, 9), "Single Patient Room 3": (2, 9), 
    
        "Toilet Room 1": (4, 9), "Toilet Room 2": (5, 9), "Toilet Room 3": (6, 9),
        "Shower Room 1": (4, 8), "Shower Room 2": (5, 8), 
    
        "Guest Room": (2, 8), "Mech Room": (6, 8),
    
        "Office 1": (8, 9), "Office 2": (9, 9), 
        "Counseling": (10, 9), "Intake": (11, 9), 
        "Medical Storage": (9, 8), "Records": (10, 8), "Employee Room": (9, 7),
    
        "Study Library": (12, 9), "Group Patient Room": (13, 9), 
        "Activity Room": (14, 9), "Storage Room": (15, 9)
    }

    # Plot del grafo
    plt.figure(figsize=(20, 20))
    nx.draw(G2, pos2, with_labels=True, node_size=500, node_color="skyblue", font_size=10, font_weight="bold", edge_color="gray", arrows=True)

    plt.title("2° Piano")
    plt.show()

def get_piano3():
    # 3° PIANO

    # Definisco tutti i nodi e le connessioni
    nodes3 = [
        "Supply Closet", "Minor Procedures", "Exam 1", "Exam 2", "Exam 3", "Drug/Supply", "Doctor Office 1",
        "Waiting Area", "Business Office", "Lab 1", "Lab 2", "Nurses Station", "Break Room",
        "Staff Room", "Doctor Office 2", "Multi-Purpose", "X-ray Suite", "Control Area", "Entry", "Stairs"
    ]

    edges3 = [
        ("Entry", "Waiting Area"),
        ("Waiting Area", "Business Office"),
        ("Business Office", "Lab 1"),
        ("Lab 1", "Lab 2"),
        ("Lab 2", "Nurses Station"),
        ("Nurses Station", "Exam 1"),
        ("Exam 1", "Exam 2"),
        ("Exam 2", "Exam 3"),
        ("Exam 3", "Minor Procedures"),
        ("Minor Procedures", "Drug/Supply"),
        ("Drug/Supply", "Doctor Office 1"),
        ("Doctor Office 1", "Staff Room"),
        ("Staff Room", "Break Room"),
        ("Break Room", "Doctor Office 2"),
        ("Doctor Office 2", "Multi-Purpose"),
        ("Multi-Purpose", "X-ray Suite"),
        ("X-ray Suite", "Control Area"),
        ("Control Area", "Stairs"),
        ("Supply Closet", "Exam 1")  
    ]

    pos3 = {
        "Entry": (1, 0),
        "Waiting Area": (2, 0),
        "Business Office": (3, 0),
        "Lab 1": (4, 0.5),
        "Lab 2": (5, 0.5),
        "Nurses Station": (4.5, 0),
        "Exam 1": (6, 0.3),
        "Exam 2": (7, 0.3),
        "Exam 3": (8, 0.3),
        "Minor Procedures": (5, -0.5),
        "Drug/Supply": (4, -0.5),
        "Doctor Office 1": (3, -0.5),
        "Staff Room": (3, -1),
        "Break Room": (4, -1),
        "Doctor Office 2": (5, -1),
        "Multi-Purpose": (6, -1),
        "X-ray Suite": (7, -1),
        "Control Area": (8, -1),
        "Stairs": (9, -1),
        "Supply Closet": (6, 0.7)
    }

    # Creating the directed graph
    G3 = nx.DiGraph()
    G3.add_nodes_from(nodes3)
    G3.add_edges_from(edges3)

    # Plotting the graph with accurate node positions
    plt.figure(figsize=(15, 10))
    nx.draw(G3, pos3, with_labels=True, node_color='skyblue', node_size=500, arrowstyle='-|>', arrowsize=20)
    plt.title("3° Piano")
    plt.show()


if __name__ == '__main__':
    get_piano1()
    get_piano2()
    get_piano3()
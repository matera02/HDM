import networkx as nx
from collections import deque

# Funzione BFS per esplorare un grafo partendo da un nodo specifico
def bfs(grafo, nodo_partenza):
    coda = deque([nodo_partenza])
    visitati = set([nodo_partenza])
    while coda:
        nodo = coda.popleft()
        print("Nodo visitato:", nodo)
        for vicino in grafo.neighbors(nodo):  # Modifica qui per accedere ai vicini usando NetworkX
            if vicino not in visitati:
                coda.append(vicino)
                visitati.add(vicino)

# Funzione per dividere il grafo in isole (sottografi) basate su un criterio specifico
def dividere_in_isole(grafo, criterio):
    isole = []
    nodi_visitati = set()
    for nodo in grafo:
        if nodo not in nodi_visitati:
            isola_corrente = []
            coda = deque([nodo])
            while coda:
                nodo = coda.popleft()
                isola_corrente.append(nodo)
                nodi_visitati.add(nodo)
                for vicino in grafo.neighbors(nodo):  # Modifica qui per accedere ai vicini usando NetworkX
                    if criterio(grafo, nodo, vicino) and vicino not in nodi_visitati:
                        coda.append(vicino)
            isole.append(isola_corrente)
    return isole

# Funzione per ottenere sottografi corrispondenti a ciascuna isola
def sottografi_da_isole(grafo, isole):
    sottografi = []
    for isola in isole:
        sottografo = grafo.subgraph(isola)
        sottografi.append(sottografo)
    return sottografi

# Creazione del grafo da NetworkX con attributo del piano
grafo1 = nx.Graph()
grafo1.add_nodes_from([('A', {'piano': 1}), ('B', {'piano': 1}), ('C', {'piano': 2}), ('D', {'piano': 2}), ('E', {'piano': 2}), ('F', {'piano': 2})])
grafo1.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('C', 'E'), ('E', 'F')])

# Nuovo criterio per dividere il grafo in isole basate sul piano dei nodi
def criterio_piano(grafo, nodo, vicino):
    return grafo.nodes[nodo]['piano'] == grafo.nodes[vicino]['piano']

# Eseguire la funzione per dividere il grafo in isole basate sul nuovo criterio
isole_piano = dividere_in_isole(grafo1, criterio_piano)
print("Isole basate sul piano dei nodi:", isole_piano)

# Estrai i sottografi corrispondenti a ciascuna isola
sottografi = sottografi_da_isole(grafo1, isole_piano)

# Esegui la BFS su ciascun sottografo
for sottografo in sottografi:
    print("BFS su sottografo:", list(sottografo.nodes))
    bfs(sottografo, list(sottografo.nodes)[0])  # Qui passo il sottografo e faccio la ricerca per ciascun sottografo


# in questo caso cerco di riadattare quello che ho fatto in island per Stanza
import networkx as nx
from collections import deque
import src.graph.hospital as hp

def bfs(graph, start_node):
    queue = deque([start_node])
    visited = set([start_node.get_id()])
    while queue:
        current_node = queue.popleft()
        print("Nodo visitato:", current_node)
        for neighbor in graph.neighbors(current_node):
            if neighbor.get_id() not in visited:
                queue.append(neighbor)
                visited.add(neighbor.get_id())

# Funzione per dividere il grafo in isole (sottografi) basate sul piano
# questo metodo permette di vedere se sono presenti anche altre isole all'interno di un piano
# i nodi non raggiungibili vengono presi come isole, posso sfruttarlo per costruire meglio il grafo
# e poi per la suddivisione in piani
def dividere_in_isole(grafo, criterio):
    isole = []
    nodi_visitati = set()
    for nodo in grafo.nodes:
        if nodo not in nodi_visitati:
            isola_corrente = []
            coda = deque([nodo])
            while coda:
                nodo_corrente = coda.popleft()
                if nodo_corrente not in nodi_visitati:
                    nodi_visitati.add(nodo_corrente)
                    isola_corrente.append(nodo_corrente)
                    for vicino in grafo.neighbors(nodo_corrente):
                        if criterio(nodo_corrente, vicino) and vicino not in nodi_visitati:
                            coda.append(vicino)
            isole.append(isola_corrente)
    return isole

# Criterio basato sul piano dei nodi, direttamente da oggetti Stanza
def criterio_piano(nodo, vicino):
    return nodo.piano == vicino.piano



def provaIsole():
    g1, n1, e1 = hp.get_piano1()
    g2, n2, e2 = hp.get_piano2()
    g3, n3, e3 = hp.get_piano3()
    
    g = nx.compose(g1, g2)
    g = nx.compose(g, g3)

    #provo a collegare le Scale del primo piano all'ingresso del secondo
    # e controllo se la suddivisione per piano funziona
    a = hp.get_node(124, g.nodes())
    b = hp.get_node(201, g.nodes())
    g.add_edge(a, b)
    print(g.has_edge(a, b))
    #provo a collegare le scale del secondo piano con l'ingresso del terzo
    c = hp.get_node(202, g.nodes())
    d = hp.get_node(301, g.nodes())
    g.add_edge(c, d)
    print(g.has_edge(c, d))

    isole_piano = dividere_in_isole(g, criterio_piano)
    print("Isole basate sul piano dei nodi: \n")
    for isola in isole_piano:
        print(isola)


if __name__ == '__main__':
    provaIsole()
from graph import bfs
import src.graph.island as isl
import networkx as nx


G = nx.DiGraph()
G.add_nodes_from([('A', {'piano': 1}), ('B', {'piano': 1}), ('C', {'piano': 2}), ('D', {'piano': 2}), ('E', {'piano': 2}), ('F', {'piano': 2})])
G.add_edges_from([('A', 'B'), ('B', 'A'), ('A', 'C'), ('B', 'C'), ('C', 'D'), ('C', 'E'), ('E', 'F')])

# Eseguire la funzione per dividere il grafo in isole basate sul nuovo criterio
isole_piano = isl.dividere_in_isole(G, isl.criterio_piano)
print("Isole basate sul piano dei nodi:", isole_piano)

# Estrai i sottografi corrispondenti a ciascuna isola
sottografi = isl.sottografi_da_isole(G, isole_piano)

print(bfs(sottografi[0], 'A', 'B'))
print(bfs(sottografi[1], 'C', 'F'))
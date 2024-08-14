from src.graph.hospital import Hospital
from src.util.utility import Utility as util
import networkx as nx

FILENAME = 'src/graph/data/hospital.pl'


# grafo bidirezionale dell'ospedale
class BiHospital:
    _instance = None

    def __init__(self, di_graph=Hospital.get_hospital()):
        if BiHospital._instance is not None:
            raise Exception("È una classe singleton!")
        else:
            BiHospital._instance = self
            self.__bi_graph = self.__get_bi_graph(di_graph=di_graph)

    def __get_bi_graph(self, di_graph):
        return nx.compose(di_graph, di_graph.reverse())
    
    @staticmethod
    def get_bi_hospital():
        if BiHospital._instance is None:
            BiHospital()
        return BiHospital._instance.__bi_graph
    

if __name__ == '__main__':
    bi_hospital = BiHospital.get_bi_hospital()
    util.plot_and_interact_by_floor(bi_hospital, filename=FILENAME)
    


        
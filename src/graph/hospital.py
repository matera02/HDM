from src.util.utility import Utility as util
from pyswip import Prolog
import networkx as nx
FILENAME = 'src/graph/data/hospital.pl'

# classe singleton
class Hospital:
    _instance = None

    class Room:
        def __init__(self, number, name, x, y, floor):
            self.number = number
            self.name = name
            self.x = x
            self.y = y
            self.floor = floor

        def __str__(self):
            return f'Room {self.number} ({self.name}) on floor {self.floor} at position ({self.x}, {self.y})'
    
        def __repr__(self):
            return (f"Room(number={self.number}, name='{self.name}', x={self.x}, "
                    f"y={self.y}, floor={self.floor})")
    

    def __init__(self, filename=FILENAME):
        if Hospital._instance is not None:
            raise Exception("È una classe singleton!")
        else:
            Hospital._instance = self
            self.filename = filename
            self.__hospital = self.__build_graph_from_prolog()

    def __build_graph_from_prolog(self):
        prolog = Prolog()
        prolog.consult(self.filename)
        G = nx.DiGraph()
        rooms = list(prolog.query("room(X)"))
        for room in rooms:
            number = room['X']
            details = next(prolog.query(f"name({number}, T), x({number}, X), y({number}, Y), floor({number}, P)"))
            room_obj = self.Room(number, details['T'], float(details['X']), float(details['Y']), int(details['P']))
            G.add_node(number, room=room_obj)
        
        connections = list(prolog.query("connection(A, B, Weight)"))
        for conn in connections:
            G.add_edge(conn['A'], conn['B'], weight=float(conn['Weight']))
        return G

    @staticmethod
    def get_hospital(filename=FILENAME):
        if Hospital._instance is None:
            Hospital(filename)
        return Hospital._instance.__hospital



if __name__ == '__main__':
    hospital = Hospital.get_hospital()
    util.plot_and_interact_by_floor(hospital, filename=FILENAME)
class Stanza:
    # se si tratta di ricerca non informata metto euristica a 0
    def __init__(self, numero, tipo, x, y, piano, heuristic=0):
        self.numero = numero
        self.tipo = tipo
        self.x = x
        self.y = y
        self.piano = piano
        self.heuristic = heuristic

    def get_pos(self):
        return (self.x, self.y)
    
    def __str__(self):
        return f'{self.numero}'
    
    def __repr__(self):
        return f'{self.numero}'

    def get_id(self):
        return self.numero  

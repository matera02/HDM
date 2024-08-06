class Stanza:
    # HO RIMOSSO L'EURISTICA, NON HA SENSO CHE STIA QUI DENTRO
    # FAI HASHMAP NODO -> EURISTICA, ha più senso per
    # ciò che devi fare dopo

    # se si tratta di ricerca non informata metto euristica a 0
    def __init__(self, numero, tipo, x, y, piano):
        self.numero = numero
        self.tipo = tipo
        self.x = x
        self.y = y
        self.piano = piano

    def get_pos(self):
        return (self.x, self.y)
    
    def __str__(self):
        return f'{self.numero}'
    
    def __repr__(self):
        return f'{self.numero}'

    def get_id(self):
        return self.numero  

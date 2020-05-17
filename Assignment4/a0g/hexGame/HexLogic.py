'''
Author: Martijn Swenne
        Amin Moradi
        Bartek Piaskowski
Date: May 10, 2020.
Board class.
Board data:
  1=Red, -1=Blue, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
class Board():

    # list of all 6 directions on the Hex board, as (x,y) offsets
    __directions = [(1,1),(1,0),(0,-1),(-1,-1),(-1,0),(0,1)]

    def __init__(self, n):
        "Set up initial board configuration."
        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)"""
        moves = []  # stores the legal moves.
        # Get all the squares with pieces of the given color.
        for x in range(self.n):
            for y in range(self.n):
                if self[x][y]==0:
                    moves.append((x,y))
        return moves

    def get_start_border(self, color):
        if color == 1:
            return [(0,i) for i in range(self.n)]
        else:
            return [(i,0) for i in range(self.n)]

    def border(self, color, move):
        (nx, ny) = move
        return (color == 1 and nx == self.n-1) or (color == -1 and ny == self.n-1)

    def has_legal_moves(self, color):
        moves = self.get_legal_moves(color)
        if len(moves) != 0:
            return True
        return False

    def execute_move(self, move, color):
        (x,y) = move
        self[x][y] = color
        
    def get_neighbors(self, coordinates):
        (cx,cy) = coordinates
        neighbors = []
        if cx-1>=0:                    neighbors.append((cx-1,cy))
        if cx+1<self.n:                neighbors.append((cx+1,cy))
        if cx-1>=0 and cy+1<=self.n-1: neighbors.append((cx-1,cy+1))
        if cx+1<self.n and cy-1>=0:    neighbors.append((cx+1,cy-1))
        if cy+1<self.n:                neighbors.append((cx,cy+1))
        if cy-1>=0:                    neighbors.append((cx,cy-1))
        return neighbors

    # TODO: Is this still needed? Idk what it does
    @staticmethod
    def _increment_move(move, direction, n):
        # print(move)
        """ Generator expression for incrementing moves """
        move = list(map(sum, zip(move, direction)))
        #move = (move[0]+direction[0], move[1]+direction[1])
        while all(map(lambda x: 0 <= x < n, move)): 
        #while 0<=move[0] and move[0]<n and 0<=move[1] and move[1]<n:
            yield move
            move=list(map(sum,zip(move,direction)))
            #move = (move[0]+direction[0],move[1]+direction[1])


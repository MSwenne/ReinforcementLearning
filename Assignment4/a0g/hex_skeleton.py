import numpy as np

class HexBoard:
    BLUE = 1
    RED = -1
    EMPTY = 0
    def __init__(self, board_size):
        self.board = {}
        self.size = board_size
        self.game_over = False
        for x in range(board_size):
            for y in range (board_size):
                self.board[x,y] = HexBoard.EMPTY

    def HexGameToBoard(self, board):
        for x in range(self.size):
            for y in range(self.size):
                self.board[(x,y)] = board[x][y]
    def set_board(self, board, game_over=False):
        self.board = board
        self.game_over = game_over
    def is_game_over(self):
        return self.game_over
    def is_empty(self, coordinates):
        return self.board[coordinates] == HexBoard.EMPTY
    def is_color(self, coordinates, color):
        return self.board[coordinates] == color
    def get_color(self, coordinates):
        if coordinates == (-1,-1):
            return HexBoard.EMPTY
        return self.board[coordinates]
    def place(self, coordinates, color):
        if not self.game_over and self.board[coordinates] == HexBoard.EMPTY:
            self.board[coordinates] = color
            if self.check_win(HexBoard.RED) or self.check_win(HexBoard.BLUE):
                self.game_over = True
    def clear(self, coordinates):
        if not (self.board[coordinates] == HexBoard.EMPTY):
            self.board[coordinates] = HexBoard.EMPTY
            if not (self.check_win(HexBoard.RED) or self.check_win(HexBoard.BLUE)):
                self.game_over = False
    def get_opposite_color(self, current_color):
        if current_color == HexBoard.BLUE:
            return HexBoard.RED
        return HexBoard.BLUE
    def get_neighbors(self, coordinates):
        (cx,cy) = coordinates
        neighbors = []
        if cx-1>=0:     neighbors.append((cx-1,cy))
        if cx+1<self.size: neighbors.append((cx+1,cy))
        if cx-1>=0        and cy+1<=self.size-1: neighbors.append((cx-1,cy+1))
        if cx+1<self.size    and cy-1>=0: neighbors.append((cx+1,cy-1))
        if cy+1<self.size: neighbors.append((cx,cy+1))
        if cy-1>=0:     neighbors.append((cx,cy-1))
        return neighbors
    def border(self, color, move):
        (nx, ny) = move
        return (color == HexBoard.BLUE and nx == self.size-1) or (color == HexBoard.RED and ny == self.size-1)
    def get_start_border(self, color):
        if color == HexBoard.BLUE:
            return [(0,i) for i in range(self.size)]
        else:
            return [(i,0) for i in range(self.size)]
    def traverse(self, color, move, visited):
        if not self.is_color(move, color) or (move in visited and visited[move]): return False
        if self.border(color, move): return True
        visited[move] = True
        for n in self.get_neighbors(move):
            if self.traverse(color, n, visited): return True
        return False
    def check_win(self, color):
        for i in range(self.size):
            if color == HexBoard.BLUE: move = (0,i)
            else: move = (i,0)
            if self.traverse(color, move, {}):
                return True
        return False
    def print(self):
        print("     ",end="")
        for y in range(self.size):
                print(chr(y+ord('a')),"",end="")
        print("")
        print(" -----------------------")
        for y in range(self.size):
                print(y, "|",end="")
                for z in range(y):
                        print(" ", end="")
                for x in range(self.size):
                        piece = self.board[x,y]
                        if piece == HexBoard.BLUE: print("b ",end="")
                        elif piece == HexBoard.RED: print("r ",end="")
                        else:
                                if x==self.size:
                                        print("-",end="")
                                else:
                                        print("- ",end="")
                print("|")
        print("     -----------------------")

    # Added
    # Places empty on coordinates
    def unplace(self, coordinates):
        if not (self.board[coordinates] == HexBoard.EMPTY):
            self.board[coordinates] = HexBoard.EMPTY
            self.game_over = False

    # Returns all empty coordinates
    def getMoveList(self):
        # Initialise an empty list of empty coordinates
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                # If the coordinates are empty...
                if(self.is_empty((x,y))):
                    # ... append the coordinates to the list
                    moves.append((x,y))
        # Return the list of empty coordinates
        return moves

    # Added
    # Checks if two board states are equal (overwrites ==)
    def __eq__(self, other): 
        return len([(i,j) 
                for i in range(self.size)
                for j in range(self.size) 
                if self.board[(i,j)] == other.board[(i,j)]]) == self.size * self.size

    # Gotten from HexGame, needed for makeMove of Alpha0General
    def getCanonicalForm(self, player):
        # return state if player==1, else return -state if player==-1
        board = [None]*self.size
        for i in range(self.size):
            board[i] = [0]*self.size
            for j in range(self.size):
                board[i][j] = self.board[(i,j)]
        if player == 1:
            return board
        else:
            return np.fliplr(np.rot90(-1*board, axes=(1, 0)))

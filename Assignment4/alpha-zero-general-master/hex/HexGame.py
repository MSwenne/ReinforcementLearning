from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .HexLogic import Board
import numpy as np
import heapq

class HexGame(Game):
    square_content = {
        -1: "R",
        +0: "-",
        +1: "B"
    }

    @staticmethod
    def getSquarePiece(piece):
        return HexGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n

    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        for x, y in legalMoves:
            valids[self.n*x+y]=1
        return np.array(valids)

    def getGameEnded(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        for i in range(self.n):
            move = (0,i) if player == 1 else (i,0)
            if self.dijkstra(b, move, player) == 0:
                return True
        return False

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s


    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        max_score = -np.inf
        for i in range(self.n):
            move = (0,i) if player == 1 else (i,0)
            # dijkstra returns minimal number of moves to win
            # turn into maximizing score by counting size minus moves to win
            score = self.n - self.dijkstra(b, move, player)
            if score > max_score:
                max_score = score
        return max_score

    # Performs the dijkstra algorithm on the current board with the current player
    # Returns the shortest path from border to border, or inf if there is no path
    def dijkstra(self, board, root, player):
        # Initialise Q, dist, visited and result
        Q = []
        dist = {}
        visited = []
        result = []
        # For all coordinates on the board...
        for x in range(board.size):
            for y in range(board.size):
                # ...if it is not filled by the enemy...
                if not (board[x][y] == -player):
                    # ...and if it is in the root...
                    if (x,y) in root:
                        # ...set the initial distance to 1 if empty or 0 if filled
                        length = 0 if board[x][y] == player else 1
                    # ...and if it is not in the root...
                    else:
                        # ...set the initial distance to inf
                        length = np.inf
                    # Add the length to the distance dict
                    dist[x,y] = length
                    # Heappush the coordinates with the corresponding length
                    heapq.heappush(Q, (length,(x,y)))
        # While the queue Q is not empty...
        while not (len(Q) == 0): 
            # ...get the top coordinates u of Q
            u = heapq.heappop(Q)
            # If these coordinates u are already visited, continue
            if u in visited:
                continue
            # Add the coordinates u to visited
            visited.append(u)
            # For all neighbors v of the coordinates u...
            for v in board.get_neighbors(u[1]):
                # ...if it is not filled by the enemy...
                if not (board[v] == -player):
                    # ...get the length between u and v
                    length = self.dijkstraLength(board, u[1], v, player)
                    # Add this length to the distance of u...
                    alt = dist[u[1]] + length
                    # ...and if this is larger than the distance of v...
                    if alt < dist[v]:
                        # If v is on the border...
                        if board.border(player, v):
                            # If v is empty, add 1 to length
                            if board.is_empty(v):
                                alt = alt + 1
                            # ...heappush v to the results
                            heapq.heappush(result, (alt,v))
                        # ...store the new distance of v
                        dist[v] = alt
                        # Heappush the new distance of v
                        heapq.heappush(Q,(alt,v))
        # After checking all nodes, if there is a path between borders...
        if len(result) != 0:
            # ...return the shortest path
            return (heapq.heappop(result)[0])/2
        # If there is no path, return inf
        else:
            return np.inf

    # Returns the length between two coordinates
    def dijkstraLength(self, board, coord1, coord2, player):
        if board[coord1] == player:
            if board[coord2] == player:
                # If both coordinates have the same player, distance is 0
                return 0
        elif board[coord2] == 0:
            # If both coordinates are empty, distance = 2
            return 2
        # If exactly one coordinate is empty, distance is 1
        return 1

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("     ",end="")
        for y in range(n):
            print(chr(y+ord('a')),"",end="")
        print("")
        print(" -----------------------")
        for y in range(n):
            print(y, "|",end="") # print the row #
            for _ in range(y):
                print(" ", end="")
            for x in range(n):
                piece = board[y][x]
                print(HexGame.square_content[piece], end=" ")
            print("|")
        print("     -----------------------")
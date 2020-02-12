import numpy as np
from hex_skeleton import HexBoard
import heapq
depth = 3

def main():
    print("Hex game: how big is the board?")
    size = input()
    while(size == '' or not(2 < int(size) and int(size) < 10 )):
        print("Invalid size!")
        size = input()
    size = int(size)
    print("(r)ed vs. (b)lue")
    print("blue goes from left to right, red goes from top to bottom.")
    print("which color will you be? (red=0, blue=1)")
    color = input()
    while(color == '' or (int(color) != 0 and int(color) != 1)):
        print("Invalid color!")
        color = input()
    color = int(color)
    board = HexBoard(size)
    player = HexBoard.BLUE if color else HexBoard.RED
    bot = HexBoard.RED if color else HexBoard.BLUE

    while(not board.is_game_over() and not board.fullBoard()):
        print("make a move...")
        valid = False
        while(not valid):
            x, y = get_coordinates()
            if(not (0 <= x and x < size and 0 <= y and y < size)):
                print("Invalid coordinates!")
            else:
                if(not makeMove(board, player, (x,y))):
                    print("Place already taken! ")
                else:
                    valid = True
        if(not board.is_game_over() and not board.fullBoard()):
            print("enemy's turn:")
            # makeRandomMove(board, bot)
            makeAlphaBetaMove(board, bot)
        board.print()
    if(board.check_win(player)):
        print("You win!")
    elif(board.check_win(bot)):
        print("You lose!")
    else:
        print("It's a draw!")

def get_coordinates():
    print("x = ",end="")
    x = input()
    while x == '':
        print("invalid x-coordinate!")
        print("x = ",end="")
        x = input()
    print("y = ",end="")
    y = input()
    while y == '':
        print("invalid y-coordinate!")
        print("y = ",end="")
        y = input()
    return int(x), int(y)

def getMoveList(board, color):
    moves = []
    for x in range(board.size):
        for y in range(board.size):
            if(board.is_empty((x,y))):
                moves.append((x,y))
    return moves

def makeMove(board, color, coordinates):
    if(board.is_empty(coordinates)):
        board.place(coordinates, color)
        return True
    print("Cannot place, not empty!")
    return False

def makeRandomMove(board, color):
    x = np.random.randint(0,board.size)
    y = np.random.randint(0,board.size)
    while(not makeMove(board, color, (x,y))):
        x = np.random.randint(0,board.size)
        y = np.random.randint(0,board.size)
    return True

def makeAlphaBetaMove(board, color):
    enemy = board.get_opposite_color(color)
    best_value = -np.inf
    for move in getMoveList(board, color):
        makeMove(board, color, move)
        value = alpha_beta(board, depth, -np.inf, np.inf, enemy, False)
        board.print()
        print(value)
        unMakeMove(board, move)
        if(value > best_value):
            best_move = move
            best_value = value
    makeMove(board, color, best_move)

def unMakeMove(board, coordinates):
    if(not board.is_empty(coordinates)):
        board.clear(coordinates)
        return True
    else:
        print("Cannot undo, place is empty!")
        return False

def alpha_beta(board, depth, alpha, beta, color, maximize):
    # board.print()
    enemy = board.get_opposite_color(color)
    if depth == 0:
        return dijkstra(board,board.get_start_border(enemy),enemy)
    if board.is_game_over():
        return 0
    if maximize:
        value = -np.inf
        for move in getMoveList(board, color):
            makeMove(board, color, move)
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False))
            unMakeMove(board, move)
            alpha = max(alpha, value)
            if (alpha >= beta):
                break
    else:
        value = np.inf
        for move in getMoveList(board, color):
            makeMove(board, color, move)
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, True))
            unMakeMove(board, move)
            beta = min(beta, value)
            if (alpha >= beta):
                break
    return value

def dijkstra(board, root, color):
    Q = []
    dist = {}
    visited = []
    result = []
    for x in range(board.size):
        for y in range(board.size):
            if not (board.get_color((x,y)) == board.get_opposite_color(color)):
                if (x,y) in root:
                    if board.get_color((x,y)) == color: 
                        length = 0
                    else:
                        length = 1
                else:
                    length = np.inf
                dist[x,y] = length
                heapq.heappush(Q, (length,(x,y)))
    while not (len(Q) == 0): 
        u = heapq.heappop(Q)
        if u in visited:
            continue
        visited.append(u)
        for v in board.get_neighbors(u[1]):
            if not (board.get_color(v) == board.get_opposite_color(color)):
                length = dijkstra_Length(board, u[1], v, color)
                alt = dist[u[1]] + length
                if alt < dist[v]:
                    if board.border(color, v):
                        if board.is_empty(v):
                            alt = alt + 1
                        heapq.heappush(result, (alt,v))
                    dist[v] = alt
                    heapq.heappush(Q,(alt,v))
    return (heapq.heappop(result)[0])/2

def dijkstra_Length(board, coord1, coord2, color):
    if board.get_color(coord1) == color:
        if board.get_color(coord2) == color:
            return 0
    elif board.get_color(coord2) == HexBoard.EMPTY:
            return 2
    return 1

main()
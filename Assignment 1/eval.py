from hex_skeleton import HexBoard
import numpy as np
import random
import heapq

def play_game():
    board, turn = init()
    color = [HexBoard.RED, HexBoard.BLUE]
    while(not board.is_game_over()):
        if turn:
            print("make a move...")
            makeMove(board, color[turn])
        else:
            print("enemy's turn:")
            makeAlphaBetaMove(board, color[turn], 3)
        board.print()
        turn = int(not turn)
    if board.check_win(color[0]):
        print("RED wins!")
    else:
        print("BLUE wins!")

def init():
    print("Hex game: how big is the board? (minimal 2x2 and maximal 10x10)")
    size = validate("size", 2, 10)
    print("(r)ed vs. (b)lue")
    print("blue goes from left to right, red goes from top to bottom.")
    print("which color will you be? (red=0, blue=1)")
    turn = validate("color", -1, 2)
    board = HexBoard(size)
    return board, turn

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

def makeMove(board, color):
    valid = False
    while(not valid):
        x, y = get_coordinates()
        if(not (0 <= x and x < board.size and 0 <= y and y < board.size)):
            print("Invalid coordinates!")
        else:
            if(board.is_empty((x,y))):
                board.place((x,y), color)
                return True
            else:
                print("Place already taken! ")

def getMoveList(board, color):
    moves = []
    for x in range(board.size):
        for y in range(board.size):
            if(board.is_empty((x,y))):
                moves.append((x,y))
    return moves

def validate(val, lower, upper):
    print(val+" = ", end="")
    res = input()
    while(res == '' or not(lower < int(res) and int(res) < upper)):
        print("Invalid input!")
        res = input()
    res = int(res)
    return res

def makeAlphaBetaMove(board, color, depth):
    enemy = board.get_opposite_color(color)
    best_value = np.inf
    best_move = 0
    for move in getMoveList(board, color):
        board.place(move, color)
        value = alpha_beta(board, depth, -np.inf, np.inf, enemy, True)
        board.unplace(move)
        if(value < best_value):
            best_move = move
            best_value = value
    if best_move == 0:
        best_move = move
    board.place(best_move, color)

    
def alpha_beta(board, depth, alpha, beta, color, maximize):
    enemy = board.get_opposite_color(color)
    val1 = dijkstra(board,board.get_start_border(enemy),enemy)
    val2 = dijkstra(board,board.get_start_border(color),color)
    if not all([val1, val2, depth]):
        if maximize:
            return val1 - val2
        else:
            return val2 - val1
    if maximize:
        value = -np.inf
        for move in getMoveList(board, color):
            board.place(move, color)
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False))
            board.unplace(move)
            alpha = max(alpha, value)
            if (alpha >= beta):
                break
    else:
        value = np.inf
        for move in getMoveList(board, color):
            board.place(move, color)
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, True))
            board.unplace(move)
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
                    length = 0 if board.get_color((x,y)) == color else 1
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
    if len(result) != 0:
        return (heapq.heappop(result)[0])/2
    else:
        return np.inf

def dijkstra_Length(board, coord1, coord2, color):
    if board.get_color(coord1) == color:
        if board.get_color(coord2) == color:
            return 0
    elif board.get_color(coord2) == HexBoard.EMPTY:
            return 2
    return 1

play_game()
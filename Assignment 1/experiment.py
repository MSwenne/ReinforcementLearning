from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
import numpy as np
import random
import heapq

def getMoveList(board, color):
    moves = []
    for x in range(board.size):
        for y in range(board.size):
            if(board.is_empty((x,y))):
                moves.append((x,y))
    return moves

def makeAlphaBetaMove(board, color, depth, heuristic):
    enemy = board.get_opposite_color(color)
    best_value = np.inf
    best_move = 0
    for move in getMoveList(board, color):
        board.place(move, color)
        if dijkstra(board,board.get_start_border(color),color) == 0:
            return
        value = alpha_beta(board, depth, -np.inf, np.inf, enemy, True, heuristic)
        board.unplace(move)
        if(value <= best_value):
            best_move = move
            best_value = value
    board.place(best_move, color)

    
def alpha_beta(board, depth, alpha, beta, color, maximize, heuristic):
    enemy = board.get_opposite_color(color)
    val1 = dijkstra(board,board.get_start_border(enemy),enemy)
    val2 = dijkstra(board,board.get_start_border(color),color)
    if not all([val1, val2, depth]):
        if heuristic:
            if maximize:
                return val1 - val2
            else:
                return val2 - val1
        else:
            if board.check_win(color) == 0:
                return 0
            elif board.check_win(enemy) == 0:
                return 1
            else:
                return random.random()
    if maximize:
        value = -np.inf
        for move in getMoveList(board, color):
            board.place(move, color)
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False, heuristic))
            board.unplace(move)
            alpha = max(alpha, value)
            if (alpha >= beta):
                break
    else:
        value = np.inf
        for move in getMoveList(board, color):
            board.place(move, color)
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, True, heuristic))
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

def test_true_skill():
    bots = 3
    rounds = 5
    size = 3
    print("board size: ", size)
    color = [HexBoard.RED, HexBoard.BLUE]
    depths = [[3, 3], [3, 4], [3, 4]]
    heuristics = [[False, True], [False, True], [True, True]]
    turn = 0
    r1 = Rating()
    r2 = Rating()
    r3 = Rating()
    print(r1, r2, r3)
    for round in range(rounds):
        print("round:", round+1)
        color[0], color[1] = color[1], color[0]
        for i in range(bots):
            board = HexBoard(size)
            depth = depths[i]
            heuristic = heuristics[i]
            while(not board.is_game_over()):
                makeAlphaBetaMove(board, color[turn], depth[turn], heuristic[turn])
                turn = int(not turn)
            board.print()
            print("bots:", i, "\t",color,"\t", end='')
            if board.check_win(color[0]) == (color[0] == HexBoard.RED):
                print("RED wins! ", )
            else:
                print("BLUE wins! ", )
            if board.check_win(color[0]):
                if i == 0:
                    r1, r2 = rate_1vs1(r1, r2)
                if i == 1:
                    r1, r3 = rate_1vs1(r1, r3)
                if i == 2:
                    r2, r3 = rate_1vs1(r2, r3)
            # else:
            #     if i == 0:
            #         r1, r2 = rate_1vs1(r2, r1)
            #     if i == 1:
            #         r1, r3 = rate_1vs1(r3, r1)
            #     if i == 2:
            #         r2, r3 = rate_1vs1(r3, r2)
            print(r1, r2, r3)
            del board

test_true_skill()
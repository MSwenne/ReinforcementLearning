from hex_skeleton import HexBoard
import numpy as np
import random
import heapq
import time

table_random = []
table_dijkstra = []
MAX_TIME = 0.1

def getMoveList(board, color):
    moves = []
    for x in range(board.size):
        for y in range(board.size):
            if(board.is_empty((x,y))):
                moves.append((x,y))
    return moves

def makeAlphaBetaMove(board, color, depth, heuristic, TT):
    if depth == -1:
        depth = 1
        curr = time.time()
        while time.time() - curr < MAX_TIME and depth < board.size*board.size:
            best_move = pre_alpha_beta(board, depth, color, heuristic, TT)
            if best_move == (-1, -1):
                return
            depth += 1
    else:
        best_move = pre_alpha_beta(board, depth, color, heuristic, TT)
        if best_move == (-1, -1):
            return
    board.place(best_move, color)

def pre_alpha_beta(board, depth, color, heuristic, TT):
    enemy = board.get_opposite_color(color)
    best_value = np.inf
    best_move = 0
    for move in getMoveList(board, color):
        board.place(move, color)
        if dijkstra(board,board.get_start_border(color),color) == 0:
            return (-1, -1)
        value = alpha_beta(board, depth, -np.inf, np.inf, enemy, True, heuristic, TT)
        # print(value, " : ", end="")
        board.unplace(move)
        if(value <= best_value):
            best_move = move
            best_value = value
    print(best_value)
    return best_move

    
def alpha_beta(board, depth, alpha, beta, color, maximize, heuristic, TT):
    if TT:
        table = table_dijkstra if heuristic else table_random
        for b_v in table:
            b = b_v[0]
            if b == board:
                return b_v[1]
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
            if val2 == 0:
                return 0 if maximize else 1
            elif val1 == 0:
                return 1 if maximize else 0
            else:
                return random.random()
    if maximize:
        value = -np.inf
        for move in getMoveList(board, color):
            board.place(move, color)
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False, heuristic, TT))
            board.unplace(move)
            alpha = max(alpha, value)
            if (alpha >= beta):
                break
    else:
        value = np.inf
        for move in getMoveList(board, color):
            board.place(move, color)
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, True, heuristic, TT))
            board.unplace(move)
            beta = min(beta, value)
            if (alpha >= beta):
                break
    # if TT:
    #     add = True
    #     for b_v in table:
    #         b = b_v[0]
    #         if b == board:
    #             add = False
    #     if add:
    #         table.append((board,value))
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
    bots = 1
    rounds = 1
    size = 4
    print("board size: ", size)
    color = [HexBoard.RED, HexBoard.BLUE]
    bots_print = ["[1, 2]", "[1, 3]", "[2, 3]"]
    color_print = ["RED", "BLUE"]
    depths = [[3, -1], [3, -1], [-1, -1]]
    heuristics = [[False, True], [False, True], [True, True]]
    TT = [[False, True], [False, True], [True, True]]
    turn = 1
    for round in range(rounds):
        print("round:", round+1)
        color[0], color[1] = color[1], color[0]
        color_print[0], color_print[1] = color_print[1], color_print[0]
        for i in range(bots):
            board = HexBoard(size)
            depth = depths[i]
            heuristic = heuristics[i]
            print("bots: ", bots_print[i], "\t", color_print[0], ":", color_print[1])
            while(not board.is_game_over()):
                makeAlphaBetaMove(board, color[turn], depth[turn], heuristic[turn], TT)
                turn = int(not turn)
                board.print()
            board.print()
            if board.check_win(color[0]) == (color[0] == HexBoard.RED):
                print("RED wins! ", )
            else:
                print("BLUE wins! ", )
            del board
test_true_skill()
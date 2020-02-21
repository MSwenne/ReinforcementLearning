from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
import numpy as np
import random
import heapq
import time

table_random = []
table_dijkstra = []
MAX_TIME = 0.01

def main():
    play = input("Play game, test or trueskill? (1, 2 or 3)")
    if play == 1:
        ab = input("vs random bot or alpha beta bot (1 or 2)")
        play_game(ab)
    if play == 2:
        test()
    if play == 3:
        test_true_skill()

def validate(val, lower, upper):
    res = input(val+" = ", end="")
    while(res == '' or not(lower < int(res) and int(res) < upper)):
        print("Invalid input!")
        res = input()
    res = int(res)
    return res

def init():
    print("Hex game: how big is the board? (minimal 2x2 and maximal 10x10)")
    size = validate("size", 2, 10)
    print("(r)ed vs. (b)lue")
    print("blue goes from left to right, red goes from top to bottom.")
    print("which color will you be? (red=0, blue=1)")
    color = validate("color", -1, 2)
    board = HexBoard(size)
    player = HexBoard.BLUE if color else HexBoard.RED
    bot = HexBoard.RED if color else HexBoard.BLUE
    return board, player, bot

def play_game(ab):
    board, player, bot = init()

    while(not board.is_game_over()):
        print("make a move...")
        makeMove(board, player)
        board.print()
        if(board.is_game_over()):
            print("You win!")
        else:
            print("enemy's turn:")
            if ab == 1:
                makeRandomMove(board, bot)
            if ab == 2:
                makeAlphaBetaMove(board, bot, 3, True)
        board.print()
    if board.check_win(player):
        print("You win!")
    else:
        print("You lose!")

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

def makeMove(board, color):
    valid = False
    while(not valid):
        x, y = get_coordinates()
        if(not (0 <= x and x < board.size and 0 <= y and y < board.size)):
            print("Invalid coordinates!")
        else:
            if(board.is_empty(coordinates)):
                board.place(coordinates, color)
                print("Place already taken! ")
            else:
                valid = True
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

def makeAlphaBetaMove(board, color, depth, heuristic):
    enemy = board.get_opposite_color(color)
    best_value = np.inf
    best_move = 0
    for move in getMoveList(board, color):
        makeMove(board, color, move)
        if depth == -1:
            depth = 1
            curr = time.time()
            while time.time() - curr < MAX_TIME and depth < board.size*board.size:
                value = alpha_beta(board, depth, -np.inf, np.inf, enemy, True, heuristic)
                depth += 1
            # print("Depth: ", depth)
            depth = -1
        else:
            # print("Depth: ", depth)
            value = alpha_beta(board, depth, -np.inf, np.inf, enemy, True, heuristic)
        unMakeMove(board, move)
        if(value < best_value):
            best_move = move
            best_value = value
    if best_move == 0:
        best_move = move
    makeMove(board, color, best_move)

def unMakeMove(board, coordinates):
    if(not board.is_empty(coordinates)):
        board.clear(coordinates)
        return True
    else:
        print("Cannot undo, place is empty!")
        return False

def alpha_beta(board, depth, alpha, beta, color, maximize, heuristic):
    table = table_dijkstra if heuristic else table_random
    for b_v in table:
        b = b_v[0]
        if b == board:
            return b_v[1]
    enemy = board.get_opposite_color(color)
    val1 = dijkstra(board,board.get_start_border(enemy),enemy)
    val2 = dijkstra(board,board.get_start_border(color),color)
    if not all([val1, val2, depth]):
        return evaluate(val1, val2, heuristic, maximize)
    if maximize:
        value = -np.inf
        for move in getMoveList(board, color):
            makeMove(board, color, move)
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False, heuristic))
            unMakeMove(board, move)
            alpha = max(alpha, value)
            if (alpha >= beta):
                break
    else:
        value = np.inf
        for move in getMoveList(board, color):
            makeMove(board, color, move)
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, True, heuristic))
            unMakeMove(board, move)
            beta = min(beta, value)
            if (alpha >= beta):
                break
            
    add = True
    for b_v in table:
        b = b_v[0]
        if b == board:
            add = False
        
    if add:
        table.append((board,value))
    return value

def evaluate(val1, val2, heuristic, maximize):
    if heuristic:
        if maximize:
            return val1 - val2
        else:
            return val2 - val1
    else:
        if val1 == 0:
            return 0
        elif val2 == 0:
            return 1
        else:
            return random.random()

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
    size = 4
    print("board size: ", size)
    color = [HexBoard.RED, HexBoard.BLUE]
    depths = [[3, 3], [3, 4], [3, 4]]
    heuristics = [[False, True], [False, True], [True, True]]
    turn = 1
    r1 = Rating()
    r2 = Rating()
    r3 = Rating()
    print(r1, r2, r3)
    for round in range(rounds):
        print("round:", round+1)
        for i in range(bots):
            board = HexBoard(size)
            depth = depths[i]
            heuristic = heuristics[i]
            while(not board.is_game_over()):
                makeAlphaBetaMove(board, color[turn], depth[turn], heuristic[turn])
                turn = int(not turn)
                # board.print()
            if board.check_win(color[0]):
                if i == 0:
                    r1, r2 = rate_1vs1(r1, r2)
                if i == 1:
                    r1, r3 = rate_1vs1(r1, r3)
                if i == 2:
                    r2, r3 = rate_1vs1(r2, r3)
            else:
                if i == 0:
                    r1, r2 = rate_1vs1(r2, r1)
                if i == 1:
                    r1, r3 = rate_1vs1(r3, r1)
                if i == 2:
                    r2, r3 = rate_1vs1(r3, r2)
            board.print()
            print(r1, r2, r3)
            del board
    for b_v in table_dijkstra:
        print(b_v[1], end =" : ")
    print()

    for b_v in table_random:
        print(b_v[1], end =" : ")
    print()

def test():
    size = 4
    color = [HexBoard.RED, HexBoard.BLUE]
    depth = [-1, 3]
    heuristic = [True, True]
    win = [0, 0]
    turn = 1
    for _ in range(5):
        # print("round:", round+1)
        board = HexBoard(size)
        while(not board.is_game_over()):
            makeAlphaBetaMove(board, color[turn], depth[turn], heuristic[turn])
            turn = int(not turn)
            # board.print()
        if board.check_win(color[0]):
            print("RED")
            win[0] += 1
        if board.check_win(color[1]):
            print("BLUE")
            win[1] += 1
        board.print()
        del board
    print(win)



# test()
test_true_skill()
# main()
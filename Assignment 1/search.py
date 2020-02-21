from hex_skeleton import HexBoard
import numpy as np
import random

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
            makeAlphaBetaMove(board, bot, 3, True)
        board.print()
    if board.check_win(player):
        print("You win!")
    else:
        print("You lose!")

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
                print("Place already taken! ")
            else:
                valid = True
        return True
    print("Cannot place, not empty!")
    return False

def getMoveList(board, color):
    moves = []
    for x in range(board.size):
        for y in range(board.size):
            if(board.is_empty((x,y))):
                moves.append((x,y))
    return moves

def validate(val, lower, upper):
    res = input(val+" = ", end="")
    while(res == '' or not(lower < int(res) and int(res) < upper)):
        print("Invalid input!")
        res = input()
    res = int(res)
    return res

def makeAlphaBetaMove(board, color, depth, heuristic):
    enemy = board.get_opposite_color(color)
    best_value = np.inf
    best_move = 0
    for move in getMoveList(board, color):
        board.place(color, move)
        value = alpha_beta(board, depth, -np.inf, np.inf, enemy, True)
        board.unplace(move)
        if(value < best_value):
            best_move = move
            best_value = value
    if best_move == 0:
        best_move = move
    board.place(color, best_move)

    
def alpha_beta(board, depth, alpha, beta, color, maximize):
    enemy = board.get_opposite_color(color)
    if depth == 0:
        if  board.check_win(color) == 0:
            return 1
        elif  board.check_win(enemy) == 0:
            return 0
        else:
            return random.random()
    if maximize:
        value = -np.inf
        for move in getMoveList(board, color):
            board.place(color, move)
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False))
            board.unplace(move)
            alpha = max(alpha, value)
            if (alpha >= beta):
                break
    else:
        value = np.inf
        for move in getMoveList(board, color):
            board.place(color, move)
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, True))
            board.unplace(move)
            beta = min(beta, value)
            if (alpha >= beta):
                break
    return value
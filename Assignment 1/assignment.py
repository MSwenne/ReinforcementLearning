import numpy as np
from hex_skeleton import HexBoard
np.seed(42)

def main():
    print("Hex game: how big is the board?")
    size = int(input())
    while(not (2 < size and size < 10 )):
        print("Invalid size!")
        size = int(input())
    print("(r)ed vs. (b)lue")
    print("blue goes from left to right, red goes from top to bottom.")
    print("which color will you be? (red=0, blue=1)")
    color = int(input())
    while(color != 0 and color != 1):
        print("Invalid color!")
        color = int(input())
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
                    print("Place already taken! ", end="")
                else:
                    valid = True
        if(not board.is_game_over() and not board.fullBoard()):
            print("enemy's turn:")
            makeRandomMove(board, bot)
        board.print()
    if(board.check_win(player)):
        print("You win!")
    elif(board.check_win(bot)):
        print("You lose!")
    else:
        print("It's a draw!")

def get_coordinates():
    print("x = ",end="")
    x = int(input())
    print("y = ",end="")
    y = int(input())
    return x, y

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
    return False

def makeRandomMove(board, color):
    x = np.random.randint(0,board.size)
    y = np.random.randint(0,board.size)
    while(not makeMove(board, color, (x,y))):
        x = np.random.randint(0,board.size)
        y = np.random.randint(0,board.size)
    return True

def unMakeMove(board, coordinates):
    if(not board.is_empty(coordinates)):
        board.place(coordinates, HexBoard.EMPTY)
        return True
    else:
        print("Cannot undo, place is empty!")
        return False

def alpha_beta(board, depth, alpha, beta, color, maximize):
    enemy = board.get_opposite_color(color)
    if (depth == 0 or board.is_game_over()):
        if(board.check_win(color)):
            return 2 # WIN
        elif(board.check_win(enemy)):
            return 0 # LOSS
        else:
            return 1 # DRAW
    if maximize:
        value = -np.inf
        for move in getMoveList(board, color):
            makeMove(board, color, move)
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False))
            unMakeMove(board, move)
            alpha = max(alpha, value)
            if (alpha >= beta):
                break
        return value
    else:
        value = np.inf
        for move in getMoveList(board, color):
            makeMove(board, color, move)
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, False))
            unMakeMove(board, move)
            beta = beta(beta, value)
            if (alpha >= beta):
                break
        return value

main()
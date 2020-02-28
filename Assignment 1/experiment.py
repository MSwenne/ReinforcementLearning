#####################################################################
#                                                                   #
# Reinforcement Learning Assingment 1: HEX                          #
#   Part 1: Search                                                  #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 28 Februari 2020                                     #
# All rights reserved                                               #
#                                                                   #
#####################################################################

from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
import numpy as np
import random
import heapq

# Returns all empty coordinates
def getMoveList(board, color):
    # Initialise an empty list of empty coordinates
    moves = []
    for x in range(board.size):
        for y in range(board.size):
            # If the coordinates are empty...
            if(board.is_empty((x,y))):
                # ... append the coordinates to the list
                moves.append((x,y))
    # Return the list of empty coordinates
    return moves

# Makes a move that uses alpha-beta search and a random eval
def makeAlphaBetaMove(board, color, depth, heuristic):
    # Initialise the enemy, best_value and best_move
    enemy = board.get_opposite_color(color)
    best_value = np.inf
    best_move = 0
    # For every possible move on the board...
    for move in getMoveList(board, color):
        # ...play this move...
        board.place(move, color)
        # If this made a path from border to border, return
        if dijkstra(board,board.get_start_border(color),color) == 0:
            return
        # ...do alpha-beta search...
        value = alpha_beta(board, depth, -np.inf, np.inf, enemy, True, heuristic)
        # ...and undo the move again
        board.unplace(move)
        # If the value of the alpha-beta search is better that your current best_value...
        if(value < best_value):
            # ...update the best_value and the best_move
            best_move = move
            best_value = value
    # If there is no best_move (it is still 0), just take the last viewed move
    if best_move == 0:
        best_move = move
    # Place the best_move
    board.place(best_move, color)

# Does n-depth alpha_beta search on a board with a random or Dijkstra eval
def alpha_beta(board, depth, alpha, beta, color, maximize, heuristic):
    enemy = board.get_opposite_color(color)
    val1 = dijkstra(board,board.get_start_border(enemy),enemy)
    val2 = dijkstra(board,board.get_start_border(color),color)
    # If depth is 0 or the game ended, eval the board
    if not all([val1, val2, depth]):
        # If you want dijkstra eval, do this
        if heuristic:
            if maximize:
                return val1 - val2
            else:
                return val2 - val1
        # If you want random eval, do this
        else:
            if val2 == 0:
                return 0 if maximize else 1
            elif val1 == 0:
                return 1 if maximize else 0
            else:
                return random.random()
    # If this is the maximizing player...
    if maximize:
        # ...initialise the value...
        value = -np.inf
        # ...and for each possible move...
        for move in getMoveList(board, color):
            # ...play the move...
            board.place(move, color)
            # ...recursively call alpha_beta search on the enemy...
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False, heuristic))
            # ...undo the move...
            board.unplace(move)
            # ...and update the alpha value
            alpha = max(alpha, value)
            # If alpha is greater or equal to beta, break
            if (alpha >= beta):
                break
    # If this is the minimizing player...
    else:
        # ...initialise the value...
        value = np.inf
        # ...and for each possible move...
        for move in getMoveList(board, color):
            # ...play the move...
            board.place(move, color)
            # ...recursively call alpha_beta search on the enemy...
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, True, heuristic))
            # ...undo the move...
            board.unplace(move)
            # ...and update the alpha value
            beta = min(beta, value)
            # If alpha is greater or equal to beta, break
            if (alpha >= beta):
                break
    # Return the eval value
    return value

# Performs the dijkstra algorithm on the current board with the current color
# Returns the shortest path from border to border, or inf if there is no path
def dijkstra(board, root, color):
    # Initialise Q, dist, visited and result
    Q = []
    dist = {}
    visited = []
    result = []
    # For all coordinates on the board...
    for x in range(board.size):
        for y in range(board.size):
            # ...if it is not filled by the enemy...
            if not (board.get_color((x,y)) == board.get_opposite_color(color)):
                # ...and if it is in the root...
                if (x,y) in root:
                    # ...set the initial distance to 1 if empty or 0 if filled
                    length = 0 if board.get_color((x,y)) == color else 1
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
            if not (board.get_color(v) == board.get_opposite_color(color)):
                # ...get the length between u and v
                length = dijkstra_Length(board, u[1], v, color)
                # Add this length to the distance of u...
                alt = dist[u[1]] + length
                # ...and if this is larger than the distance of v...
                if alt < dist[v]:
                    # If v is on the border...
                    if board.border(color, v):
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
    # If there is no path...
    else:
        # ...return inf
        return np.inf

# Returns the length between two coordinates
def dijkstra_Length(board, coord1, coord2, color):
    if board.get_color(coord1) == color:
        if board.get_color(coord2) == color:
            # If both coordinates have the same color, distance is 0
            return 0
    elif board.get_color(coord2) == HexBoard.EMPTY:
        # If both coordinates are empty, distance = 2
        return 2
    # If exactly one coordinate is empty, distance is 1
    return 1

# Runs several rounds where 3 bots plays against eachother
def test_true_skill():
    # Initialise the number of bots (max 3), rounds and board size
    bots = 3
    rounds = 10
    size = 3
    print("board size: ", size)
    # Initialise color, printing params, depths and heuristics
    color = [HexBoard.BLUE, HexBoard.RED]
    bots_print = ["[1, 2]", "[1, 3]", "[2, 3]"]
    color_print = ["BLUE", "RED"]
    depths = [[3, 3], [3, 4], [3, 4]]
    heuristics = [[False, True], [False, True], [True, True]]
    # Initialise ratings
    r1 = Rating()
    r2 = Rating()
    r3 = Rating()
    print(r1, r2, r3)
    # For each round:
    for round in range(rounds):
        print("round:", round+1)
        # For each bot-pair:
        for i in range(bots):
            # Switch starting player each round
            turn = 0 if round % 2 == 0 else 1
            # Setup board, depth and heuristic
            board = HexBoard(size)
            depth = depths[i]
            heuristic = heuristics[i]
            print("bots: " ,bots_print[i],"\t", color_print[0], ":", color_print[1],"\t", color_print[turn], "starts")
            # While the game is not over
            while(not board.is_game_over()):
                # Make a ove using alpha_beta search and a random or Dijkstra eval
                makeAlphaBetaMove(board, color[turn], depth[turn], heuristic[turn])
                # Switch turns
                turn = int(not turn)
            # Print board and winner after game ends
            board.print()
            if board.check_win(color[1]):
                print("RED wins! ", )
            else:
                print("BLUE wins! ", )
            # Update ratings accordingly
            if board.check_win(color[0]):
                if i == 0:
                    r1, r2 = rate_1vs1(r1, r2)
                if i == 1:
                    r1, r3 = rate_1vs1(r1, r3)
                if i == 2:
                    r2, r3 = rate_1vs1(r2, r3)
            else:
                if i == 0:
                    r2, r1 = rate_1vs1(r2, r1)
                if i == 1:
                    r3, r1 = rate_1vs1(r3, r1)
                if i == 2:
                    r3, r2 = rate_1vs1(r3, r2)
            # Print new ratings and clean up board
            print(r1, r2, r3)
            del board

# Calls test_true_skill
if __name__ == "__main__":
    test_true_skill()
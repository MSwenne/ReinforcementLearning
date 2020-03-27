from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
from AlphaBeta import AlphaBeta
from operator import itemgetter 
from MCTS_Hex import MCTS
from PlayGame import Play
from utils import get_input
import numpy as np
import pandas as pd
MAX_TIME = 0.01
size = 5

def tune():
    def play(p1, p2, round):
        # Initialise the number of rounds, board size, Cp and iterations
        color = [HexBoard.RED, HexBoard.BLUE]
        p1_cp = p1[0]
        p2_cp = p2[0]
        p1_N = p1[1]
        p2_N = p2[1]
        bots = [
            MCTS(Cp=p1_cp, itermax=p1_N, max_time=MAX_TIME), 
            MCTS(Cp=p2_cp, itermax=p2_N, max_time=MAX_TIME)]
        turn = 0 if round % 2 == 0 else 1
        board = HexBoard(size)
        while(not board.is_game_over()):
            board = bots[turn].makeMove(board, color[turn])
            turn = int(not turn)
        
        if board.check_win(HexBoard.RED):
            del board
            return 1, 0
        else:
            del board
            return 0, 1

    CP = np.arange(0.01,1.5, 0.05).tolist()
    N = list(range(100, 10000, 200))

    configs = [(cp, n) for n in N for cp in CP]
    results = []
    champ = configs.pop()
    while len(configs) > 1:
        config = configs.pop()
        counter = 0
        p1_score = 0
        p2_score = 0
        print(' Remaining games: ', len(configs))
        print("Player1: CP=", champ[0], " - N=",champ[1], "Player2: CP=", config[0], " - N=",config[1],)
        
        while counter < 11:
            print('Match: ', counter, ' out of 11')
            print('Player 1 score: ', p1_score)
            print('Player 2 score: ', p2_score)
            counter += 1
            p1, p2 = play(champ, config, counter)
            p1_score += p1
            p2_score += p2

        results.append([champ[0], champ[1], config[0], config[1], p1_score, p2_score])
        if p1_score < p2_score:
            champ = config
    print('Highest score with cp: ', champ[0], ' and N of: ', champ[1])
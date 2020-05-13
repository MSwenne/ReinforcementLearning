#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 4: Self-Play                    #
#   Main function                                                   #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 9 May 2020                                           #
# All rights reserved                                               #
#                                                                   #
#####################################################################


from alpha-zero-general-master.hexGame.tensorflow.NNet import NNetWrapper as NNet
from alpha-zero-general-master.hexGame.HexGame import Game
from trueskill import Rating, quality_1vs1, rate_1vs1
from ParallelAlphaBeta import AlphaBeta
from operator import itemgetter 
from MCTS_Hex import MCTS
from utils import get_input
import numpy as np
import coloredlogs
import tqdm

MAX_TIME = 0.2
CP = np.sqrt(4)
ITERMAX = 10000

def Tournament():
    game = Game(7)
    n1 = NNet(game)
    n1.load_checkpoint('./TODO','TODO')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = MCTS(g, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    bot_MCTS = MCTS(Cp=CP, itermax=ITERMAX, max_time=MAX_TIME)
    bot_AB = AlphaBeta(max_time=MAX_TIME)
    players = [bot_MCTS, bot_AB, n1p]
    ratings = [Rating() for _ in players]
    text = ["MCTS", "AB", "A0G"]
    games = []
    logs = []
    for i in range(len(players)):
        for j in range(i,len(players)):
            games.append((players[i], players[j]))
            logs.append((text[i], text[j]))

    num = int(num / 2)
    oneWon = 0
    twoWon = 0
    draws = 0
    for i, game in enumerate(games):
        for _ in tqdm(range(num), desc=f'{logs[i][0]} - {logs[i][1]}'):
            gameResult = play(game[0], game[1], verbose=False)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1 
            else:
                draws += 1

def play(p1, p2, verbose):
    players = [p1, None, p2]
    curPlayer = 1
    it = 0
    while game.getGameEnded(board, curPlayer) == 0:
        it += 1
        action = players[curPlayer + 1](game.getCanonicalForm(board, curPlayer))
        valids = game.getValidMoves(game.getCanonicalForm(board, curPlayer), 1)
        if valids[action] == 0:
            log.error(f'Action {action} is not valid!')
            log.debug(f'valids = {valids}')
        board, curPlayer = self.game.getNextState(board, curPlayer, action)
    if verbose:
        assert self.display
        print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
        self.display(board)
    return curPlayer * self.game.getGameEnded(board, curPlayer)

if __name__ == "__main__":
    print("Which part of the assignment would you like to see?")
    ans = get_input("(M)CTS Hex, (E)xperiment, (T)une", ['M', 'E', 'T', 'm', 'e', 't'])
    if ans == 'T' or ans == 't':
        Tournament()
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

# Trueskill import
from trueskill import Rating, quality_1vs1, rate_1vs1

# A0G imports
from hexGame.keras.NNet import NNetWrapper as NNet
from hexGame.HexPlayers import HumanHexPlayer
from hexGame.HexLogic import Board
from hexGame.HexGame import HexGame
from MCTS import MCTS as A0MCTS
import Arena

# Previous assignment imports
from ParallelAlphaBeta import AlphaBeta
from hex_skeleton import HexBoard
from MCTS_Hex import MCTS

# Utils imports
from utils import get_input
from utils import dotdict

# Other libraries
import matplotlib.pyplot as plt
from operator import itemgetter
import numpy as np
import coloredlogs
import logging
import time
import csv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

MAX_TIME = 0.2
CP = np.sqrt(4)
ITERMAX = 10000

def HumanPlay():
    game = HexGame(7)
    n1 = NNet(game)
    n1.load_checkpoint('./checkpoint','Model1.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = A0MCTS(game, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    hp = HumanHexPlayer(game).play
    arena = Arena.Arena(n1p, hp, game, display=HexGame.display)
    print(arena.playGames(2, verbose=True))

def Tournament():
    verbose = True
    num = int(get_input("How many games?",[str(i) for i in range(1,1000)]))
    game = HexGame(7)
    n1 = NNet(game)
    n1.load_checkpoint('./checkpoint','Model1.pth.tar')
    args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts1 = A0MCTS(game, n1, args1)
    n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))
    n2 = NNet(game)
    n1.load_checkpoint('./checkpoint','Model2.pth.tar')
    args2 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
    mcts2 = A0MCTS(game, n2, args2)
    n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))
    bot_MCTS = MCTS(Cp=CP, itermax=ITERMAX, max_time=MAX_TIME)
    bot_AB = AlphaBeta(max_time=MAX_TIME)
    players = [bot_AB, bot_MCTS, n1p, n2p]
    ratings = [Rating() for _ in players]
#     ratings = []
# #  *********************** INITIAL VALUES *************************
#     ratings.append(Rating(mu=24.30733422259068))
#     ratings.append(Rating(mu=24.632270394539333))
#     ratings.append(Rating(mu=24.769898667348023))
#     ratings.append(Rating(mu=25.466144488107247))

    text = ['AB', 'MCTS', 'A0G1', 'A0G2']
    games = []
    logs = []
    index = []
    for i in range(len(players)):
        for j in range(i+1,len(players)):
            games.append((players[i], players[j]))
            logs.append((text[i], text[j]))
            index.append((i, j))

    oneWon = 0
    twoWon = 0
    draws = 0
    
    fields= [rating.mu for rating in ratings]
    with open(r'./tournaments.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    scores = [fields]
    for j in range(num):
        now = time.time()
        print(f'round {j+1}:')
        for i, game in enumerate(games):
            if j % 2 == 0:
                log.info(f'{logs[i][0]} - {logs[i][1]}')
                gameResult = play(game[0], logs[i][0], game[1], logs[i][1], verbose)
                if gameResult == 1:
                    ratings[index[i][0]], ratings[index[i][1]] = rate_1vs1(ratings[index[i][0]], ratings[index[i][1]])
                    oneWon += 1
                elif gameResult == -1:
                    ratings[index[i][1]], ratings[index[i][0]] = rate_1vs1(ratings[index[i][1]], ratings[index[i][0]])
                    twoWon += 1 
                else:
                    draws += 1
            else:
                log.info(f'{logs[i][1]} - {logs[i][0]}')
                gameResult = play(game[1], logs[i][1], game[0], logs[i][0], verbose)
                if gameResult == -1:
                    ratings[index[i][1]], ratings[index[i][0]] = rate_1vs1(ratings[index[i][1]], ratings[index[i][0]])
                    oneWon += 1
                elif gameResult == 1:
                    ratings[index[i][0]], ratings[index[i][1]] = rate_1vs1(ratings[index[i][0]], ratings[index[i][1]])
                    twoWon += 1
                else:
                    draws += 1
            print(f'{time.time()-now}\t{logs[i][j%2]}: {oneWon}, {logs[i][(j%2)-1]}: {twoWon}, draw: {draws}')
            print(ratings)
            oneWon = 0
            twoWon = 0
            draws = 0
        fields= [rating.mu for rating in ratings]
        sigmas = [rating.sigma for rating in ratings]
        with open(r'./tournaments.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        with open(r'./sigmas.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(sigmas)
        scores.append(fields)
    plt.plot(scores)
    plt.legend(text)
    plt.show()

def play(p1, text1, p2, text2, verbose=False):
    game = HexGame(7)
    board = game.getInitBoard()
    b = HexBoard(7)
    players = [p2, None, p1]
    text = [text2, None, text1]
    curPlayer = 1
    it = 0
    while game.getGameEnded(board, curPlayer) == 0:
        it += 1
        if text[curPlayer + 1][0:2] == 'A0':
            action = players[curPlayer + 1](board)
        else:
            b.HexGameToBoard(board)
            action = players[curPlayer + 1].makeMove(b, curPlayer)
            action = game.n*action[0]+action[1]
        valids = game.getValidMoves(board, 1)
        if not valids[action]:
            log.error(f'Action {action} is not valid!')
            log.debug(f'valids = {valids}')
        board, curPlayer = game.getNextState(board, curPlayer, action)
    return curPlayer * game.getGameEnded(board, curPlayer)

if __name__ == "__main__":
    print("Which part of the assignment would you like to see?")
    ans = get_input("(H)uman vs. A0G or (T)ournament", ['H', 'T', 'h', 't'])
    if ans == 'H' or ans == 'h':
        HumanPlay()
    if ans == 'T' or ans == 't':
        Tournament()
class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

# Added for hex_skeleton
from hex_skeleton import HexBoard
import copy

def get_input(message, valid, ending="\n"):
    print(message, end=ending)
    result = input()
    while result not in valid:
        print("Invalid value!",result,valid)
        print(message, end=ending)
        result = input()
    return result


def copy_board(board):
    new_board = HexBoard(board.size)
    new_board.set_board(board.board.copy(), bool(board.game_over))
    return new_board
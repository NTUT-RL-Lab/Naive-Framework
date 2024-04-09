from enum import Enum
OPS_N = 4


class Ops(Enum):
    NOOP = 0
    FIRE = 1
    UP = 2
    DOWN = 3
    LEFT = 4
    RIGHT = 5
    UPLEFT = 6
    UPRIGHT = 7
    DOWNLEFT = 8
    DOWNRIGHT = 9
    UPFIRE = 10
    DOWNFIRE = 11
    LEFTFIRE = 12
    RIGHTFIRE = 13
    UPLEFTFIRE = 14
    UPRIGHTFIRE = 15
    DOWNLEFTFIRE = 16
    DOWNRIGHTFIRE = 17

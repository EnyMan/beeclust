AMNESIA = -1
EMPTY = 0
BEE_UP = 1
BEE_RIGHT = 2
BEE_DOWN = 3
BEE_LEFT = 4
WALL = 5
HEATER = 6
COOLER = 7

BEE = [
        BEE_UP,
        BEE_RIGHT,
        BEE_DOWN,
        BEE_LEFT,
]

OBSTACLES = [
        WALL,
        HEATER,
        COOLER,
]

TURN = {
        BEE_UP: BEE_DOWN,
        BEE_RIGHT: BEE_LEFT,
        BEE_DOWN: BEE_UP,
        BEE_LEFT: BEE_RIGHT,
}

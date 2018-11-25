import cython
from libcpp.deque cimport deque
from libcpp.set cimport set
from libcpp.pair cimport pair

from cython.view cimport array as carray

cimport numpy as cnp
from libc.math cimport NAN, INFINITY

cdef int AMNESIA = -1
cdef int EMPTY = 0
cdef int BEE_UP = 1
cdef int BEE_RIGHT = 2
cdef int BEE_DOWN = 3
cdef int BEE_LEFT = 4
cdef int WALL = 5
cdef int HEATER = 6
cdef int COOLER = 7

cdef extern from "state.h":
    cdef cppclass State:
        State() except +
        State(int, int, int, int) except +
        int x, y, d, t
        int operator<(const State& rhs) const


def check_type(check, targets, key):
    for target in targets:
        if isinstance(check, target):
            return check

    raise TypeError(f'{key} is invalid type')

cdef State new_state(State a):
    return State(a.x, a.y, a.d + 1, a.t)

cdef State move_up(State a):
    if a.x - 1 >= 0:
        new = new_state(a)
        new.x -= 1
        return new
    else:
        return State(0, 0, 0, -1)

cdef State move_left(State a):
    if a.y - 1 >= 0:
        new = new_state(a)
        new.y -= 1
        return new
    else:
        return State(0, 0, 0, -1)

cdef State move_right(State a, int y_max):
    if a.y + 1 < y_max:
        new = new_state(a)
        new.y += 1
        return new
    else:
        return State(0, 0, 0, -1)

cdef State move_down(State a, int x_max):
    if a.x + 1 < x_max:
        new = new_state(a)
        new.x += 1
        return new
    else:
        return State(0, 0, 0, -1)

cdef State move_up_right(State a, int y_max):
    if a.x - 1 >= 0 and a.y + 1 < y_max:
        new = new_state(a)
        new.x -= 1
        new.y += 1
        return new
    else:
        return State(0, 0, 0, -1)

cdef State move_up_left(State a):
    if a.x - 1 >= 0 and a.y - 1 >= 0:
        new = new_state(a)
        new.x -= 1
        new.y -= 1
        return new
    else:
        return State(0, 0, 0, -1)

cdef State move_down_right(State a, int x_max, int y_max):
    if a.x + 1 < x_max and a.y + 1 < y_max:
        new = new_state(a)
        new.x += 1
        new.y += 1
        return new
    else:
        return State(0, 0, 0, -1)

cdef State move_down_left(State a, int x_max):
    if a.x + 1 < x_max and a.y - 1 >= 0:
        new = new_state(a)
        new.x += 1
        new.y -= 1
        return new
    else:
        return State(0, 0, 0, -1)

@cython.boundscheck(False)
def calculate_heatmap(double[:, :] heatmap_view,  cnp.int64_t[:, :] map_view,
                      int T_heater, int T_cooler, int T_env, double k_temp):
    cdef int row, col, i
    cdef double dist_heater, dist_cooler
    cdef double heating, cooling
    cdef deque[State] queue
    cdef State move, init, start
    cdef set[State] visited
    cdef double[:, :] heaters = carray((heatmap_view.shape[0], heatmap_view.shape[1]),sizeof(double), 'd')
    cdef double[:, :] coolers = carray((heatmap_view.shape[0], heatmap_view.shape[1]),sizeof(double), 'd')

    # finding heaters and coolers
    for row in range(map_view.shape[0]):
        for col in range(map_view.shape[1]):
            if map_view[row, col] == HEATER:
                heaters[row, col] = 0
                queue.push_back(State(row, col, 0, HEATER))
            elif map_view[row, col] == COOLER:
                coolers[row, col] = 0
                queue.push_back(State(row, col, 0, COOLER))
            elif map_view[row, col] == WALL:
                coolers[row, col] = INFINITY
                heaters[row, col] = INFINITY
            else:
                coolers[row, col] = -1
                heaters[row, col] = -1

    while queue.size() > 0:
        start = queue.front()
        queue.pop_front()

        # TODO make DRY
        move = move_up(start)
        if move.t != -1 and map_view[move.x, move.y] != WALL:
            if move.t == HEATER:
                if heaters[move.x, move.y] < 0 or heaters[move.x, move.y] > move.d:
                    heaters[move.x, move.y] = move.d
                    queue.push_back(move)
            if move.t == COOLER:
                if coolers[move.x, move.y] < 0 or coolers[move.x, move.y] > move.d:
                    coolers[move.x, move.y] = move.d
                    queue.push_back(move)

        move = move_left(start)
        if move.t != -1 and map_view[move.x, move.y] != WALL:
            if move.t == HEATER:
                if heaters[move.x, move.y] < 0 or heaters[move.x, move.y] > move.d:
                    heaters[move.x, move.y] = move.d
                    queue.push_back(move)
            if move.t == COOLER:
                if coolers[move.x, move.y] < 0 or coolers[move.x, move.y] > move.d:
                    coolers[move.x, move.y] = move.d
                    queue.push_back(move)

        move = move_right(start, map_view.shape[1])
        if move.t != -1 and map_view[move.x, move.y] != WALL:
            if move.t == HEATER:
                if heaters[move.x, move.y] < 0 or heaters[move.x, move.y] > move.d:
                    heaters[move.x, move.y] = move.d
                    queue.push_back(move)
            if move.t == COOLER:
                if coolers[move.x, move.y] < 0 or coolers[move.x, move.y] > move.d:
                    coolers[move.x, move.y] = move.d
                    queue.push_back(move)

        move = move_down(start, map_view.shape[0])
        if move.t != -1 and map_view[move.x, move.y] != WALL:
            if move.t == HEATER:
                if heaters[move.x, move.y] < 0 or heaters[move.x, move.y] > move.d:
                    heaters[move.x, move.y] = move.d
                    queue.push_back(move)
            if move.t == COOLER:
                if coolers[move.x, move.y] < 0 or coolers[move.x, move.y] > move.d:
                    coolers[move.x, move.y] = move.d
                    queue.push_back(move)

        move = move_up_right(start, map_view.shape[1])
        if move.t != -1 and map_view[move.x, move.y] != WALL:
            if move.t == HEATER:
                if heaters[move.x, move.y] < 0 or heaters[move.x, move.y] > move.d:
                    heaters[move.x, move.y] = move.d
                    queue.push_back(move)
            if move.t == COOLER:
                if coolers[move.x, move.y] < 0 or coolers[move.x, move.y] > move.d:
                    coolers[move.x, move.y] = move.d
                    queue.push_back(move)

        move = move_up_left(start)
        if move.t != -1 and map_view[move.x, move.y] != WALL:
            if move.t == HEATER:
                if heaters[move.x, move.y] < 0 or heaters[move.x, move.y] > move.d:
                    heaters[move.x, move.y] = move.d
                    queue.push_back(move)
            if move.t == COOLER:
                if coolers[move.x, move.y] < 0 or coolers[move.x, move.y] > move.d:
                    coolers[move.x, move.y] = move.d
                    queue.push_back(move)

        move = move_down_right(start, map_view.shape[0], map_view.shape[1])
        if move.t != -1 and map_view[move.x, move.y] != WALL:
            if move.t == HEATER:
                if heaters[move.x, move.y] < 0 or heaters[move.x, move.y] > move.d:
                    heaters[move.x, move.y] = move.d
                    queue.push_back(move)
            if move.t == COOLER:
                if coolers[move.x, move.y] < 0 or coolers[move.x, move.y] > move.d:
                    coolers[move.x, move.y] = move.d
                    queue.push_back(move)

        move = move_down_left(start, map_view.shape[0])
        if move.t != -1 and map_view[move.x, move.y] != WALL:
            if move.t == HEATER:
                if heaters[move.x, move.y] < 0 or heaters[move.x, move.y] > move.d:
                    heaters[move.x, move.y] = move.d
                    queue.push_back(move)
            if move.t == COOLER:
                if coolers[move.x, move.y] < 0 or coolers[move.x, move.y] > move.d:
                    coolers[move.x, move.y] = move.d
                    queue.push_back(move)


    # calculating the temperatures for empty cells
    for row in range(map_view.shape[0]):
        for col in range(map_view.shape[1]):
            if map_view[row, col] == HEATER:
                heatmap_view[row, col] = T_heater
            elif map_view[row, col] == COOLER:
                heatmap_view[row, col] = T_cooler
            elif map_view[row, col] == WALL:
                heatmap_view[row, col] = NAN
            else:
                dist_cooler = coolers[row, col]
                dist_heater = heaters[row, col]

                heating = (1 / dist_heater) * <double>(T_heater - T_env)
                cooling = (1 / dist_cooler) * <double>(T_env - T_cooler)
                heatmap_view[row, col] = \
                    T_env + k_temp * (max(heating, 0) - max(cooling, 0))

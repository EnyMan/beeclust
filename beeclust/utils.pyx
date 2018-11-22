import cython
import numpy as np
from libcpp.pair cimport pair
from libcpp.deque cimport deque
from libcpp.set cimport set

cimport numpy as cnp

cdef int AMNESIA = -1
cdef int EMPTY = 0
cdef int BEE_UP = 1
cdef int BEE_RIGHT = 2
cdef int BEE_DOWN = 3
cdef int BEE_LEFT = 4
cdef int WALL = 5
cdef int HEATER = 6
cdef int COOLER = 7

cdef list BEE = [
        BEE_UP,
        BEE_RIGHT,
        BEE_DOWN,
        BEE_LEFT,
]

cdef list OBSTACLES = [
        WALL,
        HEATER,
        COOLER,
]

def check_type(check, targets, key):
    for target in targets:
        if isinstance(check, target):
            return check

    raise TypeError(f'{key} is invalid type')

# [[x,y], [shape, step]]
ctypedef pair[pair[int, int], pair[pair[int, int],int]] state

cdef state new_state(state a):
    return state(pair[int,int](a.first.first, a.first.second),
                 pair[pair[int, int],int](a.second.first, a.second.second))

cdef state move_up(state a):
    if a.first.first - 1 >= 0:
        new = new_state(a)
        new.first.first -= 1
        return new
    else:
        return state(pair[int, int](0, 0),
                     pair[pair[int, int], int](pair[int, int](0,0), -1))

cdef state move_left(state a):
    if a.first.second - 1 >= 0:
        new = new_state(a)
        new.first.second -= 1
        return new
    else:
        return state(pair[int, int](0, 0),
                     pair[pair[int, int], int](pair[int, int](0,0), -1))

cdef state move_right(state a):
    if a.first.second + 1 < a.second.first.second:
        new = new_state(a)
        new.first.second += 1
        return new
    else:
        return state(pair[int, int](0, 0),
                     pair[pair[int, int], int](pair[int, int](0,0), -1))

cdef state move_down(state a):
    if a.first.first + 1 < a.second.first.first:
        new = new_state(a)
        new.first.first += 1
        return new
    else:
        return state(pair[int, int](0, 0),
                     pair[pair[int, int], int](pair[int, int](0,0), -1))

cdef state move_up_right(state a):
    if a.first.first - 1 >= 0 and a.first.second + 1 < a.second.first.second:
        new = new_state(a)
        new.first.first -= 1
        new.first.second += 1
        return new
    else:
        return state(pair[int, int](0, 0),
                     pair[pair[int, int], int](pair[int, int](0,0), -1))

cdef state move_up_left(state a):
    if a.first.first - 1 >= 0 and a.first.second - 1 >= 0:
        new = new_state(a)
        new.first.first -= 1
        new.first.second -= 1
        return new
    else:
        return state(pair[int, int](0, 0),
                     pair[pair[int, int], int](pair[int, int](0,0), -1))

cdef state move_down_right(state a):
    if a.first.first + 1 < a.second.first.first and a.first.second + 1 < a.second.first.second:
        new = new_state(a)
        new.first.first += 1
        new.first.second += 1
        return new
    else:
        return state(pair[int, int](0, 0),
                     pair[pair[int, int], int](pair[int, int](0,0), -1))

cdef state move_down_left(state a):
    if a.first.first + 1 < a.second.first.first and a.first.second - 1 >= 0:
        new = new_state(a)
        new.first.first += 1
        new.first.second -= 1
        return new
    else:
        return state(pair[int, int](0, 0),
                     pair[pair[int, int], int](pair[int, int](0,0), -1))

@cython.boundscheck(False)
def calculate_heatmap(double[:, :] heatmap_view,  cnp.int64_t[:, :] map_view,
                      int T_heater, int T_cooler, int T_env, float k_temp):
    cdef int row, col, i
    cdef int dist_heater, dist_cooler
    cdef double heating, cooling
    cdef deque[state] queue
    cdef state move, init, start
    cdef set[state] visited

    for row in range(map_view.shape[0]):
        for col in range(map_view.shape[1]):
            if map_view[row, col] == HEATER:
                heatmap_view[row, col] = T_heater
            elif map_view[row, col] == COOLER:
                heatmap_view[row, col] = T_cooler
            elif map_view[row, col] == WALL:
                heatmap_view[row, col] = np.nan
            else:
                dist_cooler = map_view.shape[0] * map_view.shape[1]
                dist_heater = map_view.shape[0] * map_view.shape[1]
                # queue.append(Coordinates(row, col, np_map.shape))

                init = state(pair[int,int](row, col),
                             pair[pair[int, int],int](pair[int, int](map_view.shape[0], map_view.shape[1]), 0))
                queue.clear()
                queue.push_back(init)

                visited.clear()

                while queue.size() > 0:
                    start = queue.front()
                    queue.pop_front()

                    if visited.find(start) == visited.end():
                        visited.insert(start)
                    else:
                        continue
                    if start.second.second > dist_cooler or start.second.second > dist_heater:
                        continue

                    if map_view[start.first.first, start.first.second] == HEATER:
                        if start.second.second < dist_heater:
                            dist_heater = start.second.second
                        continue
                    if map_view[start.first.first, start.first.second] == COOLER:
                        if start.second.second < dist_cooler:
                            dist_cooler = start.second.second
                        continue
                    if map_view[start.first.first, start.first.second] == WALL:
                        continue
                    if start.second.second >= map_view.shape[0] * map_view.shape[1]:
                        break
                    start.second.second += 1

                    move = move_up(start)
                    if move.second.second != -1:
                        queue.push_back(move)

                    move = move_left(start)
                    if move.second.second != -1:
                        queue.push_back(move)

                    move = move_right(start)
                    if move.second.second != -1:
                        queue.push_back(move)

                    move = move_down(start)
                    if move.second.second != -1:
                        queue.push_back(move)

                    move = move_up_right(start)
                    if move.second.second != -1:
                        queue.push_back(move)

                    move = move_up_left(start)
                    if move.second.second != -1:
                        queue.push_back(move)

                    move = move_down_right(start)
                    if move.second.second != -1:
                        queue.push_back(move)

                    move = move_down_left(start)
                    if move.second.second != -1:
                        queue.push_back(move)

                heating = (1 / dist_heater) * (T_heater - T_env)
                cooling = (1 / dist_cooler) * (T_env - T_cooler)
                heatmap_view[row, col] = \
                    T_env + k_temp * (heating - cooling)
                    # T_env + k_temp * (max(heating, 0) - max(cooling, 0))

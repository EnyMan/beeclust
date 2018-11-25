import cython
import numpy as np
from .utils import calculate_heatmap, check_type

cimport numpy
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.deque cimport deque
from cython.view cimport array as carray
from libc.stdlib cimport rand, RAND_MAX


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

TURN = {
    BEE_UP: BEE_DOWN,
    BEE_RIGHT: BEE_LEFT,
    BEE_DOWN: BEE_UP,
    BEE_LEFT: BEE_RIGHT,
}

cdef int _stop_time(pair[int, int] bee, double T_local,
                    int k_stay, int T_ideal, int min_wait):
        cdef int wait_time = <int>max(k_stay / (1 + abs(T_ideal - T_local)), min_wait)
        return -wait_time


class BeeClust:
    def __init__(self, map,
                 p_changedir=0.2,
                 p_wall=0.8,
                 p_meet=0.8,
                 k_temp=0.9,
                 k_stay=50,
                 T_ideal=35,
                 T_heater=40,
                 T_cooler=5,
                 T_env=22,
                 min_wait=2):
        check_type(map, [np.ndarray], 'map')
        self.map = map.astype(np.int64)
        self.p_changedir = check_type(p_changedir, [float, int], 'p_changedir')
        self.p_wall = check_type(p_wall, [float, int], 'p_wall')
        self.p_meet = check_type(p_meet, [float, int], 'p_meet')
        self.k_temp = check_type(k_temp, [float, int], 'k_temp')
        self.k_stay = check_type(k_stay, [float, int], 'k_stay')
        self.T_ideal = check_type(T_ideal, [float, int], 'T_ideal')
        self.T_heater = check_type(T_heater, [float, int], 'T_heater')
        self.T_cooler = check_type(T_cooler, [float, int], 'T_cooler')
        self.T_env = check_type(T_env, [float, int], 'T_env')
        self.min_wait = check_type(min_wait, [float, int], 'min_wait')

        if p_wall > 1. or p_meet > 1. or p_changedir > 1.:
            raise ValueError('maximum probability is 1')

        if p_wall < 0. or p_meet < 0. or p_changedir < 0.:
            raise ValueError('probability cant be negative')

        if k_temp < 0. or k_stay < 0. or min_wait < 0.:
            raise ValueError("coefficients cant be negative")

        if T_heater < T_env:
            raise ValueError('T_heater cant be lower then T_env')

        if T_cooler > T_env:
            raise ValueError('T_cooler cant be higher then T_env')

        if self.map.ndim != 2:
            raise ValueError('invalid shape')
        self.heatmap = np.zeros((map.shape[0], map.shape[1]))
        self.recalculate_heat()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @property
    def bees(self):
        cdef vector[pair[int, int]] bees
        cdef pair[int, int] coords
        cdef int row, col
        cdef numpy.int64_t[:, :] map_view = self.map
        for row in range(map_view.shape[0]):
            for col in range(map_view.shape[1]):
                if map_view[row, col] == EMPTY or map_view[row, col] > BEE_LEFT:
                    continue
                else:
                    coords = pair[int,int](row, col)
                    bees.push_back(coords)

        p_bees = []
        for i in range(bees.size()):
            bee = (bees[i].first, bees[i].second)
            p_bees.append(bee)

        return p_bees

    @cython.boundscheck(False)
    @property
    def swarms(self):
        # cdef set[pair[int, int]] procesed
        cdef deque[pair[int, int]] check
        cdef pair[int, int] bee_origin, bee_n
        cdef vector[pair[int, int]] swarm
        cdef vector[vector[pair[int, int]]] swarms
        cdef int found = 0
        cdef unsigned int i = 0, j = 0, k = 0
        cdef vector[pair[int, int]] bees = self.bees
        cdef numpy.int64_t[:, :] map_view = self.map
        cdef int[:, :] procesed = carray((map_view.shape[0], map_view.shape[1]),sizeof(int), 'i')
        procesed[:,:] = 0
        # for p_bee in self.bees:
        #     bees.push_back(pair[int,int](p_bee[0], p_bee[1]))

        for i in range(bees.size()):
            bee_origin = bees[i]
            if procesed[bee_origin.first, bee_origin.second] == 1:
                continue
            swarm.clear()
            check.clear()
            check.push_back(bee_origin)
            while check.size() > 0:
                bee = check.front()
                check.pop_front()
                procesed[bee.first, bee.second] = 1

                for j in range(swarm.size()):
                    if bee == swarm[j]:
                        found = 1
                        break

                if found == 1:
                    found = 0
                    continue
                else:
                    swarm.push_back(bee)

                bee_n = pair[int,int](bee.first + 1, bee.second)
                if 0 <= bee_n.first < map_view.shape[0] and 0 <= bee_n.second < map_view.shape[1]:
                    if map_view[bee_n.first, bee_n.second] < 0 or 1 <= map_view[bee_n.first, bee_n.second] <= 4:
                        if procesed[bee_n.first, bee_n.second] == 0:
                            check.push_back(bee_n)

                bee_n = pair[int,int](bee.first, bee.second + 1)
                if 0 <= bee_n.first < map_view.shape[0] and 0 <= bee_n.second < map_view.shape[1]:
                    if map_view[bee_n.first, bee_n.second] < 0 or 1 <= map_view[bee_n.first, bee_n.second] <= 4:
                        if procesed[bee_n.first, bee_n.second] == 0:
                            check.push_back(bee_n)

                bee_n = pair[int,int](bee.first - 1, bee.second)
                if 0 <= bee_n.first < map_view.shape[0] and 0 <= bee_n.second < map_view.shape[1]:
                    if map_view[bee_n.first, bee_n.second] < 0 or 1 <= map_view[bee_n.first, bee_n.second] <= 4:
                        if procesed[bee_n.first, bee_n.second] == 0:
                            check.push_back(bee_n)

                bee_n = pair[int,int](bee.first, bee.second - 1)
                if 0 <= bee_n.first < map_view.shape[0] and 0 <= bee_n.second < map_view.shape[1]:
                    if map_view[bee_n.first, bee_n.second] < 0 or 1 <= map_view[bee_n.first, bee_n.second] <= 4:
                        if procesed[bee_n.first, bee_n.second] == 0:
                            check.push_back(bee_n)

            swarms.push_back(swarm)

        i = 0
        j = 0

        p_swarms = []
        for i in range(swarms.size()):
            p_swarm = []
            for j in range(swarms[i].size()):
                p_swarm.append((swarms[i][j].first, swarms[i][j].second))
            p_swarms.append(p_swarm)
        return p_swarms

    @property
    def score(self) -> float:
        temps = 0
        bees = self.bees
        for bee in bees:
            temps += self.heatmap[bee[0], bee[1]]
        return temps / len(bees)

    def _obstacle_hit(self, numpy.int64_t[:, :] map_view,
                      pair[int, int] bee, int stop_time):
        if rand()/RAND_MAX < self.p_wall:
            #T_local = self.heatmap[bee.first, bee.second]
            map_view[bee.first, bee.second] = stop_time
        else:
            map_view[bee.first, bee.second] = TURN[map_view[bee.first, bee.second]]

    def _change_direction(self, pair[int, int] bee, int amnesia):
        moves = [BEE_LEFT, BEE_DOWN, BEE_RIGHT, BEE_UP]
        if amnesia == 0:
            moves.remove(self.map[bee.first, bee.second])
        new_dir = np.random.choice(moves)

        self.map[bee.first, bee.second] = new_dir

    @cython.boundscheck(False)
    def tick(self):
        cdef numpy.int64_t[:, :] map_view = self.map
        cdef double[:, :] heatmap_view = self.heatmap
        cdef vector[pair[int, int]] bees = self.bees
        cdef int moved = 0
        cdef int amnesia = 0
        cdef int stop_time = 0
        cdef double p_meet = self.p_meet
        cdef double p_changedir = self.p_changedir

        for i in range(bees.size()):
            amnesia = 0
            stop_time = _stop_time(bees[i], heatmap_view[bees[i].first, bees[i].second], self.k_stay, self.T_ideal, self.min_wait)
            # BEE WAITED
            if map_view[bees[i].first, bees[i].second] < AMNESIA:
                map_view[bees[i].first, bees[i].second] += 1
                continue

            # BEE CHANGE DIRECTION OR AMNESIA
            if map_view[bees[i].first, bees[i].second] == AMNESIA:
                amnesia = 1
            if rand()/RAND_MAX < p_changedir or amnesia == 1:
                self._change_direction(bees[i], amnesia)
                if amnesia == 1:
                    continue

            # BEE TRIES TO GO UP
            if map_view[bees[i].first, bees[i].second] == BEE_UP:
                if bees[i].first - 1 < 0 or map_view[bees[i].first - 1, bees[i].second] >= 5:
                    self._obstacle_hit(map_view, bees[i], stop_time)
                elif map_view[bees[i].first - 1, bees[i].second] < EMPTY or 1 <= map_view[bees[i].first - 1, bees[i].second] <= 4:
                    if rand()/RAND_MAX < p_meet:
                        map_view[bees[i].first, bees[i].second] = stop_time
                else:
                    moved += 1
                    map_view[bees[i].first - 1, bees[i].second] = map_view[bees[i].first, bees[i].second]
                    map_view[bees[i].first, bees[i].second] = EMPTY

            # BEE TRIES TO GO DOWN
            elif map_view[bees[i].first, bees[i].second] == BEE_DOWN:
                if bees[i].first + 1 >= map_view.shape[0] or \
                        map_view[bees[i].first + 1, bees[i].second] >= 5:
                    self._obstacle_hit(map_view, bees[i], stop_time)
                elif map_view[bees[i].first + 1, bees[i].second] < EMPTY or\
                        1 <= map_view[bees[i].first + 1, bees[i].second] <= 4:
                    if rand()/RAND_MAX < p_meet:
                        map_view[bees[i].first, bees[i].second] = stop_time
                else:
                    moved += 1
                    map_view[bees[i].first + 1, bees[i].second] = map_view[bees[i].first, bees[i].second]
                    map_view[bees[i].first, bees[i].second] = EMPTY

            # BEE TRIES TO GO LEFT
            elif map_view[bees[i].first, bees[i].second] == BEE_LEFT:
                if bees[i].second - 1 < 0 or map_view[bees[i].first, bees[i].second - 1] >= 5:
                    self._obstacle_hit(map_view, bees[i], stop_time)
                elif map_view[bees[i].first, bees[i].second - 1] < EMPTY or\
                        1 <= map_view[bees[i].first, bees[i].second - 1] <= 4:
                    if rand()/RAND_MAX < p_meet:
                        map_view[bees[i].first, bees[i].second] = stop_time
                else:
                    moved += 1
                    map_view[bees[i].first, bees[i].second - 1] = map_view[bees[i].first, bees[i].second]
                    map_view[bees[i].first, bees[i].second] = EMPTY

            # BEE TRIES TO GO RIGHT
            elif map_view[bees[i].first, bees[i].second] == BEE_RIGHT:
                if bees[i].second + 1 >= map_view.shape[1] or \
                        map_view[bees[i].first, bees[i].second + 1] >= 5:
                    self._obstacle_hit(map_view, bees[i], stop_time)
                elif map_view[bees[i].first, bees[i].second + 1] < EMPTY or\
                        1 <= map_view[bees[i].first, bees[i].second + 1] <= 4:
                    if rand()/RAND_MAX < p_meet:
                        map_view[bees[i].first, bees[i].second] = stop_time
                else:
                    moved += 1
                    map_view[bees[i].first, bees[i].second + 1] = map_view[bees[i].first, bees[i].second]
                    map_view[bees[i].first, bees[i].second] = EMPTY
        return moved

    def forget(self) -> None:
        for bee in self.bees:
            self.map[bee[0], bee[1]] = AMNESIA

    def recalculate_heat(self) -> None:
        cdef double[:, :] heatmap_view
        cdef numpy.int64_t[:, :] map_view
        heatmap_view = self.heatmap
        map_view = self.map

        calculate_heatmap(heatmap_view, map_view, self.T_heater,
                          self.T_cooler, self.T_env, self.k_temp)

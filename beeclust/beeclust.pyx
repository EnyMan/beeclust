import cython
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.set cimport set
from libcpp.deque cimport deque

cimport numpy

from .utils import calculate_heatmap, np, check_type

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

class BeeClust:
    def __init__(self, numpy.ndarray[numpy.int64_t, ndim=2] map,
                 float p_changedir=0.2,
                 float p_wall=0.8,
                 float p_meet=0.8,
                 float k_temp=0.9,
                 int k_stay=50,
                 int T_ideal=35,
                 int T_heater=40,
                 int T_cooler=5,
                 int T_env=22,
                 int min_wait=2):
        self.map = check_type(map, [np.ndarray], 'map')
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

        if self.map.ndim > 2:
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

    @property
    def swarms(self):
        cdef set[pair[int, int]] procesed
        cdef deque[pair[int, int]] check
        cdef pair[int, int] bee_origin, bee_l, bee_r, bee_u, bee_d, bee, t
        cdef vector[pair[int, int]] swarm
        cdef vector[vector[pair[int, int]]] swarms
        cdef int found = 0
        cdef unsigned int i = 0, j = 0, k = 0
        cdef vector[pair[int, int]] bees

        for p_bee in self.bees:
            bees.push_back(pair[int,int](p_bee[0], p_bee[1]))

        for i in range(bees.size()):
            bee_origin = bees[i]
            if procesed.find(bee_origin) != procesed.end():
                continue
            swarm.clear()
            check.clear()
            check.push_back(bee_origin)
            while check.size() > 0:
                bee = check.front()
                check.pop_front()
                procesed.insert(bee)

                for j in range(swarm.size()):
                    if bee == swarm[j]:
                        found = 1
                        break

                if found == 1:
                    found = 0
                    continue
                else:
                    swarm.push_back(bee)

                bee_l = pair[int,int](bee.first + 1, bee.second)
                bee_r = pair[int,int](bee.first, bee.second + 1)
                bee_u = pair[int,int](bee.first - 1, bee.second)
                bee_d = pair[int,int](bee.first, bee.second - 1)
                j = 0
                for j in range(bees.size()):
                    if bee_l == bees[j]:
                        check.push_back(bee_l)
                    elif bee_r == bees[j] :
                        check.push_back(bee_r)
                    elif bee_u == bees[j]:
                        check.push_back(bee_u)
                    elif bee_d == bees[j]:
                        check.push_back(bee_d)
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

    def _stop_time(self, bee):
        T_local = self.heatmap[bee[0], bee[1]]
        wait_time = max(
            int(self.k_stay / (1 + np.abs(self.T_ideal - T_local))),
            self.min_wait
        )
        return -wait_time

    def _obstacle_hit(self, bee):
        if np.random.rand() < self.p_wall:
            self.map[bee[0], bee[1]] = self._stop_time(bee)
        else:
            self.map[bee[0], bee[1]] = TURN[self.map[bee[0], bee[1]]]

    def _change_direction(self, bee, amnesia):
        moves = [BEE_LEFT, BEE_DOWN, BEE_RIGHT, BEE_UP]
        if not amnesia:
            moves.remove(int(self.map[bee[0], bee[1]]))
        new_dir = np.random.choice(moves)

        self.map[bee[0], bee[1]] = new_dir

    def _meet(self, bee):
        if np.random.rand() < self.p_meet:
            self.map[bee[0], bee[1]] = self._stop_time(bee)

    def tick(self) -> int:
        moved = 0
        bees = self.bees
        for bee in bees:
            # BEE WAITED
            if self.map[bee[0], bee[1]] < AMNESIA:
                self.map[bee[0], bee[1]] += 1
                continue

            # BEE CHANGE DIRECTION OR AMNESIA
            amnesia = self.map[bee[0], bee[1]] == AMNESIA
            if np.random.rand() < self.p_changedir or amnesia:
                self._change_direction(bee, amnesia)
                if amnesia:
                    continue

            # BEE TRIES TO GO UP
            if self.map[bee[0], bee[1]] == BEE_UP:
                if bee[0] - 1 < 0 or self.map[bee[0] - 1, bee[1]] in OBSTACLES:
                    self._obstacle_hit(bee)
                elif self.map[bee[0] - 1, bee[1]] < EMPTY or \
                        self.map[bee[0] - 1, bee[1]] in BEE:
                    self._meet(bee)
                else:
                    moved += 1
                    self.map[bee[0] - 1, bee[1]] = self.map[bee[0], bee[1]]
                    self.map[bee[0], bee[1]] = EMPTY

            # BEE TRIES TO GO DOWN
            elif self.map[bee[0], bee[1]] == BEE_DOWN:
                if bee[0] + 1 >= self.map.shape[0] or \
                        self.map[bee[0] + 1, bee[1]] in OBSTACLES:
                    self._obstacle_hit(bee)
                elif self.map[bee[0] + 1, bee[1]] < EMPTY or\
                        self.map[bee[0] + 1, bee[1]] in BEE:
                    self._meet(bee)
                else:
                    moved += 1
                    self.map[bee[0] + 1, bee[1]] = self.map[bee[0], bee[1]]
                    self.map[bee[0], bee[1]] = EMPTY

            # BEE TRIES TO GO LEFT
            elif self.map[bee[0], bee[1]] == BEE_LEFT:
                if bee[1] - 1 < 0 or self.map[bee[0], bee[1] - 1] in OBSTACLES:
                    self._obstacle_hit(bee)
                elif self.map[bee[0], bee[1] - 1] < EMPTY or\
                        self.map[bee[0], bee[1] - 1] in BEE:
                    self._meet(bee)
                else:
                    moved += 1
                    self.map[bee[0], bee[1] - 1] = self.map[bee[0], bee[1]]
                    self.map[bee[0], bee[1]] = EMPTY

            # BEE TRIES TO GO RIGHT
            elif self.map[bee[0], bee[1]] == BEE_RIGHT:
                if bee[1] + 1 >= self.map.shape[1] or \
                        self.map[bee[0], bee[1] + 1] in OBSTACLES:
                    self._obstacle_hit(bee)
                elif self.map[bee[0], bee[1] + 1] < EMPTY or\
                        self.map[bee[0], bee[1] + 1] in BEE:
                    self._meet(bee)
                else:
                    moved += 1
                    self.map[bee[0], bee[1] + 1] = self.map[bee[0], bee[1]]
                    self.map[bee[0], bee[1]] = EMPTY

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

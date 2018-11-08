from .utils import calculate_heatmap, np, check_type
from .constants import *


class BeeClust:
    def __init__(self, map, p_changedir=0.2, p_wall=0.8,
                 p_meet=0.8, k_temp=0.9, k_stay=50, T_ideal=35,
                 T_heater=40, T_cooler=5, T_env=22, min_wait=2):
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
            raise ValueError('koeficients cant be negative')

        if T_heater < T_env:
            raise ValueError('T_heater cant be lower then T_env')

        if T_cooler > T_env:
            raise ValueError('T_cooler cant be higher then T_env')

        if not len(self.map.shape) == 2:
            raise ValueError('invalid shape')

        self.heatmap = calculate_heatmap(map, self.T_heater,
                                         self.T_cooler, self.T_env,
                                         self.k_temp)

    @property
    def bees(self):
        bees = []
        for row in range(self.map.shape[0]):
            for col in range(self.map.shape[1]):
                if self.map[row, col] == EMPTY or self.map[row, col] > BEE_LEFT:
                    pass
                else:
                    bees.append((row, col))
        return bees

    @property
    def swarms(self):
        procesed = set()
        swarms = []
        bees = self.bees
        for bee_origin in bees:
            if bee_origin in procesed:
                continue
            swarm = []
            check = {bee_origin}
            while len(check) > 0:
                bee = check.pop()
                procesed.add(bee)
                if bee in swarm:
                    continue
                else:
                    swarm.append(bee)
                if (bee[0] + 1, bee[1]) in bees:
                    check.add((bee[0] + 1, bee[1]))
                if (bee[0], bee[1] + 1) in bees:
                    check.add((bee[0], bee[1] + 1))
                if (bee[0] - 1, bee[1]) in bees:
                    check.add((bee[0] - 1, bee[1]))
                if (bee[0], bee[1] - 1) in bees:
                    check.add((bee[0], bee[1] - 1))
            swarms.append(swarm)
        return list(swarms)

    @property
    def score(self):
        temps = 0
        bees = self.bees
        for bee in bees:
            temps += self.heatmap[bee[0], bee[1]]
        return temps / len(bees)

    def tick(self):
        pass

    def forget(self):
        for bee in self.bees:
            self.map[bee[0], bee[1]] = AMNESIA

    def recalculate_heat(self):
        self.heatmap = \
            calculate_heatmap(self.map, self.T_heater,
                              self.T_cooler, self.T_env, self.k_temp)

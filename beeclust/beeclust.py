from .utils import calculate_heatmap
from .constants import *


class BeeClust:
    def __init__(self, np_map, p_changedir=0.2, p_wall=0.8,
                 p_meet=0.8, k_temp=0.9, k_stay=50, T_ideal=35,
                 T_heater=40, T_cooler=5, T_env=22, min_wait=2):
        self.map = np_map
        self.heatmap = calculate_heatmap(np_map)
        self.p_changedir = p_changedir
        self.p_wall = p_wall
        self.p_meet = p_meet
        self.k_temp = k_temp
        self.k_stay = k_stay
        self.T_ideal = T_ideal
        self.T_heater = T_heater
        self.T_cooler = T_cooler
        self.T_env = T_env
        self.min_wait = min_wait

    @property
    def bees(self):
        bees = []
        for row in range(self.map.shape[0]):
            for col in range(self.map.shape[1]):
                if self.map[row, col] != EMPTY or not self.map[row, col] > BEE_LEFT:
                    bees.append((row, col))
        return bees

    @property
    def swarms(self):
        procesed = set()
        swarms = {}
        bees = self.bees
        for bee_origin in bees:
            if bee_origin in procesed:
                continue
            swarm = [bee_origin]
            check = {bee_origin}
            while len(check) > 0:
                bee = check.pop()
                procesed.add(bee)
                if bee in swarm:
                    continue
                if (bee[0] + 1, bee[1]) in bees:
                    check.add(bee)
                    swarm.append(bee)
                if (bee[0], bee[1] + 1) in bees:
                    check.add(bee)
                    swarm.append(bee)
                if (bee[0] - 1, bee[1]) in bees:
                    check.add(bee)
                    swarm.append(bee)
                if (bee[0], bee[1] - 1) in bees:
                    check.add(bee)
                    swarm.append(bee)
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

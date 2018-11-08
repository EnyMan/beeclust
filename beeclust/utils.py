from .constants import *

import numpy as np
from collections import deque


def check_type(check, targets, key):
    for target in targets:
        if isinstance(check, target):
            return check

    raise TypeError(f'{key} is invalid type')


class Coordinates:
    def __init__(self, a, b, shape, steps=0):
        self.x = a
        self.y = b
        self.shape = shape
        self.steps = steps

    def __str__(self):
        return f'x={self.x}, y={self.y}, s={self.steps}'

    def __hash__(self):
        return hash(f'{self.x}:{self.y}')

    def __eq__(self, other):
        if isinstance(other, Coordinates):
            return other.x == self.x and other.y == self.y
        else:
            return False

    def copy(self):
        return Coordinates(self.x, self.y, self.shape, self.steps)

    def step(self):
        self.steps += 1

    def move_up(self):
        if self.x - 1 >= 0:
            new = self.copy()
            new.x -= 1
            return new

    def move_left(self):
        if self.y - 1 >= 0:
            new = self.copy()
            new.y -= 1
            return new

    def move_right(self):
        if self.y + 1 < self.shape[1]:
            new = self.copy()
            new.y += 1
            return new

    def move_down(self):
        if self.x + 1 < self.shape[0]:
            new = self.copy()
            new.x += 1
            return new

    def move_up_right(self):
        if self.x - 1 >= 0 and self.y + 1 < self.shape[1]:
            new = self.copy()
            new.x -= 1
            new.y += 1
            return new

    def move_up_left(self):
        if self.x - 1 >= 0 and self.y - 1 >= 0:
            new = self.copy()
            new.x -= 1
            new.y -= 1
            return new

    def move_down_right(self):
        if self.x + 1 < self.shape[0] and self.y + 1 < self.shape[1]:
            new = self.copy()
            new.x += 1
            new.y += 1
            return new

    def move_down_left(self):
        if self.x + 1 < self.shape[0] and self.y - 1 >= 0:
            new = self.copy()
            new.x += 1
            new.y -= 1
            return new


def calculate_heatmap(np_map, T_heater, T_cooler, T_env, k_temp):
    heatmap = np.zeros(np_map.shape)
    for row in range(np_map.shape[0]):
        for col in range(np_map.shape[1]):
            if np_map[row, col] == HEATER:
                heatmap[row, col] = T_heater
            elif np_map[row, col] == COOLER:
                heatmap[row, col] = T_cooler
            elif np_map[row, col] == WALL:
                heatmap[row, col] = np.nan
            else:
                dist_cooler = np.inf
                dist_heater = np.inf
                queue = deque()
                queue.append(Coordinates(row, col, np_map.shape))
                visited = set()
                while len(queue) > 0:
                    start = queue.popleft()
                    if start in visited:
                        continue
                    else:
                        visited.add(start)
                    if np_map[start.x, start.y] == HEATER:
                        if start.steps < dist_heater:
                            dist_heater = start.steps
                        continue
                    if np_map[start.x, start.y] == COOLER:
                        if start.steps < dist_cooler:
                            dist_cooler = start.steps
                        continue
                    if np_map[start.x, start.y] == WALL:
                        continue
                    if start.steps >= np_map.shape[0] * np_map.shape[1]:
                        break
                    start.step()

                    move = start.move_up()
                    queue.append(move) if move else None
                    move = start.move_left()
                    queue.append(move) if move else None
                    move = start.move_right()
                    queue.append(move) if move else None
                    move = start.move_down()
                    queue.append(move) if move else None
                    move = start.move_up_right()
                    queue.append(move) if move else None
                    move = start.move_up_left()
                    queue.append(move) if move else None
                    move = start.move_down_right()
                    queue.append(move) if move else None
                    move = start.move_down_left()
                    queue.append(move) if move else None

                heating = (1 / dist_heater) * (T_heater - T_env)
                cooling = (1 / dist_cooler) * (T_env - T_cooler)
                heatmap[row, col] = \
                    T_env + k_temp * (max(heating, 0) - max(cooling, 0))
    return heatmap

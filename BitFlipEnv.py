from typing import Tuple
import numpy as np


class BitFlipEnv:
    def __init__(self, n: int) -> None:
        self.n = n

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.state = np.random.randint(0, 2, self.n)
        self.goal_state = np.random.randint(0, 2, self.n)
        return np.copy(self.state), self.goal_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.state[action] ^= 1
        if np.array_equal(self.state, self.goal_state):
            return np.copy(self.state), 0.0, True
        return np.copy(self.state), -1.0, False

    def __str__(self) -> str:
        return f"Current state: {self.state.__str__()}\n"\
               f"Goal state: {self.goal_state.__str__()}"

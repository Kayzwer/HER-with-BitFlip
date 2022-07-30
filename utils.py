import torch
import numpy as np
from typing import Dict, Tuple


class ExperienceReplayBuffer:
    def __init__(self, state_dim: int, buffer_size: int, batch_size: int
                 ) -> None:
        self.state_memory = np.zeros((buffer_size, state_dim),
                                     dtype=np.float32)
        self.action_memory = np.zeros(buffer_size, dtype=np.longlong)
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, state_dim),
                                          dtype=np.float32)
        self.done_memory = np.zeros(buffer_size, dtype=np.bool8)
        self.goal_memory = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.ptr, self.cur_size = 0, 0

    def store(self, state: np.ndarray, action: float, reward: float,
              next_state: np.ndarray, done: bool, goal: np.ndarray) -> None:
        self.state_memory[self.ptr] = state
        self.action_memory[self.ptr] = action
        self.reward_memory[self.ptr] = reward
        self.next_state_memory[self.ptr] = next_state
        self.done_memory[self.ptr] = done
        self.goal_memory[self.ptr] = goal
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)

    def sample(self) -> Dict[str, torch.Tensor]:
        idxs = np.random.choice(self.cur_size, self.batch_size, False)
        return {
            "states": torch.from_numpy(self.state_memory[idxs]),
            "actions": torch.from_numpy(self.action_memory[idxs]),
            "rewards": torch.from_numpy(self.reward_memory[idxs]),
            "next_states": torch.from_numpy(self.next_state_memory[idxs]),
            "dones": torch.from_numpy(self.done_memory[idxs]),
            "goals": torch.from_numpy(self.goal_memory[idxs])
        }

    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


class EpsilonController:
    def __init__(self, init_eps: float, eps_dec_rate: str, min_eps: float
                 ) -> None:
        self.eps = init_eps
        self._deci_place, self.eps_dec_rate = self._get_deci_place(
            eps_dec_rate)
        self.min_eps = min_eps

    def _get_deci_place(self, eps_dec_rate: str) -> Tuple[int, float]:
        after_dot = False
        count = 0
        for char in eps_dec_rate:
            if char == ".":
                after_dot = True
            if after_dot:
                count += 1
        return count, float(eps_dec_rate)

    def decay(self) -> None:
        self.eps = round(self.eps - self.eps_dec_rate, self._deci_place) if \
            self.eps > self.min_eps else self.min_eps

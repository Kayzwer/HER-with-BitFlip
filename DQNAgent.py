from utils import ExperienceReplayBuffer, EpsilonController
from models import Network
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch


class Agent:
    def __init__(self, state_dim: int, action_dim: int, lr: float,
                 buffer_size: int, batch_size: int, init_eps: float,
                 eps_dec_rate: str, min_eps: float, gamma: float, c: int
                 ) -> None:
        self.network = Network(state_dim * 2, action_dim)
        self.target_network = Network(state_dim * 2, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr)
        self.update_target_network()
        self.replay_buffer = ExperienceReplayBuffer(state_dim, buffer_size,
                                                    batch_size)
        self.epsilon_controller = EpsilonController(init_eps, eps_dec_rate,
                                                    min_eps)
        self.action_dim, self.gamma = action_dim, gamma
        self.update_count, self.c = 0, c

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def choose_action(self, state: np.ndarray, goal: np.ndarray) -> int:
        if np.random.random() > self.epsilon_controller.eps:
            input_ = np.concatenate([state, goal], dtype=np.float32)
            return self.network.forward(
                torch.from_numpy(input_)).detach().argmax().item()
        else:
            return np.random.randint(self.action_dim)

    def update(self) -> float:
        if self.replay_buffer.is_ready():
            batch = self.replay_buffer.sample()
            states = batch.get("states")
            actions = batch.get("actions").view(-1, 1)
            rewards = batch.get("rewards").view(-1, 1)
            next_states = batch.get("next_states")
            not_dones = ~batch.get("dones").view(-1, 1)
            goals = batch.get("goals")
            states = torch.hstack((states, goals))
            next_states = torch.hstack((next_states, goals))

            q_pred = self.network.forward(states).gather(1, actions)
            q_next = self.target_network.forward(
                next_states).max(dim=1)[0].view(-1, 1)
            q_target = rewards + self.gamma * q_next * not_dones
            self.optimizer.zero_grad()
            loss = F.mse_loss(q_pred, q_target.detach())
            loss.backward()
            self.optimizer.step()
            self.update_count += 1
            if self.update_count % self.c == 0:
                self.update_target_network()
            self.epsilon_controller.decay()
            return loss.item()
        return 0.0

from BitFlipEnv import BitFlipEnv
from DQNAgent import Agent
import numpy as np


if __name__ == "__main__":
    n_bit = 8
    episodes = 30000
    env = BitFlipEnv(n_bit)
    agent = Agent(n_bit, n_bit, 0.0001, 10000, 64, 0.9, "0.00001", 0.1, 0.99,
                  50)

    success = 0
    transitions = []
    for episode in range(episodes):
        state, goal = env.reset()
        done = False
        score = 0.0
        loss = 0.0
        for _ in range(n_bit):
            action = agent.choose_action(state, goal)
            next_state, reward, done = env.step(action)
            agent.replay_buffer.store(state, action, reward, next_state, done,
                                      goal)
            transitions.append((state, action, reward, next_state))
            state = next_state
            score += reward
            loss += agent.update()
            if done:
                success += 1
                break
        if not done:
            new_goal = np.copy(state)
            for transition in transitions:
                if np.array_equal(transition[3], new_goal):
                    agent.replay_buffer.store(transition[0], transition[1],
                                              0.0, transition[3], True,
                                              new_goal)
                    loss += agent.update()
                else:
                    agent.replay_buffer.store(transition[0], transition[1],
                                              transition[2], transition[3],
                                              False, new_goal)
                    loss += agent.update()
        transitions.clear()
        print(f"Episode: {episode + 1}, Score: {score}, Loss: {loss:.3f}, "
              f"Success Rate: {success / (episode + 1):.3f}")

import gym
import numpy as np
import time

SEED = 42
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPISODES = 10000

env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset(seed=SEED)

q_table = np.zeros((env.observation_space.shape[0], env.action_space.n))
last_failure = 0

for episode in range(EPISODES):
    total_reward = 0

    for step in range(1000):
        state = observation
        action = np.argmax(q_table[state])

        next_observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + \
                                 LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[next_observation]))

        observation = next_observation

        if terminated or truncated:
            time.sleep(0.5)
            observation, info = env.reset()
            print(f"Episode {episode} terminated after {step - last_failure + 1} steps. Total reward: {total_reward}")
            last_failure = step + 1
            break

env.close()

import gym
import numpy as np
import time

SEED = 42

env = gym.make("ALE/Breakout-v5", render_mode = "human")
observation, info = env.reset(seed = SEED)

lastFailure = 0
for i in range (1000):
    action = np.random.randint(0, env.action_space.n)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        time.sleep(0.5)
        observation, info = env.reset()
        print(f"Enviroment stopped after {i-lastFailure} steps")
        lastFailure = i

env.close()
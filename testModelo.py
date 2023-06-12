import gym
import numpy as np
import time
import pickle

# Carregar a tabela Q de um arquivo
filename = "q_table.pkl"
with open(filename, "rb") as f:
    Q = pickle.load(f)

SEED = 42

env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset(seed=SEED)

total_reward = 0

while True:
    # Escolha da ação com base na tabela Q
    action = np.argmax(Q[observation])

    # Executa a ação no ambiente
    next_observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    # Verificação se o ambiente foi reiniciado
    if terminated or truncated:
        time.sleep(0.5)
        next_observation, info = env.reset()
        print(f"Episode terminated. Total reward: {total_reward}")
        break

    observation = next_observation

env.close()

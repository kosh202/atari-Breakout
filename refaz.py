import gym
import numpy as np
import time
import pickle

SEED = 42

env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset(seed=SEED)

# Inicialização da tabela Q
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
Q = np.zeros((num_states, num_actions))

# Parâmetros do Q-learning
alpha = 0.5  # taxa de aprendizado
gamma = 0.9  # fator de desconto
epsilon = 0.1  # taxa de exploração (epsilon-greedy)

lastFailure = 0
for i in range(1000):
    # Escolha da ação com base na política epsilon-greedy
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(Q[observation])

    # Executa a ação no ambiente
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Atualização da tabela Q
    Q[observation, action] += alpha * (reward + gamma * np.max(Q[next_observation]) - Q[observation, action])

    # Verificação se o ambiente foi reiniciado
    if terminated or truncated:
        time.sleep(0.5)
        next_observation, info = env.reset()
        print(f"Enviroment stopped after {i - lastFailure} steps")
        lastFailure = i

    observation = next_observation

env.close()

# Salvar a tabela Q em um arquivo
filename = "q_table.pkl"
with open(filename, "wb") as f:
    pickle.dump(Q, f)

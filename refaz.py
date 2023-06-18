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
alpha = 0.1  # taxa de aprendizado
gamma = 0.2  # fator de desconto
epsilon = 0.8  # taxa de exploração (epsilon-greedy)
total_rewards = 0
num_episodes = 10000
epAtual = 0

lastFailure = 0
for i in range(num_episodes):
    # Escolha da ação com base na política epsilon-greedy
    epAtual = i
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(Q[observation])

    # Executa a ação no ambiente
    next_observation, reward, terminated, truncated, info = env.step(action)
    total_rewards += reward


    # Atualização da tabela Q
    Q[observation, action] += alpha * ((reward-5) + gamma * np.max(Q[next_observation]) - Q[observation, action])

    # Verificação se o ambiente foi reiniciado
    if terminated or truncated:
        time.sleep(0.3)
        next_observation, info = env.reset()
        print(f"Enviroment stopped after {i - lastFailure} steps\nNum ep: {epAtual}")
        lastFailure = i

    observation = next_observation
    #print("reward: ", reward)

env.close()

average_reward = total_rewards / num_episodes
print("Average Reward:", average_reward)

# Salvar a tabela Q em um arquivo
filename = "q_table.pkl"
with open(filename, "wb") as f:
    pickle.dump(Q, f)

print(Q)

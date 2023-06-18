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

# import gymnasium as gym
# import numpy as np
# import tensorflow as tf

# # Carregar o modelo treinado
# modelo_carregado = tf.keras.models.load_model("modelo_treinado.h5")

# # Criação do ambiente
# env = gym.make("ALE/Breakout-v5", render_mode="human")
# observation, info = env.reset()

# state_shape = (210, 160, 3)
# num_actions = 4  # Exemplo: 4 ações possíveis no Breakout

# total_rewards = 0
# num_episodes = 10  # Número de episódios para jogar

# # Loop de episódios
# for episode in range(num_episodes):
#     terminated = False
#     total_reward = 0

#     while not terminated:
#         # Obter a ação com base no modelo carregado
#         q_values = modelo_carregado.predict(np.expand_dims(observation, axis=0))
#         action = np.argmax(q_values)

#         observation, reward, terminated, _, _ = env.step(action)
#         total_reward += reward

#         env.render()  # Renderizar o ambiente para exibir o jogo

#     print(f"Episode: {episode}, Total Reward: {total_reward}")
#     total_rewards += total_reward

# env.close()

# average_reward = total_rewards / num_episodes
# print("Average Reward:", average_reward)

#por enquanto funcional

import gymnasium as gym
import numpy as np
import tensorflow as tf

# Função para criar a rede neural Q
def create_q_network(state_shape, num_actions):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=state_shape))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(num_actions))

    return model

def update_target_network(target_network, main_network):
    target_network.set_weights(main_network.get_weights())

# Criação do ambiente
env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset()

state_shape = (210, 160, 3)
num_actions = 4  # Exemplo: 4 ações possíveis no Breakout

q_network = create_q_network(state_shape, num_actions)
target_network = create_q_network(state_shape, num_actions)
update_target_network(target_network, q_network)  # Inicialização da rede de destino com os pesos da rede principal

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # Otimizador para atualização dos pesos
gamma = 0.8  # Fator de desconto

update_frequency = 100  # Intervalo de atualização da rede de destino
update_counter = 0

total_rewards = 0
num_episodes = 1000

# Loop de episódios
for episode in range(num_episodes):
    action = env.action_space.sample()  # Política atual
    observation, reward, terminated, truncated, info = env.step(action)

    total_rewards += reward

    # Atualização dos valores Q
    target = reward

    print("reward: ",reward)
    if not terminated and not truncated:
        next_q_values = target_network.predict(np.expand_dims(observation, axis=0))
        target = reward + gamma * np.max(next_q_values)

    with tf.GradientTape() as tape:
        q_values = q_network(np.expand_dims(observation, axis=0))
        q_value = tf.reduce_sum(tf.multiply(q_values, tf.one_hot(action, num_actions)))
        loss = tf.square(target - q_value)

    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    if update_counter % update_frequency == 0:
        update_target_network(target_network, q_network)

    update_counter += 1

    if terminated or truncated:
        print(f"Episode: {episode}, Loss: {loss}, Q-Values: {q_values}, Total Rewards: {total_rewards}")
        #total_rewards = 0
        observation, info = env.reset()

env.close()

q_network.save('modelo_treinado.h5')

average_reward = total_rewards / num_episodes
print("Average Reward:", average_reward)

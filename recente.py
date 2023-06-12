import gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Criação do ambiente
env = gym.make("ALE/Breakout-v5", render_mode="human")

# Parâmetros do agente DQN
gamma = 0.99  # Fator de desconto
epsilon = 1.0  # Taxa de exploração inicial
epsilon_decay = 0.995  # Decaimento da taxa de exploração
epsilon_min = 0.01  # Taxa de exploração mínima
learning_rate = 0.001  # Taxa de aprendizado
memory = []  # Memória de replay
memory_capacity = 10000  # Capacidade da memória de replay
batch_size = 32  # Tamanho do lote para atualização da rede

# Criação do modelo DQN
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)),
        layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(env.action_space.n)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Inicialização da rede DQN
model = create_model()

# Função para pré-processar a imagem de entrada
def preprocess_state(state):
    state = state[0][35:195, :]  # Cortar a região de interesse da imagem
    state = state[::2, :]  # Reduzir a resolução pela metade
    state = np.mean(state, axis=2).astype(np.uint8)  # Converter para escala de cinza
    state = state.reshape(1, 80, 80, 1)  # Redimensionar a imagem para (1, 80, 80, 1)
    return state

# Função para tomar uma ação com base na política epsilon-greedy
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return np.argmax(model.predict(state))

# Laço principal de treinamento
for episode in range(500):
    state = env.reset()
    state = preprocess_state(state)
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = choose_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)[0], env.step(action)[1], env.step(action)[2], {}
        next_state = preprocess_state(next_state)
        total_reward += reward

        # Armazenar a transição na memória de replay
        memory.append((state, action, reward, next_state, done))
        if len(memory) > memory_capacity:
            del memory[0]

        state = next_state

        # Realizar o treinamento da rede DQN
        if len(memory) > batch_size:
            batch = np.random.choice(memory, batch_size, replace=False)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*batch))
            q_values_next = model.predict(next_states_batch)
            targets_batch = reward_batch + gamma * np.max(q_values_next, axis=1) * (1 - done_batch)
            target_f = model.predict(states_batch)
            target_f[np.arange(batch_size), action_batch] = targets_batch
            model.fit(states_batch, target_f, epochs=1, verbose=0)

    # Atualizar a taxa de exploração
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode+1} - Reward: {total_reward}")

env.close()

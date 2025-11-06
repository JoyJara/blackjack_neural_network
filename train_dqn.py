import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from blackjack_env import BlackjackEnv
from dqn_neuraljack import DQN
from replay_buffer import ReplayBuffer
from tqdm import trange

import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots()
avg_rewards = []


# --- Configuración ---
env = BlackjackEnv(casino_type=1)
state = env._state_to_array(env.reset(force_new_shoe=False))
state_dim = len(state)
n_actions = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(state_dim, n_actions).to(device)
target_model = DQN(state_dim, n_actions).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

buffer = ReplayBuffer(capacity=100000)

epsilon = 1.0        # exploración inicial
epsilon_min = 0.05
epsilon_decay = 0.999

gamma = 0.99          # descuento futuro
batch_size = 128
target_update = 1000   # cada 100 episodios actualiza la red objetivo

episodes = 500000
rewards_log = []
epsilons = []
ax2 = ax.twinx()

for episode in trange(episodes):
    state = env._state_to_array(env.reset())
    total_reward = 0

    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Política ε-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

        next_state_raw, reward, done, _ = env.step(action)
        next_state = env._state_to_array(next_state_raw)

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Entrenamiento si hay suficientes muestras
        if len(buffer) >= batch_size:
            s, a, r, s_next, d = buffer.sample(batch_size)

            s = torch.tensor(s, dtype=torch.float32).to(device)
            a = torch.tensor(a, dtype=torch.long).to(device)
            r = torch.tensor(r, dtype=torch.float32).to(device)
            s_next = torch.tensor(s_next, dtype=torch.float32).to(device)
            d = torch.tensor(d, dtype=torch.float32).to(device)

            # Q(s,a)
            q_values = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            # Q_target = r + γ * max_a' Q_target(s', a')
            with torch.no_grad():
                max_next_q = target_model(s_next).max(1)[0]
                q_target = r + gamma * max_next_q * (1 - d)

            loss = criterion(q_values, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    rewards_log.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    epsilons.append(epsilon)

    # Actualizar red objetivo
    if episode % target_update == 0:
        target_model.load_state_dict(model.state_dict())

    if (episode + 1) % 1000 == 0:
        avg = np.mean(rewards_log[-100:])
        avg_rewards.append(avg)
        eps_to_plot = epsilons[::1000]  # <-- reduce puntos de epsilon

        ax.clear()
        ax2.clear()

        # Línea azul: recompensa promedio
        ax.plot(avg_rewards, label='Promedio de recompensa (últimos 100)', color='tab:blue')
        ax.set_xlabel("Episodios / 1000")
        ax.set_ylabel("Recompensa promedio", color='tab:blue')

        # Línea naranja: epsilon
        ax2.plot(eps_to_plot, label='Epsilon', color='tab:orange', alpha=0.7)
        ax2.set_ylabel("Epsilon", color='tab:orange')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='lower right')

        plt.pause(0.001)

torch.save(model.state_dict(), "dqn_blackjack.pth")
print("Entrenamiento completado.")

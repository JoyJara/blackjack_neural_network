import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from blackjack_env import BlackjackEnv
from dqn_neuraljack import DQN
from replay_buffer import ReplayBuffer
from tqdm import trange

import matplotlib.pyplot as plt
from datetime import datetime  # para el nombre de los modelos

# ===================== #
#   CONFIGURACIÓN BASE  #
# ===================== #

SEED = 42

# False -> solo 2 acciones (plantarse, pedir)
# True  -> 5 acciones (plantarse, pedir, doblar, dividir, rendirse)
USE_ALL_ACTIONS = False

# Semillas para reproducibilidad
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Detección de dispositivo
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")  # para Macs con Apple Silicon
else:
    device = torch.device("cpu")

print(f"Usando dispositivo: {device}")

# ===================== #
#   FUNCIÓN EVALUACIÓN  #
# ===================== #

def evaluate_policy(env, model, device, episodes_eval=10000):
    """
    Evalúa la política actual sin exploración (acción greedy)
    y muestra win rate, loss rate, push rate y recompensa promedio.
    """
    model.eval()
    wins = 0
    losses = 0
    pushes = 0
    total_rewards = 0.0

    for _ in range(episodes_eval):
        state = env._state_to_array(env.reset())
        done = False
        ep_reward = 0.0

        while not done:
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

            next_state_raw, reward, done, _ = env.step(action)
            state = env._state_to_array(next_state_raw)
            ep_reward += reward

        total_rewards += ep_reward

        if ep_reward > 0:
            wins += 1
        elif ep_reward < 0:
            losses += 1
        else:
            pushes += 1

    win_rate = wins / episodes_eval * 100.0
    loss_rate = losses / episodes_eval * 100.0
    push_rate = pushes / episodes_eval * 100.0
    avg_reward = total_rewards / episodes_eval

    print("\n----- Evaluación de la política (greedy, sin exploración) -----")
    print(f"Episodios de evaluación: {episodes_eval}")
    print(f"Win rate  : {win_rate:.2f}%")
    print(f"Loss rate : {loss_rate:.2f}%")
    print(f"Push rate : {push_rate:.2f}%")
    print(f"Reward promedio por episodio: {avg_reward:.4f}")
    print("--------------------------------------------------------------\n")

# ===================== #
#      ENTORNO DQN      #
# ===================== #

plt.ion()
fig, ax = plt.subplots()
avg_rewards = []

env = BlackjackEnv(casino_type=1)

# Estado inicial para saber la dimensión de entrada
state = env._state_to_array(env.reset(force_new_shoe=False))
state_dim = len(state)

# Acciones: por ahora 2 (hit/stand). Más adelante puedes cambiar a 5.
n_actions = 5 if USE_ALL_ACTIONS else 2

# Redes Q
model = DQN(state_dim, n_actions).to(device)
target_model = DQN(state_dim, n_actions).to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()  # la red objetivo no se entrena directamente

optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

buffer = ReplayBuffer(capacity=100_000)

# Hiperparámetros de RL
epsilon = 1.0          # exploración inicial
epsilon_min = 0.05
epsilon_decay = 0.999

gamma = 0.99           # factor de descuento
batch_size = 128
target_update = 1_000  # cada 1000 episodios copia pesos a la red objetivo

episodes = 500_000
rewards_log = []
epsilons = []

ax2 = ax.twinx()

# ===================== #
#   BUCLE DE ENTRENOS   #
# ===================== #

for episode in trange(episodes):
    state = env._state_to_array(env.reset())
    total_reward = 0.0
    done = False

    while not done:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Política ε-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(n_actions)  # 0..n_actions-1
        else:
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values, dim=1).item()

        next_state_raw, reward, done, _ = env.step(action)
        next_state = env._state_to_array(next_state_raw)

        buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Entrenamiento si el buffer tiene suficientes muestras
        if len(buffer) >= batch_size:
            s, a, r, s_next, d = buffer.sample(batch_size)

            s = torch.tensor(s, dtype=torch.float32, device=device)
            a = torch.tensor(a, dtype=torch.long, device=device)
            r = torch.tensor(r, dtype=torch.float32, device=device)
            s_next = torch.tensor(s_next, dtype=torch.float32, device=device)
            d = torch.tensor(d, dtype=torch.float32, device=device)

            # Q(s, a) actual
            q_values = model(s).gather(1, a.unsqueeze(1)).squeeze(1)

            # Objetivo: r + γ * max_a' Q_target(s', a')
            with torch.no_grad():
                max_next_q = target_model(s_next).max(1)[0]
                q_target = r + gamma * max_next_q * (1.0 - d)

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

    # Actualizar gráfica cada 1000 episodios
    if (episode + 1) % 1000 == 0:
        avg = np.mean(rewards_log[-100:])
        avg_rewards.append(avg)
        eps_to_plot = epsilons[::1000]  # reduce puntos en la gráfica de epsilon

        ax.clear()
        ax2.clear()

        # Recompensa promedio
        ax.plot(avg_rewards, label='Promedio recompensa (últimos 100)', color='tab:blue')
        ax.set_xlabel("Episodios / 1000")
        ax.set_ylabel("Recompensa promedio", color='tab:blue')

        # Epsilon
        ax2.plot(eps_to_plot, label='Epsilon', color='tab:orange', alpha=0.7)
        ax2.set_ylabel("Epsilon", color='tab:orange')

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='lower right')

        plt.pause(0.001)

# ===================== #
#   EVALUAR Y GUARDAR   #
# ===================== #

# Evaluar política final sin exploración
evaluate_policy(env, model, device, episodes_eval=10000)

# Guardar modelo con nombre único para no sobreescribir
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"dqn_blackjack_{timestamp}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Entrenamiento completado. Modelo guardado en {model_filename}")

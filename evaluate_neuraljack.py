import torch
import numpy as np
from blackjack_env import BlackjackEnv
from dqn_neuraljack import DQN

# --- Configuración ---
EPISODES = 100000          # número de manos para evaluar
CASINO_TYPE = 1            # 1 o 2 según el entorno que quieras probar
MODEL_PATH = "dqn_blackjack.pth"

# --- Configurar dispositivo (forzar CPU) ---
device = torch.device("cpu")
print("Usando dispositivo:", device)

# --- Inicializar entorno y modelo ---
env = BlackjackEnv(casino_type=CASINO_TYPE)
state = env._state_to_array(env.reset())

state_dim = len(state)
n_actions = 2

model = DQN(state_dim, n_actions).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Métricas ---
wins = 0
pushes = 0
losses = 0
total_reward = 0

# --- Evaluación ---
for episode in range(EPISODES):
    state = env._state_to_array(env.reset(force_new_shoe=False))
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        next_state_raw, reward, done, _ = env.step(action)
        state = env._state_to_array(next_state_raw)
        total_reward += reward

    # Clasificar resultado final
    if reward > 0:
        wins += 1
    elif reward == 0:
        pushes += 1
    else:
        losses += 1

# --- Resultados finales ---
win_rate = wins / EPISODES
push_rate = pushes / EPISODES
loss_rate = losses / EPISODES
avg_reward = total_reward / EPISODES

print("\nRESULTADOS DEL AGENTE")
print(f"Episodios evaluados: {EPISODES}")
print(f"Win rate:   {win_rate:.2%}")
print(f"Push rate:  {push_rate:.2%}")
print(f"Loss rate:  {loss_rate:.2%}")
print(f"Recompensa promedio: {avg_reward:.3f}")

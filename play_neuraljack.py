import torch, time
import numpy as np
from blackjack_env import BlackjackEnv
from dqn_neuraljack import DQN

# --- Configuración general ---
device = torch.device("cpu")
print("Usando dispositivo:", device)

HUMAN_SPEED = True     # Si True: pausas más largas (modo visual humano)
PAUSE_ACTION = 2.0     # segundos entre acciones
PAUSE_ROUND = 4.0      # segundos entre rondas

# --- Inicializar entorno y modelo ---
env = BlackjackEnv(casino_type=1)
state = env._state_to_array(env.reset())
state_dim = len(state)
n_actions = 2

model = DQN(state_dim, n_actions).to(device)
model.load_state_dict(torch.load("dqn_blackjack.pth", map_location=device))
model.eval()

# --- Función auxiliar para mostrar el estado actual ---
def mostrar_estado(action, reward, done):
    print("\n=========================")
    print(f"Acción tomada: {action}")
    env.render(reveal_dealer=done)
    print(f"Recompensa: {reward}")
    print("=========================\n")

# --- Simulación del juego ---
while True:
    done = False
    state = env._state_to_array(env.reset())
    env.render()  # Mostrar las cartas iniciales
    time.sleep(2.0 if HUMAN_SPEED else 0.5)

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        # Ejecutar acción y mostrar resultado paso a paso
        next_state_raw, reward, done, _ = env.step(action)
        mostrar_estado(action, reward, done)

        # Actualizar estado
        state = env._state_to_array(next_state_raw)

        # Pausa para simular juego humano
        time.sleep(PAUSE_ACTION if HUMAN_SPEED else 0.5)

    print("\nRonda terminada")
    time.sleep(PAUSE_ROUND if HUMAN_SPEED else 1.0)

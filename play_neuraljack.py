import torch, time
import numpy as np
from blackjack_env import BlackjackEnv
from dqn_neuraljack import DQN

env = BlackjackEnv(casino_type=1)
state = env._state_to_array(env.reset())
state_dim = len(state)
n_actions = 2

model = DQN(state_dim, n_actions)
model.load_state_dict(torch.load("dqn_blackjack.pth"))
model.eval()

done = False
while True:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()

    next_state_raw, reward, done, _ = env.step(action)
    env.render(reveal_dealer=done)
    print(f"Acción tomada: {action}, Recompensa: {reward}")

    state = env._state_to_array(next_state_raw)

    # Pausa de 1 segundo entre jugadas (como un humano)
    time.sleep(2.0)

    if done:
        print("\n--- Ronda terminada ---")
        time.sleep(2.0)
        state = env._state_to_array(env.reset())

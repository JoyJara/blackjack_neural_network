import torch
import numpy as np
from blackjack_env import BlackjackEnv
from dqn_neuraljack import DQN

# Crear entorno
env = BlackjackEnv(casino_type=1)
state = env.reset()
state_array = env._state_to_array(state)

# Convertir a tensor
state_tensor = torch.tensor(state_array, dtype=torch.float32)

# Crear red
input_dim = len(state_array)
output_dim = 5  # Stand, Hit, Double, Split, Surrender
model = DQN(input_dim, output_dim)

# Probar forward pass
q_values = model(state_tensor)
print("Valores Q predichos:", q_values)
print("Acción con mayor Q:", torch.argmax(q_values).item())

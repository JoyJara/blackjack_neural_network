from blackjack_env import BlackjackEnv

# Creamos el entorno con el casino tipo 2 (par perfecto / mixto)
env = BlackjackEnv(casino_type=2)

# --- Caso 1: Par perfecto (mismo valor y mismo palo)
perfect_pair = [("8", "♠"), ("8", "♠")]
reward_perfect = env._calculate_side_bet(perfect_pair)
print("Caso 1 - Par perfecto:", reward_perfect)  # Esperado: 15

# --- Caso 2: Par mixto (mismo valor, distinto palo)
mixed_pair = [("8", "♠"), ("8", "♥")]
reward_mixed = env._calculate_side_bet(mixed_pair)
print("Caso 2 - Par mixto:", reward_mixed)  # Esperado: 10

# --- Caso 3: Sin par (valores distintos)
no_pair = [("8", "♠"), ("9", "♠")]
reward_none = env._calculate_side_bet(no_pair)
print("Caso 3 - Sin par:", reward_none)  # Esperado: 0

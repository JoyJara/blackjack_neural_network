from blackjack_env import BlackjackEnv

# Creamos el entorno con el casino tipo 1
env = BlackjackEnv(casino_type=1)

# --- Caso 1: Total = 13  → paga 10:1
hand_equal_13 = [("8", "♠"), ("5", "♥")]  # 8 + 5 = 13
reward_equal_13 = env._calculate_side_bet(hand_equal_13)
print("Caso 1 - Total = 13:", reward_equal_13)  # Esperado: 10

# --- Caso 2: Total < 13  → paga 1:1
hand_less_13 = [("4", "♠"), ("8", "♦")]  # 4 + 8 = 12
reward_less_13 = env._calculate_side_bet(hand_less_13)
print("Caso 2 - Total < 13:", reward_less_13)  # Esperado: 1

# --- Caso 3: Total > 13  → paga 1:1
hand_greater_13 = [("9", "♠"), ("7", "♥")]  # 9 + 7 = 16
reward_greater_13 = env._calculate_side_bet(hand_greater_13)
print("Caso 3 - Total > 13:", reward_greater_13)  # Esperado: 1

# --- Caso 4: Total diferente (sólo para verificar 0 en errores)
hand_random = [("A", "♣"), ("A", "♦")]  # 2 → < 13 = 1
reward_random = env._calculate_side_bet(hand_random)
print("Caso 4 - Par de Ases (< 13):", reward_random)  # Esperado: 1

import random
from game_env import GameEnv

if __name__ == "__main__":
    env = GameEnv()

    # Inicia una nueva partida
    state = env.reset()
    print("Estado inicial:", state)
    print("Cartas jugador:", env.player)
    print("Carta visible del dealer:", env.dealer[0])

    done = False
    total_reward = 0

    # Simula una partida con acciones aleatorias
    while not done:
        action = random.choice([0, 1])  # 0 = plantarse, 1 = pedir
        new_state, reward, done = env.step(action)
        total_reward += reward
        print(f"Acción: {'Pedir' if action==1 else 'Plantarse'}")
        print("Nuevo estado:", new_state)
        print("Recompensa:", reward)
        print("Cartas jugador:", env.player)
        print("Cartas dealer:", env.dealer)
        print("-------------")

    print("Partida terminada. Recompensa total:", total_reward)

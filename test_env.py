from blackjack_env_1 import BlackjackEnv

def jugar_una_ronda(env):
    state = env.reset()
    done = False

    print("\n===== NUEVA RONDA =====\n")

    while not done:

        # Render del estado actual
        env.render()

        # Si el dealer ya tiene blackjack, terminamos
        if env.done and (env.reward in [-1, 0]):
            break

        try:
            action = int(input(
                "\n¿Qué deseas hacer? "
                "(Plantar = 0), (Pedir = 1), (Doblar = 2), (Dividir = 3), (Rendirse = 4): "
            ))
        except ValueError:
            print("Entrada inválida. Ingresa un número entre 0 y 4.")
            continue

        # Acción del jugador
        try:
            state, reward, done, info = env.step(action)
        except Exception as e:
            print(f"Error al procesar la acción: {e}")
            continue

    # Mostrar resultado final mostrando todas las cartas
    print("\n===== RESULTADO FINAL =====")
    env.render(reveal_dealer=True)
    print(f"\nRecompensa final (env.reward): {env.reward}")
    print("===========================\n")

def main():
    env = BlackjackEnv(casino_type=1)

    while True:
        jugar_una_ronda(env)
        again = input("¿Quieres jugar otra ronda? (s/n): ").strip().lower()
        if again != "s":
            print("Saliendo del juego. ¡Gracias por jugar!")
            break

if __name__ == "__main__":
    main()

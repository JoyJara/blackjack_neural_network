import sys
from blackjack_env_1 import BlackjackEnv

def pedir_accion():
    while True:
        try:
            accion = int(input(
                "\n¿Qué deseas hacer? "
                "(Plantar = 0), (Pedir = 1), (Doblar = 2), "
                "(Dividir = 3), (Rendirse = 4): "
            ))
            if accion in [0, 1, 2, 3, 4]:
                return accion
            print("Acción inválida. Ingresa un número entre 0 y 4.")
        except ValueError:
            print("Entrada inválida, solo números enteros.")


def main():
    print("=== TEST DEL ENTORNO DE BLACKJACK ===\n")

    env = BlackjackEnv(training_mode=False, verbose=True)

    while True:
        print("\n===========================\n")
        print("===== NUEVA RONDA =====\n")

        state = env.reset()
        env.render()

        # BUCLE PRINCIPAL DE LA RONDA
        while True:

            # 🚫 La ronda terminó (blackjack natural, dealer, bust etc.)
            if env.done:
                break

            # 🚫 La mano actual terminó (por blackjack natural, doblar, bust, etc.)
            hand = env.hands[env.current]
            if hand.done:
                break

            # Obtener acción del usuario
            action = pedir_accion()

            # Ejecutar acción
            state, reward, done, info = env.step(action)

            # Render
            env.render()

            # Si el entorno reporta "done"
            if done:
                break

        print("\n===========================")
        print(f"Recompensa final: {env.reward}")
        print("===========================\n")

        # Preguntar si quiere jugar otra
        otra = input("¿Quieres jugar otra ronda? (s/n): ").strip().lower()
        if otra != "s":
            print("\nGracias por jugar. ¡Hasta luego!")
            break


if __name__ == "__main__":
    main()

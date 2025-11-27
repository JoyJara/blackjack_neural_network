import torch
import numpy as np
from blackjack_env_1 import BlackjackEnv
from train_dqn import DQN

# ============================
#   CONFIGURACIÓN
# ============================
MODEL_PATH = "models/dqn_phase2.pth"   # Cambia según fase
AUTO_MODE = True                      # True: IA juega sola
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
#  FUNCIÓN PARA FORMATEAR TABLA Q
# ============================
def mostrar_tabla_q(q_values):
    acciones = ["Plantar (0)", "Pedir (1)", "Doblar (2)", "Dividir (3)", "Rendirse (4)"]

    print("\n=== TABLA Q — Estado actual ===")
    print("--------------------------------------")
    print(f"{'Acción':<15} | {'Q-Valor':>10}")
    print("--------------------------------------")

    for i, val in enumerate(q_values):
        print(f"{acciones[i]:<15} | {val:>10.3f}")

    print("--------------------------------------")
    mejor_accion = int(np.argmax(q_values))
    print(f"Mejor acción predicha: {acciones[mejor_accion]}\n")


# ============================
#        MAIN
# ============================
def jugar():

    print("\n=== EVALUACIÓN DEL MODELO DQN ===")
    print(f"Dispositivo: {DEVICE}")
    print(f"Modelo cargado: {MODEL_PATH}\n")

    # Cargar modelo
    modelo = DQN(5, 5).to(DEVICE)
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    modelo.eval()

    # Crear entorno real (modo humano)
    env = BlackjackEnv(training_mode=False, verbose=True)

    while True:
        print("\n===========================\n")
        print("===== NUEVA RONDA =====\n")

        estado = env.reset()
        done = False

        # Mostrar estado inicial
        env.render(reveal_dealer=False)

        while not done:

            estado_array = np.array(estado, dtype=np.float32)
            estado_tensor = torch.tensor(estado_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Calcular Q-values
            with torch.no_grad():
                q_vals = modelo(estado_tensor).cpu().numpy()[0]

            # Mostrar tabla Q
            mostrar_tabla_q(q_vals)

            # ============================
            #     IA JUEGA SOLA
            # ============================
            if AUTO_MODE:
                accion = int(np.argmax(q_vals))
                print(f"IA ejecuta acción: {accion}")
            else:
                # ============================
                #     HUMANO DECIDE
                # ============================
                print(
                    "¿Acción? (0=Plantar, 1=Pedir, 2=Doblar, 3=Dividir, 4=Rendirse): ",
                    end=""
                )
                try:
                    accion = int(input().strip())
                except:
                    print("Entrada inválida, repite.")
                    continue

                if accion not in [0, 1, 2, 3, 4]:
                    print("Acción inválida.")
                    continue

            # Ejecutar acción
            estado, reward, done, info = env.step(accion)

            # Mostrar estado después de la acción
            print("\n===== ESTADO DESPUÉS DE LA ACCIÓN =====")
            env.render(reveal_dealer=False)

            if done:
                print("\n===== RESULTADO FINAL =====")
                env.render(reveal_dealer=True)
                print(f"Recompensa final: {reward}")
                print("===========================\n")

        again = input("¿Quieres jugar otra ronda? (s/n): ").strip().lower()
        if again != "s":
            break


if __name__ == "__main__":
    jugar()

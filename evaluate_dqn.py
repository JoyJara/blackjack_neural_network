import torch
import numpy as np
import random
from blackjack_env_1 import BlackjackEnv
from train_dqn import DQN
import matplotlib.pyplot as plt


# ============================
# CONFIGURACIÓN
# ============================
MODEL_PATH = "models/dqn_phase2.pth"
NUM_EPISODES = 100000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# FUNCIÓN PARA FILTRAR ACCIONES INVALIDAS
# ============================
def filtrar_acciones_invalidas(env, q_values):

    hand = env.hands[env.current]
    valid = [True] * 5

    # doblar solo con 2 cartas y no split de A
    if len(hand.cards) != 2 or hand.is_ace_split:
        valid[2] = False

    # dividir: valor igual, 2 cartas, no exceder max splits
    if len(hand.cards) != 2:
        valid[3] = False
    else:
        c1, c2 = hand.cards
        if env._card_value(c1) != env._card_value(c2):
            valid[3] = False
    if len(env.hands) >= env.max_splits + 1:
        valid[3] = False

    # rendirse solo con 2 cartas
    if len(hand.cards) != 2:
        valid[4] = False

    # aplicar máscara
    for i, v in enumerate(valid):
        if not v:
            q_values[0][i] = -1e9

    return q_values


# ============================
# FUNCIÓN PRINCIPAL
# ============================
def evaluar_modelo():

    print("\n=== EVALUACIÓN MASIVA DEL MODELO DQN ===")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Episodios: {NUM_EPISODES}")
    print(f"Dispositivo: {DEVICE}\n")

    # cargar modelo
    modelo = DQN(5, 5).to(DEVICE)
    modelo.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    modelo.eval()

    env = BlackjackEnv(training_mode=False, verbose=False)

    wins = 0
    losses = 0
    pushes = 0

    for epi in range(1, NUM_EPISODES + 1):

        estado = env.reset()
        done = False

        while not done:

            estado_arr = np.array(estado, dtype=np.float32)
            estado_tensor = torch.tensor(estado_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                q_vals = modelo(estado_tensor).cpu()

            # FILTRAR acciones inválidas (CLAVE PARA EVITAR BLOQUEOS)
            q_vals = filtrar_acciones_invalidas(env, q_vals)

            accion = int(torch.argmax(q_vals).item())

            estado, reward, done, info = env.step(accion)

        # resultado final
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            pushes += 1

        # progreso en tiempo real
        if epi % 200 == 0:
            win_rate = wins / epi * 100
            push_rate = pushes / epi * 100
            loss_rate = losses / epi * 100

            print(
                f"Episodio {epi}/{NUM_EPISODES} — "
                f"win: {win_rate:.2f}% | "
                f"push: {push_rate:.2f}% | "
                f"loss: {loss_rate:.2f}%"
            )

    # =============================
    #     RESULTADOS FINALES
    # =============================
    win_rate = wins / NUM_EPISODES * 100
    push_rate = pushes / NUM_EPISODES * 100
    loss_rate = losses / NUM_EPISODES * 100

    print("\n===== RESULTADOS FINALES =====")
    print(f"Win rate :  {win_rate:.2f}%")
    print(f"Push rate:  {push_rate:.2f}%")
    print(f"Loss rate:  {loss_rate:.2f}%")
    print("==============================\n")

    # =============================
    #       GRÁFICO DE BARRAS
    # =============================
    categorias = ["Win", "Push", "Loss"]
    valores = [win_rate, push_rate, loss_rate]

    plt.figure(figsize=(7, 6))
    plt.bar(categorias, valores)
    plt.title("Performance del modelo DQN")
    plt.ylabel("Porcentaje (%)")
    plt.ylim(0, 100)

    for i, v in enumerate(valores):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')

    plt.show()


if __name__ == "__main__":
    evaluar_modelo()

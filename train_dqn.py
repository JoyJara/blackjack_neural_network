import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from blackjack_env import BlackjackEnv
import os


# ============================================================
#                 RED NEURONAL (DQN)
# ============================================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
#                 FUNCIÓN PARA FILTRAR ACCIONES
# ============================================================
def filtrar_acciones_invalidas(env, q_values):
    """
    Filtra acciones inválidas durante la fase 2 avanzada.

    NOTE:
    - Solo para fase 2.
    - Solo se usa cuando env.training_mode=True.
    - Si la acción es inválida, se pone en -inf para evitar que el DQN la elija.
    """

    hand = env.hands[env.current]
    validas = [True] * 5  # acciones [0,1,2,3,4]

    # Solo puedes doblar si hay exactamente 2 cartas
    if len(hand.cards) != 2 or hand.is_ace_split:
        validas[2] = False

    # Solo puedes dividir si las cartas tienen mismo valor y son exactamente 2
    if len(hand.cards) != 2:
        validas[3] = False
    else:
        c1, c2 = hand.cards
        if env._card_value(c1) != env._card_value(c2):
            validas[3] = False

    # Rendirse solo con 2 cartas
    if len(hand.cards) != 2:
        validas[4] = False

    # Split máximo
    if len(env.hands) >= env.max_splits + 1:
        validas[3] = False

    # Aplicar máscara
    for i, v in enumerate(validas):
        if not v:
            q_values[0][i] = -1e9

    return q_values


# ============================================================
#                 FUNCIÓN DE ENTRENAMIENTO (GENÉRICA)
# ============================================================
def entrenar_fase(
    nombre_fase,
    num_episodios,
    acciones_disponibles,
    usar_filtro_acciones,
    modelo,
    dispositivo,
    archivo_guardado,
    archivo_inicial=None,
):
    """
    Entrena una fase (fase 1 o fase 2).
    """

    print(f"\n==============================")
    print(f"   INICIANDO {nombre_fase}")
    print(f"==============================\n")

    # Configurar entorno
    env = BlackjackEnv(training_mode=True, verbose=False)

    # Dimensión del estado
    estado = env.reset()
    estado_dim = len(estado)

    # Crear modelo si es fase 1
    if archivo_inicial is not None:
        print(f"Cargando modelo de fase previa: {archivo_inicial}")
        modelo.load_state_dict(torch.load(archivo_inicial))

    optimizer = optim.Adam(modelo.parameters(), lr=0.0003)
    criterio = nn.MSELoss()

    memoria = deque(maxlen=50000)
    batch_size = 64
    gamma = 0.99

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.9995

    for episodio in range(1, num_episodios + 1):

        estado = env.reset()
        estado = np.array(estado, dtype=np.float32)
        done = False

        total_reward_epi = 0

        while not done:
            estado_tensor = torch.tensor(estado, dtype=torch.float32).to(dispositivo).unsqueeze(0)

            # -------------------------------
            #   EPSILON-GREEDY ACTION
            # -------------------------------
            if random.random() < epsilon:
                accion = random.choice(acciones_disponibles)
            else:
                q_values = modelo(estado_tensor)

                if usar_filtro_acciones:
                    q_values = filtrar_acciones_invalidas(env, q_values)

                accion = int(torch.argmax(q_values).item())

            # -------------------------------
            #     STEP DEL ENTORNO
            # -------------------------------
            siguiente_estado, recompensa, done, info = env.step(accion)
            siguiente_estado = np.array(siguiente_estado, dtype=np.float32)

            total_reward_epi += recompensa

            # Guardar transición
            memoria.append((estado, accion, recompensa, siguiente_estado, done))

            estado = siguiente_estado

            # -------------------------------
            #   ENTRENAMIENTO DEL DQN
            # -------------------------------
            if len(memoria) > batch_size:
                lote = random.sample(memoria, batch_size)

                estados_batch = torch.tensor([x[0] for x in lote], dtype=torch.float32).to(dispositivo)
                acciones_batch = torch.tensor([x[1] for x in lote]).long().to(dispositivo)
                recompensas_batch = torch.tensor([x[2] for x in lote], dtype=torch.float32).to(dispositivo)
                sig_estados_batch = torch.tensor([x[3] for x in lote], dtype=torch.float32).to(dispositivo)
                dones_batch = torch.tensor([x[4] for x in lote], dtype=torch.float32).to(dispositivo)

                q_valores = modelo(estados_batch).gather(1, acciones_batch.unsqueeze(1)).squeeze(1)
                q_siguientes = modelo(sig_estados_batch).max(1)[0].detach()

                objetivo = recompensas_batch + gamma * q_siguientes * (1 - dones_batch)

                loss = criterio(q_valores, objetivo)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Reducción de epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Logs
        if episodio % 5000 == 0:
            print(f"Episodio {episodio}/{num_episodios}  |  Reward: {total_reward_epi:.3f}")

        # Guardado incremental
        if episodio % 20000 == 0:
            torch.save(modelo.state_dict(), archivo_guardado.replace(".pth", f"_{episodio}.pth"))
            print(f"Guardado checkpoint: {archivo_guardado.replace('.pth', f'_{episodio}.pth')}")

    # Guardar modelo final de la fase
    torch.save(modelo.state_dict(), archivo_guardado)
    print(f"\nModelo guardado: {archivo_guardado}\n")


# ============================================================
#                        MAIN
# ============================================================
if __name__ == "__main__":

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Entrenando en:", dispositivo)

    # Crear carpeta modelos si no existe
    os.makedirs("models", exist_ok=True)

    # Crear modelo base
    modelo = DQN(input_dim=5, output_dim=5).to(dispositivo)  # 5 acciones posibles

    # ============================================================
    #                  FASE 1 → SOLO 2 ACCIONES
    # ============================================================
    entrenar_fase(
        nombre_fase="FASE 1 (plantar + pedir)",
        num_episodios=100000,
        acciones_disponibles=[0, 1],
        usar_filtro_acciones=False,      # filtrado no aplica aquí
        modelo=modelo,
        dispositivo=dispositivo,
        archivo_guardado="models/dqn_phase1.pth",
        archivo_inicial=None,
    )

    # ============================================================
    #                  FASE 2 → TODAS LAS ACCIONES
    # ============================================================
    entrenar_fase(
        nombre_fase="FASE 2 (doblar, dividir, rendirse)",
        num_episodios=150000,
        acciones_disponibles=[0, 1, 2, 3, 4],
        usar_filtro_acciones=True,       # filtrado activado después del comienzo
        modelo=modelo,
        dispositivo=dispositivo,
        archivo_guardado="models/dqn_phase2.pth",
        archivo_inicial="models/dqn_phase1.pth",
    )

    print("\nEntrenamiento completo terminado.")

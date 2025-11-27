import random
import numpy as np


class BlackjackEnv:
    def __init__(self, casino_type=1, cut_card_position=52):
        self.casino_type = casino_type
        self.cut_card_position = cut_card_position
        self.shoe = self._init_shoe()

    # ============================================================
    # ------------------  UTILIDADES DE CARTAS -------------------
    # ============================================================
    def _init_shoe(self):
        valores = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        palos = ["♠", "♥", "♦", "♣"]
        shoe = []
        for _ in range(6):
            for v in valores:
                for p in palos:
                    shoe.append((v, p))
        random.shuffle(shoe)
        return shoe

    def _draw_card(self):
        if len(self.shoe) <= self.cut_card_position:
            print("Carta de corte alcanzada. Nuevo zapato.")
            self.shoe = self._init_shoe()
        return self.shoe.pop()

    def _card_value(self, card):
        v, _ = card
        if v == "A":
            return 1
        if v in ["J", "Q", "K"]:
            return 10
        return int(v)

    def _hand_value(self, hand):
        total = sum(self._card_value(c) for c in hand)
        aces = sum(1 for c in hand if c[0] == "A")
        while aces > 0 and total + 10 <= 21:
            total += 10
            aces -= 1
        return total

    def _usable_ace(self, hand):
        total = sum(self._card_value(c) for c in hand)
        has_ace = any(c[0] == "A" for c in hand)
        return int(has_ace and total + 10 <= 21)

    # ============================================================
    # ----------- HELPERS PARA MANEJO DE ESTADOS ----------------
    # ============================================================
    def _make_safe_state(self, reward):
        """Devuelve un estado válido + termina episodio con reward dado."""
        next_state = (
            self._hand_value(self.player),
            self.dealer[0],
            self._usable_ace(self.player),
            len(self.player),
            int(self._card_value(self.dealer[0]) in [1, 10]),
        )
        self.done = True
        self.reward = reward
        return next_state, reward, True, {}

    def _state_to_array(self, state):
        ps, du, ua, nc, pb = state

        if isinstance(du, tuple):
            du_val = self._card_value(du)
        else:
            du_val = du

        return np.array([
            ps / 21,
            du_val / 10,
            float(ua),
            nc / 10,
            float(pb),
        ], dtype=np.float32)

    # ============================================================
    # ------------------- BLACKJACK CHECKS -----------------------
    # ============================================================
    def _is_blackjack(self, hand):
        if len(hand) != 2:
            return False
        v1 = self._card_value(hand[0])
        v2 = self._card_value(hand[1])
        return (v1 == 1 and v2 == 10) or (v2 == 1 and v1 == 10)

    def _calculate_side_bet(self, hand):
        c1, c2 = hand
        if self.casino_type == 1:
            total = self._card_value(c1) + self._card_value(c2)
            if total == 13:
                return 10
            return 1
        elif self.casino_type == 2:
            v1, s1 = c1
            v2, s2 = c2
            if v1 == v2:
                return 15 if s1 == s2 else 10
            return 0
        return 0

    # ============================================================
    # ---------------- MANEJO DE MULTI-MANOS (SPLITS) ------------
    # ============================================================
    def _handle_next_hand(self, reward, done):
        """Maneja el avance cuando hay splits."""
        if not (done and hasattr(self, "hands") and len(self.hands) > 1):
            return None

        last = len(self.hands) - 1

        # Aún quedan manos por jugar
        if self.current_hand_index < last:
            self.current_hand_index += 1
            self.player = self.hands[self.current_hand_index]

            next_state = (
                self._hand_value(self.player),
                self.dealer[0],
                self._usable_ace(self.player),
                len(self.player),
                int(self._card_value(self.dealer[0]) in [1, 10]),
            )
            return next_state, 0, False, {}

        # Ya se jugaron todas — ahora juega el dealer
        while self._hand_value(self.dealer) < 17:
            self.dealer.append(self._draw_card())

        dealer_total = self._hand_value(self.dealer)
        total_reward = 0

        for hand in self.hands:
            p = self._hand_value(hand)
            if p > 21:
                total_reward += -1
            elif dealer_total > 21 or p > dealer_total:
                total_reward += 1
            elif p == dealer_total:
                total_reward += 0
            else:
                total_reward += -1

        reward = total_reward / len(self.hands)

        self.current_hand_index = last
        self.player = self.hands[last]
        self.reward = reward
        self.done = True

        next_state = (
            self._hand_value(self.player),
            self.dealer[0],
            self._usable_ace(self.player),
            len(self.player),
            int(self._card_value(self.dealer[0]) in [1, 10]),
        )
        return next_state, reward, True, {}

    # ============================================================
    # ---------------------- RESET DEL JUEGO ---------------------
    # ============================================================
    def reset(self, force_new_shoe=False):
        if len(self.shoe) <= self.cut_card_position or force_new_shoe:
            self.shoe = self._init_shoe()
            print("Carta de corte, nuevo zapato a continuación.")

        self.player = [self._draw_card(), self._draw_card()]
        self.dealer = [self._draw_card(), self._draw_card()]

        ps = self._hand_value(self.player)
        du = self.dealer[0]
        ua = self._usable_ace(self.player)
        nc = len(self.player)
        pb = int(self._card_value(du) in [1, 10])

        self.done = False
        self.reward = 0
        self.hands = [self.player]
        self.current_hand_index = 0
        self.split_count = 0
        self.max_splits = 4

        self.side_bet_reward = self._calculate_side_bet(self.player)

        # --- Dealer blackjack ---
        if self._card_value(du) in [1, 10] and self._hand_value(self.dealer) == 21:
            self.done = True
            if self._is_blackjack(self.player):
                self.reward = 0
            else:
                self.reward = -1
            return ps, du, ua, nc, pb

        # --- Player blackjack ---
        if self._is_blackjack(self.player):
            self.done = True
            self.reward = 1.5
            return ps, du, ua, nc, pb

        return ps, du, ua, nc, pb

    # ============================================================
    # ------------------------- STEP -----------------------------
    # ============================================================
    def step(self, action):

        # ----------------------------------------------------------
        # PEDIR
        # ----------------------------------------------------------
        if action == 1:
            self.player.append(self._draw_card())
            ps = self._hand_value(self.player)

            if ps > 21:
                reward, done = -1, True
            else:
                reward, done = 0, False

            result = self._handle_next_hand(reward, done)
            if result:
                return result

            if done:
                self.done = True
                self.reward = reward

            return (
                ps,
                self.dealer[0],
                self._usable_ace(self.player),
                len(self.player),
                int(self._card_value(self.dealer[0]) in [1, 10]),
            ), reward, done, {}

        # ----------------------------------------------------------
        # PLANTARSE
        # ----------------------------------------------------------
        elif action == 0:
            while self._hand_value(self.dealer) < 17:
                self.dealer.append(self._draw_card())

            ps = self._hand_value(self.player)
            ds = self._hand_value(self.dealer)

            if ds > 21 or ps > ds:
                reward = 1
            elif ps == ds:
                reward = 0
            else:
                reward = -1

            done = True

            result = self._handle_next_hand(reward, done)
            if result:
                return result

            self.done = True
            self.reward = reward

            return (
                ps,
                self.dealer[0],
                self._usable_ace(self.player),
                len(self.player),
                int(self._card_value(self.dealer[0]) in [1, 10]),
            ), reward, True, {}

        # ----------------------------------------------------------
        # DOBLAR
        # ----------------------------------------------------------
        elif action == 2:
            if len(self.player) != 2:
                print("Acción inválida: no puedes doblar con más de dos cartas.")
                return self._make_safe_state(-1.0)

            self.player.append(self._draw_card())
            ps = self._hand_value(self.player)

            if ps > 21:
                reward, done = -2, True
            else:
                # Si es la última mano, el dealer juega
                if self.current_hand_index == len(self.hands) - 1:
                    while self._hand_value(self.dealer) < 17:
                        self.dealer.append(self._draw_card())
                    ds = self._hand_value(self.dealer)

                    if ds > 21 or ps > ds:
                        reward = 2
                    elif ps == ds:
                        reward = 0
                    else:
                        reward = -2
                else:
                    reward = 0

                done = True

            result = self._handle_next_hand(reward, done)
            if result:
                return result

            self.done = True
            self.reward = reward

            return (
                ps,
                self.dealer[0],
                self._usable_ace(self.player),
                len(self.player),
                int(self._card_value(self.dealer[0]) in [1, 10]),
            ), reward, True, {}

        # ----------------------------------------------------------
        # RENDIRSE
        # ----------------------------------------------------------
        elif action == 4:
            if len(self.player) != 2:
                print("Acción inválida: no puedes rendirte después de pedir carta.")
                return self._make_safe_state(-1.0)

            ps = self._hand_value(self.player)
            reward, done = -0.5, True

            result = self._handle_next_hand(reward, done)
            if result:
                return result

            self.done = True
            self.reward = reward

            return (
                ps,
                self.dealer[0],
                self._usable_ace(self.player),
                len(self.player),
                int(self._card_value(self.dealer[0]) in [1, 10]),
            ), reward, True, {}

        # ----------------------------------------------------------
        # DIVIDIR  (TOTALMENTE CORREGIDO)
        # ----------------------------------------------------------
        elif action == 3:

            # Solo se permite dividir pares por VALOR
            if len(self.player) == 2 and self.player[0][0] == self.player[1][0]:

                if self.split_count >= self.max_splits:
                    print("Ya no puedes dividir más.")
                    return self._make_safe_state(-1.0)

                self.split_count += 1

                cA = self.player[0]
                cB = self.player[1]

                h1 = [cA, self._draw_card()]
                h2 = [cB, self._draw_card()]

                # Reemplazar mano actual por la primera mitad
                self.hands[self.current_hand_index] = h1
                # Insertar la segunda mano justo después
                self.hands.insert(self.current_hand_index + 1, h2)

                self.player = self.hands[self.current_hand_index]

                reward, done = 0, False

                return (
                    self._hand_value(self.player),
                    self.dealer[0],
                    self._usable_ace(self.player),
                    len(self.player),
                    int(self._card_value(self.dealer[0]) in [1, 10]),
                ), reward, done, {}

            else:
                print("Acción inválida: solo puedes dividir cuando tienes un par.")
                return self._make_safe_state(-1.0)

        else:
            raise ValueError(
                "Acción inválida. Acciones válidas: 0=Plantarse, 1=Pedir, 2=Doblar, 3=Dividir, 4=Rendirse."
            )

    # ============================================================
    # ------------------------- RENDER ----------------------------
    # ============================================================
    def render(self, reveal_dealer=False):
        if reveal_dealer or getattr(self, "done", False):
            print(f"\nDealer: {self.dealer} (suma: {self._hand_value(self.dealer)})")
        else:
            print(f"\nDealer: [{self.dealer[0]}, ?]")

        print(f"\nJugador: {self.player} (suma: {self._hand_value(self.player)})")

        if hasattr(self, "hands") and len(self.hands) > 1:
            print(f"Mano {self.current_hand_index + 1} de {len(self.hands)}")

        if getattr(self, "split_count", 0) > 0:
            print(f"Divisiones realizadas: {self.split_count}")

        if self.done:
            print(f"Recompensa final: {self.reward}")

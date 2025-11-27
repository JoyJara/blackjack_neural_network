import random
import numpy as np


# ============================================================
#                        CLASE HAND
# ============================================================
class Hand:
    """
    Representa una mano individual del jugador.
    """

    def __init__(self, cards, is_ace_split=False):
        self.cards = cards[:]           
        self.is_ace_split = is_ace_split
        self.done = False
        self.bet = 1               # apuesta base (2 si se dobla)
        self.result = None         # +1, 0, -1, -2, +2, -0.5

        # Depuración / auditoría
        self.history_actions = []
        self.history_states = []
        self.initial_cards = cards[:]
        self.hand_id = None
        self.is_busted = False
        self.stayed = False
        self.is_doubled = False
        self.is_split_child = False



# ============================================================
#                 ENTORNO PROFESIONAL DE BLACKJACK
# ============================================================
class BlackjackEnv:
    """
    Reglas implementadas:
    - Split REAL (2ª carta solo cuando toca turno)
    - Split A–A especial
    - Dealer no juega si el jugador se pasa (muestra carta oculta)
    - Doblar correctamente (+2, -2, 0 según dealer)
    - Modo humano y modo RL
    """

    def __init__(self, casino_type=1, cut_card_position=52,
                 training_mode=False, verbose=True):

        self.casino_type = casino_type
        self.cut_card_position = cut_card_position
        self.training_mode = training_mode

        # Modo RL → silencioso
        self.verbose = verbose if not training_mode else False

        self.max_splits = 4
        self.shoe = self._init_shoe()

        self.hands = []
        self.current = 0
        self.dealer = []
        self.done = False
        self.reward = 0



    # ============================================================
    #                   INICIALIZACIÓN DEL ZAPATO
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
            if self.verbose:
                print("Nueva baraja (carta de corte).")
            self.shoe = self._init_shoe()
        return self.shoe.pop()

    def _card_value(self, card):
        v, _ = card
        if v == "A":
            return 1
        if v in ["J", "Q", "K"]:
            return 10
        return int(v)

    def _hand_value(self, cards):
        total = sum(self._card_value(c) for c in cards)
        aces = sum(1 for c in cards if c[0] == "A")

        while aces > 0 and total + 10 <= 21:
            total += 10
            aces -= 1

        return total

    def _usable_ace(self, cards):
        total = sum(self._card_value(c) for c in cards)
        return int(any(c[0] == "A" for c in cards) and total + 10 <= 21)



    # ============================================================
    #                       ESTADO PARA RL
    # ============================================================
    def _get_state(self):
        hand = self.hands[self.current]
        ps = self._hand_value(hand.cards)

        # ✔ Valor numérico de la carta visible del dealer (corregido)
        du = self._card_value(self.dealer[0])

        ua = self._usable_ace(hand.cards)
        nc = len(hand.cards)
        pb = int(du in [1, 10])  # dealer peligroso

        return (ps, du, ua, nc, pb)

    def _make_safe_state(self, reward):
        self.done = True
        self.reward = reward
        return self._get_state(), reward, True, {}



    # ============================================================
    #                           RESET
    # ============================================================
    def reset(self, force_new_shoe=False):

        if len(self.shoe) <= self.cut_card_position or force_new_shoe:
            if self.verbose:
                print("Nuevo zapato.")
            self.shoe = self._init_shoe()

        # Mano inicial
        h = Hand([self._draw_card(), self._draw_card()])

        self.hands = [h]
        self.current = 0

        # Dealer
        self.dealer = [self._draw_card(), self._draw_card()]

        self.done = False
        self.reward = 0

        # Blackjack natural del jugador
        if self._is_blackjack(h.cards):
            h.done = True
            h.result = +1.5
            self.done = True
            return self._get_state()

        # Dealer BJ natural
        if self._is_blackjack(self.dealer):
            h.done = True
            h.result = -1
            self.done = True
            return self._get_state()

        return self._get_state()



    # ============================================================
    #                    CHECK DE BLACKJACK
    # ============================================================
    def _is_blackjack(self, cards):
        if len(cards) != 2:
            return False
        v1 = self._card_value(cards[0])
        v2 = self._card_value(cards[1])
        return (v1 == 1 and v2 == 10) or (v2 == 1 and v1 == 10)



    # ============================================================
    #                ACCIÓN INVÁLIDA (MODO MIXTO)
    # ============================================================
    def _invalid_action(self):
        if self.training_mode:
            return self._handle_end_of_hand(-1)
        else:
            if self.verbose:
                print("Acción inválida. Elige otra.")
            return self._get_state(), 0, False, {}



    # ============================================================
    #                     FINALIZAR UNA MANO
    # ============================================================
    def _handle_end_of_hand(self, result):
        """
        Finaliza la mano actual y avanza,
        o si es la última, dealer juega.
        """
        hand = self.hands[self.current]
        hand.done = True
        hand.result = result

        # Más manos → avanzar
        if self.current + 1 < len(self.hands):
            self.current += 1

            next_hand = self.hands[self.current]
            if len(next_hand.cards) == 1 and not next_hand.is_ace_split:
                next_hand.cards.append(self._draw_card())

            return self._get_state(), 0, False, {}

        # Última mano → dealer juega
        return self._dealer_phase()



    # ============================================================
    #                     NUEVO BUST DEL JUGADOR
    # ============================================================
    def _player_bust_last_hand(self, reward):
        """
        Si el jugador se pasa en la última mano:
        - Dealer NO roba más cartas.
        - Solo revela su carta oculta.
        - Termina la ronda.
        """
        self.done = True
        self.reward = reward

        if self.verbose:
            print(f"Dealer: {self.dealer} (suma: {self._hand_value(self.dealer)})")

        return self._get_state(), reward, True, {}



    # ============================================================
    #                     FASE DEL DEALER FINAL
    # ============================================================
    def _dealer_phase(self):

        while self._hand_value(self.dealer) < 17:
            self.dealer.append(self._draw_card())

        dealer_total = self._hand_value(self.dealer)
        total_reward = 0

        for hand in self.hands:

            if hand.result is not None:
                total_reward += hand.result
                continue

            p = self._hand_value(hand.cards)

            if p > 21:
                total_reward += -1
            elif dealer_total > 21 or p > dealer_total:
                total_reward += +1 * hand.bet
            elif p == dealer_total:
                total_reward += 0
            else:
                total_reward += -1 * hand.bet

        self.done = True
        self.reward = total_reward

        return self._get_state(), total_reward, True, {}



    # ============================================================
    #                     SPLIT REAL DE CASINO
    # ============================================================
    def _perform_split(self, hand):
        c1, c2 = hand.cards

        # Caso especial: A–A
        if c1[0] == "A":
            h1 = Hand([c1, self._draw_card()], is_ace_split=True)
            h2 = Hand([c2, self._draw_card()], is_ace_split=True)
            h1.done = True
            h2.done = True

        else:
            h1 = Hand([c1])
            h2 = Hand([c2])
            h1.is_split_child = True
            h2.is_split_child = True

        self.hands[self.current] = h1
        self.hands.insert(self.current + 1, h2)



    # ============================================================
    #                           STEP
    # ============================================================
    def step(self, action):

        if self.done:
            return self._make_safe_state(self.reward)

        hand = self.hands[self.current]

        if hand.done:
            return self._invalid_action()

        # -----------------------------
        # PEDIR
        # -----------------------------
        if action == 1:

            if hand.is_ace_split:
                return self._invalid_action()

            hand.cards.append(self._draw_card())
            val = self._hand_value(hand.cards)

            if val > 21:
                hand.is_busted = True

                if self.current == len(self.hands) - 1:
                    return self._player_bust_last_hand(-1)

                return self._handle_end_of_hand(-1)

            return self._get_state(), 0, False, {}

        # -----------------------------
        # PLANTARSE
        # -----------------------------
        elif action == 0:

            hand.stayed = True
            hand.done = True
            hand.result = None  # dealer decidirá

            if self.current + 1 < len(self.hands):
                self.current += 1

                next_hand = self.hands[self.current]
                if len(next_hand.cards) == 1 and not next_hand.is_ace_split:
                    next_hand.cards.append(self._draw_card())

                return self._get_state(), 0, False, {}

            return self._dealer_phase()

        # -----------------------------
        # DOBLAR (CORREGIDO)
        # -----------------------------
        elif action == 2:

            if len(hand.cards) != 2:
                return self._invalid_action()

            if hand.is_ace_split:
                return self._invalid_action()

            hand.is_doubled = True
            hand.bet = 2

            hand.cards.append(self._draw_card())
            val = self._hand_value(hand.cards)

            if val > 21:

                if self.current == len(self.hands) - 1:
                    return self._player_bust_last_hand(-2)

                return self._handle_end_of_hand(-2)

            hand.done = True
            hand.result = None

            if self.current + 1 < len(self.hands):
                self.current += 1

                next_hand = self.hands[self.current]
                if len(next_hand.cards) == 1 and not next_hand.is_ace_split:
                    next_hand.cards.append(self._draw_card())

                return self._get_state(), 0, False, {}

            return self._dealer_phase()

        # -----------------------------
        # RENDIRSE (CORREGIDO)
        # -----------------------------
        elif action == 4:

            if len(hand.cards) != 2:
                return self._invalid_action()

            hand.done = True
            hand.result = -0.5

            if self.current + 1 < len(self.hands):
                self.current += 1

                next_hand = self.hands[self.current]
                if len(next_hand.cards) == 1 and not next_hand.is_ace_split:
                    next_hand.cards.append(self._draw_card())

                return self._get_state(), 0, False, {}

            self.done = True
            self.reward = -0.5
            return self._get_state(), -0.5, True, {}

        # -----------------------------
        # DIVIDIR
        # -----------------------------
        elif action == 3:

            if len(hand.cards) != 2:
                return self._invalid_action()

            c1, c2 = hand.cards
            if self._card_value(c1) != self._card_value(c2):
                return self._invalid_action()

            if len(self.hands) >= self.max_splits + 1:
                return self._invalid_action()

            self._perform_split(hand)

            new_hand = self.hands[self.current]
            if not new_hand.is_ace_split and len(new_hand.cards) == 1:
                new_hand.cards.append(self._draw_card())

            return self._get_state(), 0, False, {}

        # -----------------------------
        # ACCIÓN DESCONOCIDA
        # -----------------------------
        else:
            return self._invalid_action()



    # ============================================================
    #                           RENDER
    # ============================================================
    def render(self, reveal_dealer=False):

        if not self.verbose:
            return

        print("\n===== ESTADO ACTUAL =====")

        if reveal_dealer or self.done:
            print(f"Dealer: {self.dealer} (suma: {self._hand_value(self.dealer)})")
        else:
            print(f"Dealer: [{self.dealer[0]}, ?]")

        for i, hand in enumerate(self.hands):
            marcador = "<--" if i == self.current and not hand.done and not self.done else ""
            print(f"Mano {i+1}/{len(self.hands)}: {hand.cards} "
                  f"(suma: {self._hand_value(hand.cards)}) {marcador}")

        if self.done:
            print(f"Recompensa total: {self.reward}")

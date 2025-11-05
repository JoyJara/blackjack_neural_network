import random
import numpy as np

class BlackjackEnv:
    def __init__(self, casino_type=1, cut_card_position=52):
        self.casino_type = casino_type
        self.cut_card_position = cut_card_position
        self.shoe = self._init_shoe()

    def _init_shoe(self):
        # Crea un zapato realista de 6 mazos (312 cartas)
        valores = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
        palos = ["♠", "♥", "♦", "♣"]
        shoe = []

        for _ in range(6):  # 6 mazos
            for valor in valores:
                for palo in palos:
                    shoe.append((valor, palo))

        random.shuffle(shoe)
        return shoe

    def _draw_card(self):
        if len(self.shoe) <= self.cut_card_position:
            print("Se llego al límite del zapato.")
            self.shoe = self._init_shoe()
        return self.shoe.pop()

    def _card_value(self, card):
        valor, palo = card
        if valor == "A":
            return 1
        elif valor in ["J", "Q", "K"]:
            return 10
        else:
            return int(valor)

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
        return int(1 in hand and total + 10 <= 21)

    def _handle_next_hand(self, reward, done):
        if done and hasattr(self, "hands") and len(self.hands) > 1:
            self.current_hand_index +=1

            # Checa faltan manos por jugar
            if self.current_hand_index < len(self.hands):
                self.player = self.hands[self.current_hand_index]
                done = False
                reward = 0
                next_state = (
                    self._hand_value(self.player),
                    self.dealer[0],
                    self._usable_ace(self.player),
                    len(self.player),
                    int(self.dealer[0] in [1, 10])
                )
                return next_state, reward, done, {}

            # Si ya se jugaron todas las manos del jugador, continua el dealer
            while self._hand_value(self.dealer) < 17:
                self.dealer.append(self._draw_card())

            total_reward = 0
            for hand in self.hands:
                player_sum = self._hand_value(hand)
                dealer_sum = self._hand_value(self.dealer)
                if player_sum > 21:
                    total_reward += -1
                elif dealer_sum > 21 or player_sum > dealer_sum:
                    total_reward += 1
                elif player_sum == dealer_sum:
                    total_reward += 0
                else:
                    total_reward += -1

            reward = total_reward / len(self.hands)

            done = True
            next_state = (
                self._hand_value(self.player),
                self.dealer[0],
                self._usable_ace(self.player),
                len(self.player),
                int(self.dealer[0] in [1, 10])
            )

            if hasattr(self, "side_bet_reward"):
                reward += self.side_bet_reward
                self.side_bet_reward = 0

            return next_state, reward, done, {}
        return None

    def _is_blackjack(self, hand):
        # Devuelve true si es un Blackjack natural (A + 10) o (10 + A)
        if len(hand) != 2:
            return False
        v1 = self._card_value(hand[0])
        v2 = self._card_value(hand[1])
        return (v1 == 1 and v2 == 10) or (v2 == 1 and v1 == 10)

    def _calculate_side_bet(self, hand):
        card_1, card_2 = hand[0], hand[1]

        # Casino 1 (Macao)
        if self.casino_type == 1:
            total = self._card_value(card_1) + self._card_value(card_2)
            if total == 13:
                return 10
            elif total < 13 or total > 13:
                return 1
            else:
                return 0
        # Casino 2 (Jubilee)
        elif self.casino_type == 2:
            v1, s1 = card_1
            v2, s2 = card_2
            if v1 == v2:
                if s1 == s2:
                    return 15   # Par perfecto
                else:
                    return 10
            return 0

    def _state_to_array(self, state):
        # Transforma los estados del entorno a vectores, para poder usar directamente en pytorch
        player_sum, dealer_upcard, usable_ace, num_cards, possible_blackjack = state

        if isinstance(dealer_upcard, tuple):
            dealer_upcard_value = self._card_value(dealer_upcard)
        else:
            dealer_upcard_value = dealer_upcard

        # Normalización entre 0 - 1
        return np.array([
            player_sum / 21,
            dealer_upcard_value / 10,
            float(usable_ace),
            num_cards / 10,
            float(possible_blackjack)
        ], dtype=np.float32)

    def reset(self, force_new_shoe=False):
        if len(self.shoe) <= self.cut_card_position or force_new_shoe:
            self.shoe = self._init_shoe()
            print("Carta de corte, nuevo zapato a continuación.")

        # Repartir cartas al jugador
        self.player = [self._draw_card(), self._draw_card()]
        # Repartir cartas al crupier
        self.dealer = [self._draw_card(), self._draw_card()]

        player_sum = self._hand_value(self.player)
        dealer_upcard = self.dealer[0]
        player_usable_ace = self._usable_ace(self.player)
        player_num_cards = len(self.player)
        possible_blackjack = int(dealer_upcard in [1, 10])

        self.dealer_has_blackjack = False
        self.player_has_blackjack = False
        self.done = False
        self.reward = 0
        self.hands = [self.player]
        self.current_hand_index = 0
        self.split_count = 0
        self.max_splits = 4
        self.side_bet_reward = self._calculate_side_bet(self.player)

        # Checa si el dealer tiene blackjack [A,10]
        if dealer_upcard in [1, 10]:
            if self._hand_value(self.dealer) == 21:
                self.dealer_has_blackjack = True
                self.done = True

                if self._is_blackjack(self.player):
                    self.player_has_blackjack = True
                    self.reward = 0
                else:
                    self.reward = -1
                return(
                    player_sum,
                    dealer_upcard,
                    player_usable_ace,
                    player_num_cards,
                    possible_blackjack
                )
        # Si el jugador tiene blackjack natural entonces
        if self._is_blackjack(self.player):
            self.player_has_blackjack = True
            self.done = True
            self.reward = 1.5
            return(
                player_sum,
                dealer_upcard,
                player_usable_ace,
                player_num_cards,
                possible_blackjack
            )
        # Ronda normal (sin blackjack)
        state = (
            player_sum,
            dealer_upcard,
            player_usable_ace,
            player_num_cards,
            possible_blackjack
        )
        return state


    def step(self, action):
        # Pedir
        if action == 1:
            self.player.append(self._draw_card())
            player_sum = self._hand_value(self.player)

            if player_sum > 21:
                reward = -1
                done = True
            else:
                reward = 0
                done = False
            
            next_state = (
                player_sum,
                self.dealer[0],
                self._usable_ace(self.player),
                len(self.player),
                int(self.dealer[0] in [1, 10])
            )
            
            result = self._handle_next_hand(reward, done)
            if result:
                return result

            return next_state, reward, done, {}
        # Plantarse
        elif action == 0:
            # El dealer reparte sus cartas
            while self._hand_value(self.dealer) < 17:
                self.dealer.append(self._draw_card())

            player_sum = self._hand_value(self.player)
            dealer_sum = self._hand_value(self.dealer)

            if dealer_sum > 21 or player_sum > dealer_sum:
                reward = 1      # Gana el jugador
            elif player_sum == dealer_sum:
                reward = 0      # Empate
            else:
                reward = -1     # Pierde el jugador
            
            done = True
            next_state = (
                player_sum,
                self.dealer[0],
                self._usable_ace(self.player),
                len(self.player),
                int(self.dealer[0] in [1, 10])
            )

            result = self._handle_next_hand(reward, done)
            if result:
                return result

            return next_state, reward, done, {}
        # Doblar
        elif action == 2:
            if len(self.player) == 2:
                self.player.append(self._draw_card())
                player_sum = self._hand_value(self.player)

                if player_sum > 21:
                    reward = -2
                    done = True
                else:
                    # El dealer no juega si todavía hay manos jugables (por haber dividido)
                    # Si, sí es la última mano disponible, el dealer juega
                    if (
                        not hasattr(self, "hands")
                        or len(self.hands) == 1
                        or self.current_hand_index == len(self.hands) -1
                    ):
                        while self._hand_value(self.dealer) < 17:
                            self.dealer.append(self._draw_card())
                        
                        dealer_sum = self._hand_value(self.dealer)
                        if dealer_sum > 21 or player_sum > dealer_sum:
                            reward = 2
                        elif player_sum == dealer_sum:
                            reward = 0
                        else:
                            reward = - 2
                    else:
                        reward = 0

                    done = True

                next_state = (
                    player_sum,
                    self.dealer[0],
                    self._usable_ace(self.player),
                    len(self.player),
                    int(self.dealer[0] in [1, 10])
                )
                
                result = self._handle_next_hand(reward, done)
                if result:
                    return result

                return next_state, reward, done, {}
            else:
                # Aquí penaliza si intenta doblar con más de dos cartas
                safe_state = (
                    self._hand_value(self.player),
                    self.dealer[0],
                    self._usable_ace(self.player),
                    len(self.player),
                    int(self.dealer[0] in [1, 10])
                )
                print("Acción inválida, no puedes doblar con más de dos cartas.")
                return safe_state, -1.0, True, {}
        # Rendirse
        elif action == 4:
            if len(self.player) == 2:
                player_sum = self._hand_value(self.player)
                reward = -0.5
                done = True
                next_state = (
                    player_sum,
                    self.dealer[0],
                    self._usable_ace(self.player),
                    len(self.player),
                    int(self.dealer[0] in [1, 10])
                )
                
                result = self._handle_next_hand(reward, done)
                if result:
                    return result

                return next_state, reward, done, {}
            else:
                # Acción inválida: no puedes rendirte con más de dos cartas
                safe_state = (
                    self._hand_value(self.player),
                    self.dealer[0],
                    self._usable_ace(self.player),
                    len(self.player),
                    int(self.dealer[0] in [1, 10])
                )
                print("⚠️ Acción inválida: no puedes rendirte después de pedir carta.")
                return safe_state, -1.0, True, {}
        # Dividir
        elif action == 3:
            if len(self.player) == 2 and self.player[0] == self.player[1]:
                if self.split_count < self.max_splits:
                    self.split_count += 1
                    card_value = self.player[0]

                    hand_1 = [card_value, self._draw_card()]
                    hand_2 = [card_value, self._draw_card()]

                    # Reemplaza la mano inicial por las dos nuevas
                    self.hands.pop(self.current_hand_index)
                    self.hands[self.current_hand_index:self.current_hand_index] = [hand_1, hand_2]

                    self.player = self.hands[self.current_hand_index]
                    done = False
                    reward = 0
                    next_state = (
                        self._hand_value(self.player),
                        self.dealer[0],
                        self._usable_ace(self.player),
                        len(self.player),
                        int(self.dealer[0] in [1, 10]),
                    )
                    return next_state, reward, done, {}
                else:
                    raise ValueError("Ya no puedes dividir más.")
        else:
            raise ValueError("Acción inválida. (Plantarse = 0), (Pedir = 1), (Doblar = 2) ")
        
        # Si una acción del entorno no se puede ejecutar, entonces devuelve esto
        safe_state = (
            self._hand_value(self.player),
            self.dealer[0],
            self._usable_ace(self.player),
            len(self.player),
            int(self.dealer[0] in [1,10]),
        )
        return safe_state, 0, True, {}

    def render(self, reveal_dealer=False):
        if reveal_dealer or self.done:
            print(f"\nDealer: {self.dealer} (suma: {self._hand_value(self.dealer)})")
        else:
            print(f"\nDealer: [{self.dealer[0]}, ?]")

        print(f"\nJugador: {self.player} (suma: {self._hand_value(self.player)})")    

        # Mostrar si hay varias manos
        if hasattr(self, "hands") and len(self.hands) > 1:
            print(f"Mano {self.current_hand_index + 1} de {len(self.hands)}")

        # Mostrar el número de divisiones realizadas
        if hasattr(self, "split_count") and self.split_count > 0:
            print(f"Divisiones realizadas: {self.split_count}")

        if self.dealer_has_blackjack and self.player_has_blackjack:
            print("Ambos tienen Blackjack. Empate.")
        elif self.dealer_has_blackjack:
            print("El dealer tiene Blackjack. Suerte en la próxima.")
        elif self.player_has_blackjack:
            print("Blackjack paga 3:2. ¡Felicidades!")    

        if reveal_dealer or self.done:
            player_val = self._hand_value(self.player)
            dealer_val = self._hand_value(self.dealer)
            if not (self.dealer_has_blackjack or self.player_has_blackjack):
                if player_val > 21:
                    print("\nDemasiadas cartas. El dealer gana.")
                elif dealer_val > 21:
                    print("\n¡Buster! El jugador gana.")
                elif player_val > dealer_val:
                    print("\n¡Felicidades! El jugador gana.")
                elif player_val == dealer_val:
                    print("\nEmpate.")
                else:
                    print("\nEl dealer gana. Suerte para la próxima.")
            print(f"Recompensa final: {self.reward}")
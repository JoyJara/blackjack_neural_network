import random

class BlackjackEnv:
    def __init__(self):
        self.shoe = self._init_shoe()

    def _init_shoe(self):
        shoe = [1,2,3,4,5,6,7,8,9,10,10,10,10] * 24
        random.shuffle(shoe)
        return shoe

    def _draw_card(self):
        if len(self.shoe) == 0:
            self.shoe = self._init_shoe()
        return self.shoe.pop()

    def _hand_value(self, hand):
        total = sum(hand)
        aces = hand.count(1)

        while aces > 0 and total + 10 <= 21:
            total += 10
            aces -= 1
        return total
    
    def _usable_ace(self, hand):
        total = sum(hand)
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
            return next_state, reward, done, {}
        return None

    def reset(self):
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

        # Checa si el dealer tiene blackjack [A,10]
        if dealer_upcard in [1, 10]:
            if self._hand_value(self.dealer) == 21:
                self.dealer_has_blackjack = True
                self.done = True

                # En caso de que el jugador también tenga blackjack entonces empate
                if self._hand_value(self.player) == 21:
                    self.player_has_blackjack = True
                    self.reward = 0
                else:
                    self.reward = -1

        # Si el dealer no tiene blackjack el juego continua
        state = (player_sum, dealer_upcard, player_usable_ace, player_num_cards, possible_blackjack)
        return state

        # Crea y regresa el estado inicial
        state = (player_sum, dealer_upcard, player_usable_ace, player_num_cards)
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
            elif player_sum == player_sum:
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
                raise ValueError("Solo puedes doblar con las dos primeras cartas.")
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
                raise ValueError("Solo puedes rendirte con las primeras dos cartas.")
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

    def render(self, reveal_dealer=False):
        if reveal_dealer:
            print(f"\nDealer: {self.dealer} (suma: {self._hand_value(self.dealer)})")
        else:
            print(f"\nDealer: [{self.dealer[0]}, ?]")

        print(f"\nJugador: {self.player} (suma: {self._hand_value(self.player)})")    

        if reveal_dealer:
            player_val = self._hand_value(self.player)
            dealer_val = self._hand_value(self.dealer)
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
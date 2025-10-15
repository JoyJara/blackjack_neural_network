import random

class GameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Aquí se genera el mazo y se barajean las cartas.
        self.deck = [1,2,3,4,5,6,7,8,9,10,10,10,10]*4
        random.shuffle(self.deck)

        # Repartición de las primeras cartas.
        self.player = [self.deck.pop(), self.deck.pop()]
        self.dealer = [self.deck.pop(), self.deck.pop()]
        self.done = False
        return self._get_state()

    # Reparte la última carta del mazo.
    def draw_card(self):
        return self.deck.pop()

    def _get_state(self):
        return (sum(self.player), self.dealer[0])

    # Representa una paso dentro del juego (una jugada).
    def step(self, action):
        if self.done:   # Checa si finalizo la ronda.
            raise ValueError("Ronda finalizada.")

        if action == 1: # Pedir
            self.player.append(self.draw_card())
            if sum(self.player) > 21:
                self.done = True
                reward = -1
            else:
                reward = 0
        else: # Plantarse
            while sum(self.dealer) < 17:
                self.dealer.append(self.draw_card())

            self.done = True
            if sum(self.dealer) > 21 or sum(self.player) > sum(self.dealer):
                reward = +1
            elif sum(self.player) == sum(self.dealer):
                reward = 0
            else:
                reward = -1
        
        # Resultados
        return self._get_state(), reward, self.done

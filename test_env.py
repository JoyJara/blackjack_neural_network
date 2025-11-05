from blackjack_env import BlackjackEnv

env = BlackjackEnv()
state = env.reset()
done = False

while not done:
    env.render()
    if env.dealer_has_blackjack:
        print("El dealer tiene blackjack.")
        done = True
    else:
        action = int(input("\n¿Que deseas hacer? (Plantar = 0), (Pedir = 1), (Doblar = 2), (Dividir = 3), (Rendirse = 4): "))
        state, reward, done, info = env.step(action)

env.render(reveal_dealer=True)
print(f"\nRecompensa final: {reward}")
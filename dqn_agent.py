import torch
import Flappy_env
import pygame

from train import QNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

qnet = QNet().to(device)
qnet.load_state_dict(torch.load("flappy_dqn.pth", map_location=device))
qnet.eval()

Flappy_env.RENDER = True
state = Flappy_env.reset_env()
done = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    with torch.no_grad():
        q = qnet(state.unsqueeze(0).to(device))
        action = int(torch.argmax(q, dim=1).item())

    state, reward, done, info = Flappy_env.step(action)

    # render
    Flappy_env.draw()
    pygame.display.update()
    Flappy_env.clock.tick(Flappy_env.FPS)

    if done:
        state = Flappy_env.reset_env()
        done = False
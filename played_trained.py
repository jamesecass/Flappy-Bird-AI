# play_trained.py
# Loads flappy_dqn.pth and plays Flappy Bird using the trained network

import pygame
import torch
import torch.nn as nn

import Flappy_env  # must match your env filename


class QNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Flappy_env.RENDER = True
    Flappy_env.init_pygame()

    qnet = QNet().to(device)
    qnet.load_state_dict(torch.load("flappy_dqn.pth", map_location=device))
    qnet.eval()

    Flappy_env.reset_env()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        state = Flappy_env.get_state().to(device)

        with torch.no_grad():
            q = qnet(state.unsqueeze(0))
            action = int(torch.argmax(q, dim=1).item())

        _, _, done, info = Flappy_env.step(action)

        # Watch at human speed
        Flappy_env.render_frame(limit_fps=True)

        if done:
            Flappy_env.reset_env()


if __name__ == "__main__":
    main()
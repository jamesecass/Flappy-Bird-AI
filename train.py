import os
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import Flappy_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        # store CPU tensors (cheap, stable)
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        # stack on CPU; move to GPU in train_step
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(ns),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def select_action(qnet, state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 1)
    with torch.no_grad():
        q = qnet(state.unsqueeze(0).to(device))
        return int(q.argmax(dim=1).item())


@torch.no_grad()
def soft_update(target, online, tau=0.005):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


def train_step_double_dqn(qnet, target, buffer, optimizer, batch_size=128, gamma=0.99):
    if len(buffer) < batch_size:
        return None

    s, a, r, ns, done = buffer.sample(batch_size)

    s = s.to(device)
    ns = ns.to(device)
    a = a.to(device)
    r = r.to(device)
    done = done.to(device)

    # Q(s,a)
    q_sa = qnet(s).gather(1, a.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        # Double DQN:
        # choose a* using online net, evaluate using target net
        next_a = qnet(ns).argmax(dim=1)  # [B]
        next_q = target(ns).gather(1, next_a.view(-1, 1)).squeeze(1)
        target_q = r + gamma * next_q * (1.0 - done)

    loss = nn.SmoothL1Loss()(q_sa, target_q)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
    optimizer.step()

    return float(loss.item())


def save_checkpoint(path, qnet, target, optimizer, epsilon, steps):
    torch.save(
        {
            "qnet": qnet.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epsilon": float(epsilon),
            "steps": int(steps),
        },
        path,
    )


def load_checkpoint(path, qnet, target, optimizer):
    ckpt = torch.load(path, map_location=device)
    qnet.load_state_dict(ckpt["qnet"])
    target.load_state_dict(ckpt["target"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return float(ckpt.get("epsilon", 1.0)), int(ckpt.get("steps", 0))


def main():
    # IMPORTANT for speed: render off while training
    Flappy_env.RENDER = False

    qnet = QNet().to(device)
    target = QNet().to(device)
    target.load_state_dict(qnet.state_dict())
    optimizer = optim.Adam(qnet.parameters(), lr=3e-4)
    buffer = ReplayBuffer(capacity=100_000)

    gamma = 0.99
    batch_size = 128
    tau = 0.005

    # Exploration: keep it higher for longer (your old schedule hit min too early)
    epsilon = 1.0
    epsilon_min = 0.10
    epsilon_decay = 0.9995

    steps = 0
    episodes = 10_000

    warmup_steps = 5000  # collect experiences before training hard

    if os.path.exists("checkpoint.pth"):
        epsilon, steps = load_checkpoint("checkpoint.pth", qnet, target, optimizer)
        epsilon = max(epsilon, 0.2)
        print(f"Resumed | eps={epsilon:.3f} | steps={steps}")

    print_every_eps = 10
    save_every_eps = 100

    for ep in range(1, episodes + 1):
        state = Flappy_env.reset_env()  # CPU tensor
        done = False
        ep_reward = 0.0
        last_loss = None
        last_info = {"score": 0.0}

        while not done:
            steps += 1
            action = select_action(qnet, state, epsilon)

            next_state, reward, done, info = Flappy_env.step(action)

            buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            last_info = info

            # train after warmup
            if steps > warmup_steps:
                loss = train_step_double_dqn(qnet, target, buffer, optimizer, batch_size, gamma)
                if loss is not None:
                    last_loss = loss
                soft_update(target, qnet, tau=tau)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if ep % print_every_eps == 0:
            print(
                f"Episode {ep:4d} | score={last_info.get('score'):5.1f} | "
                f"ep_reward={ep_reward:8.2f} | eps={epsilon:.3f} | loss={last_loss}"
            )

        if ep % save_every_eps == 0:
            torch.save(qnet.state_dict(), "flappy_dqn.pth")
            save_checkpoint("checkpoint.pth", qnet, target, optimizer, epsilon, steps)
            print("Saved flappy_dqn.pth + checkpoint.pth")

    torch.save(qnet.state_dict(), "flappy_dqn.pth")
    save_checkpoint("checkpoint.pth", qnet, target, optimizer, epsilon, steps)
    print("Saved final flappy_dqn.pth + checkpoint.pth")


if __name__ == "__main__":
    main()
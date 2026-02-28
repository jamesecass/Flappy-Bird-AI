# Flappy_env.py
# A training-ready Flappy Bird environment for DQN:
# - reset_env() -> state tensor
# - step(action) -> next_state, reward, done, info
# - optional rendering: set RENDER = True and call init_pygame() once

import pygame
from sys import exit
import random
import torch

# -----------------------------
# Config
# -----------------------------
GAME_WIDTH = 360
GAME_HEIGHT = 640
FPS = 60

RENDER = False  # set True when you want to watch

bird_x = GAME_WIDTH // 8
bird_y = GAME_HEIGHT // 2
bird_width = 34
bird_height = 24

pipe_x = GAME_WIDTH
pipe_y = 0
pipe_width = 64
pipe_height = 512

velocity_x = -2
gravity = 0.4
FLAP_VELOCITY = -6

PIPE_SPAWN_FRAMES = int(FPS * 1.5)  # ~1.5 seconds
PIPE_SPAWN_X = int(GAME_WIDTH * 0.75)  # spawn closer so learning starts sooner

# Reward shaping (helps learning early)
ALIVE_REWARD = 0.0
PASS_REWARD = 1
DEATH_PENALTY = -1
SHAPING = True  # set False if you want pure sparse reward

# -----------------------------
# Pygame globals (initialised only if rendering)
# -----------------------------
window = None
clock = None

# -----------------------------
# Sprites / Rects
# -----------------------------
class Bird(pygame.Rect):
    def __init__(self, img):
        super().__init__(bird_x, bird_y, bird_width, bird_height)
        self.img = img


class Pipe(pygame.Rect):
    def __init__(self, img):
        super().__init__(pipe_x, pipe_y, pipe_width, pipe_height)
        self.img = img
        self.passed = False


# -----------------------------
# Load images (paths: change if needed)
# -----------------------------
# NOTE: these load at import time. If you ever want "headless" training without images,
# we can refactor, but this is simplest for now.
BACKGROUND_PATH = r"flappybirdbg.png"
BIRD_PATH = r"flappybird.png"
TOP_PIPE_PATH = r"toppipe.png"
BOTTOM_PIPE_PATH = r"bottompipe.png"

background_image = pygame.image.load(BACKGROUND_PATH)
bird_image = pygame.image.load(BIRD_PATH)
bird_image = pygame.transform.scale(bird_image, (bird_width, bird_height))

top_pipe_image = pygame.image.load(TOP_PIPE_PATH)
top_pipe_image = pygame.transform.scale(top_pipe_image, (pipe_width, pipe_height))

bottom_pipe_image = pygame.image.load(BOTTOM_PIPE_PATH)
bottom_pipe_image = pygame.transform.scale(bottom_pipe_image, (pipe_width, pipe_height))

# -----------------------------
# Environment state
# -----------------------------
bird = Bird(bird_image)
pipes = []
velocity_y = 0.0
score = 0.0
game_over = False
frames_since_pipe = 0


def init_pygame():
    """Call this once if you want to render."""
    global window, clock
    if window is not None:
        return
    pygame.init()
    window = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))
    pygame.display.set_caption("Flappy Bird (RL)")
    clock = pygame.time.Clock()


def draw():
    """Draw current frame (only works if init_pygame() was called)."""
    if window is None:
        return
    window.blit(background_image, (0, 0))
    window.blit(bird.img, bird)

    for p in pipes:
        window.blit(p.img, p)

    text_font = pygame.font.SysFont("Mono", 45)
    text_str = str(int(score))
    if game_over:
        text_str = "Game Over: " + text_str
    text_render = text_font.render(text_str, True, "white")
    window.blit(text_render, (5, 0))


def render_frame(limit_fps: bool = False):
    """Convenience: draw + update display. Optionally cap FPS."""
    if not RENDER:
        return
    if window is None:
        init_pygame()
    draw()
    pygame.display.update()
    if limit_fps:
        clock.tick(FPS)


def create_pipes():
    """Create a top+bottom pipe pair with a gap."""
    # Randomize vertical position (same as your logic)
    random_pipe_y = pipe_y - pipe_height / 4 - random.random() * (pipe_height / 2)
    opening_space = GAME_HEIGHT / 4

    top_pipe = Pipe(top_pipe_image)
    top_pipe.x = PIPE_SPAWN_X
    top_pipe.y = random_pipe_y
    pipes.append(top_pipe)

    bottom_pipe = Pipe(bottom_pipe_image)
    bottom_pipe.x = PIPE_SPAWN_X
    bottom_pipe.y = top_pipe.y + top_pipe.height + opening_space
    pipes.append(bottom_pipe)


def get_next_pipe_pair():
    """Return (top, bottom) pipe pair that is in front of the bird."""
    for i in range(0, len(pipes), 2):
        if i + 1 >= len(pipes):
            continue
        top = pipes[i]
        bottom = pipes[i + 1]
        if top.x + top.width >= bird.x:
            return top, bottom
    return None, None


def get_state():
    """State vector: [bird_y, velocity_y, pipe_dx, gap_center_y] (all normalized)."""
    top, bottom = get_next_pipe_pair()

    if top is None:
        pipe_dx = 1.0
        gap_center_y = 0.5
    else:
        pipe_dx = (top.x + top.width - bird.x) / GAME_WIDTH
        pipe_dx = max(0.0, min(pipe_dx, 1.0))

        gap_center = (top.y + top.height + bottom.y) / 2
        gap_center_y = gap_center / GAME_HEIGHT
        gap_center_y = max(0.0, min(gap_center_y, 1.0))

    return torch.tensor(
        [
            bird.y / GAME_HEIGHT,
            velocity_y / 10.0,
            pipe_dx,
            gap_center_y,
        ],
        dtype=torch.float32,
    )


def move_one_frame():
    """Physics + collisions + scoring. Updates globals."""
    global velocity_y, score, game_over

    velocity_y += gravity
    bird.y += velocity_y
    bird.y = max(bird.y, 0)

    # Ground collision
    if bird.y + bird.height >= GAME_HEIGHT:
        game_over = True
        return

    # Move pipes and check collisions
    for p in pipes:
        p.x += velocity_x

        if (not p.passed) and (bird.x > p.x + p.width):
            score += 0.5
            p.passed = True

        if bird.colliderect(p):
            game_over = True
            return

    # Remove offscreen pipes
    while len(pipes) > 0 and pipes[0].x < -pipe_width:
        pipes.pop(0)


def reset_env():
    """Start a new episode."""
    global velocity_y, score, game_over, frames_since_pipe

    velocity_y = 0.0
    score = 0.0
    game_over = False
    frames_since_pipe = 0

    bird.y = bird_y
    pipes.clear()

    create_pipes()  # immediate target
    return get_state()


def step(action: int):
    """
    action: 0 = do nothing, 1 = flap
    returns: next_state, reward, done, info
    """
    global velocity_y, game_over, frames_since_pipe

    prev_score = score

    # Apply action
    if action == 1 and not game_over:
        velocity_y = FLAP_VELOCITY

    # Spawn pipes by frames (deterministic for RL)
    if not game_over:
        frames_since_pipe += 1
        if frames_since_pipe >= PIPE_SPAWN_FRAMES:
            create_pipes()
            frames_since_pipe = 0

        move_one_frame()

    # Reward
    reward = ALIVE_REWARD

    if game_over:
        reward = DEATH_PENALTY
    if score > prev_score:
     print("PASSED! score=", score)

    # Optional shaping: reward being near the gap center
    if SHAPING and (not game_over):
        top, bottom = get_next_pipe_pair()
        if top is not None:
            gap_center = (top.y + top.height + bottom.y) / 2
            dist = abs(bird.y - gap_center) / GAME_HEIGHT  # 0 = perfect
            reward += (0.5 - dist) * 0.05  # small guidance

    return get_state(), float(reward), bool(game_over), {"score": float(score)}


def heuristic_action():
    """A strong baseline bot (not used by DQN)."""
    top, bottom = get_next_pipe_pair()
    if top is None:
        return 0
    gap_center = (top.y + top.height + bottom.y) / 2
    margin = 25
    return 1 if bird.y > gap_center + margin else 0


if __name__ == "__main__":
    # Demo: watch heuristic play
    RENDER = True
    init_pygame()
    reset_env()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        a = heuristic_action()
        _, _, done, info = step(a)

        render_frame(limit_fps=True)

        if done:

            reset_env()

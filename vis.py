# import pygame
# import sys
# import time
# import random
# from bradleyenv import BradleyAirportEnv

# WIDTH, HEIGHT = 800, 800

# # Colors
# WHITE = (255, 255, 255)
# RED = (255, 0, 0)
# BLACK = (0, 0, 0)
# GREEN = (0, 255, 0)
# BLUE = (0, 0, 255)
# GRAY = (200, 200, 200)
# DARK_GRAY = (50, 50, 50)
# YELLOW = (255, 255, 0)

# # Setup display
# screen=None

# game_ended = False

# fps = 60
# sleeptime = 0.1
# clock = None

# # Initialize simulation
# game = BradleyAirportEnv(WIDTH, HEIGHT)

# # Initialize Pygame
# def setup(GUI=True):
#     global screen
#     if GUI:
#         pygame.init()
#         screen = pygame.display.set_mode((WIDTH, HEIGHT))
#         pygame.display.set_caption("AI ATC Simulation")

# def main():
#     global game_ended
#     clock = pygame.time.Clock()
#     running = True
#     if len(game.planes) == 0:
#         game.add_plane()

#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type == pygame.KEYDOWN:
#                 if event.key in [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4]:
#                     selected_plane_index = int(event.unicode)  
#                     print(f"Selected plane: {selected_plane_index}")
#                 elif not game_ended:
#                     action = None
#                     if event.key == pygame.K_w:
#                         action = 2
#                     if event.key == pygame.K_s:
#                         action = 3
#                     if event.key == pygame.K_a:
#                         action = 0
#                     if event.key == pygame.K_d:
#                         action = 1
#                     if event.key == pygame.K_u:
#                         action = 4
#                     if event.key == pygame.K_i:
#                         action = 5
#                     if event.key == pygame.K_o:
#                         action = 6
#                     if event.key == pygame.K_p:
#                         action = 7
#                     if event.key == pygame.K_m:
#                         for plane in game.planes:
#                             plane.move()
#                     if event.key == pygame.K_z:
#                         game.add_plane()
#                     if action is not None:
#                         print(f"Action chosen: {action}")  # ✅ Add this
#                         if selected_plane_index < len(game.planes):
#                             actions = [12] * len(game.planes)  # All go straight
#                             actions[selected_plane_index] = action
#                             game.step(actions)

#                         else:
#                             print(f"Plane index {selected_plane_index} is out of range.")
#         screen.fill(BLACK)

#         # Draw runways
#         pygame.draw.rect(screen, GRAY, (100, 200, 300, 10))  # Horizontal runway
#         pygame.draw.rect(screen, GRAY, (200, 100, 10, 300))  # Intersecting vertical runway

#         # Draw taxiway
#         pygame.draw.rect(screen, DARK_GRAY, (80, 200, 10, 150))
#         pygame.draw.rect(screen, DARK_GRAY, (200, 80, 150, 10))

#         # Draw each plane
#         for plane in game.planes:
#             pygame.draw.circle(screen, plane.color, (int(plane.x), int(plane.y)), plane.size)
            
#             # ✅ Draw direction line (10px ahead)
#             end_x = int(plane.x + 15 * plane.dx)
#             end_y = int(plane.y + 15 * plane.dy)
#             pygame.draw.line(screen, RED, (int(plane.x), int(plane.y)), (end_x, end_y), 2)

        
#         pygame.display.flip()
#         clock.tick(30)

#     pygame.quit()
#     sys.exit()


# if __name__ == "__main__":
#     setup()
#     main()

import pygame
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bradleyenv import BradleyAirportEnv
from collections import Counter

# -----------------------------
# Constants
# -----------------------------
action_counter = Counter()
WIDTH, HEIGHT = 800, 800
MAX_PLANES = 5
STATE_DIM = 6 * MAX_PLANES
ACTION_DIM = 13 * MAX_PLANES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Colors
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
OLIVE = (162, 148, 119)

# -----------------------------
# Centralized DQN (matches train.py)
# -----------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=STATE_DIM, output_dim=ACTION_DIM):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------
# Load model
# -----------------------------
env = BradleyAirportEnv(WIDTH, HEIGHT)
model = DQN(input_dim=STATE_DIM, output_dim=ACTION_DIM).to(device)
model.load_state_dict(torch.load("centralized_dqn_airport.pth", map_location=device))
model.eval()

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Centralized DQN Tower Controller")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 16)

# Add initial planes
if len(env.planes) == 0:
    env.add_plane()
for _ in range(3):  # Add more planes
    env.add_plane()

# -----------------------------
# Main loop
# -----------------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    # Draw runway & taxiways
    pygame.draw.rect(screen, GRAY, (100, 200, 300, 10))
    pygame.draw.rect(screen, GRAY, (200, 100, 10, 300))
    pygame.draw.rect(screen, DARK_GRAY, (80, 200, 10, 150))
    pygame.draw.rect(screen, DARK_GRAY, (200, 80, 150, 10))
    # Draw gates
    for gx, gy in env.gate_zones.values():
        pygame.draw.rect(screen, CYAN, pygame.Rect(gx - 10, gy - 10, 20, 20))
    # Draw entires
    for entry in env.runway_entries.values():
        pygame.draw.circle(screen, BLUE, entry, 6)
    # Draw exists
    for exit in env.runway_exits.values():
        pygame.draw.circle(screen, OLIVE, exit, 6)


    # Prepare joint state
    obs = []
    for plane in env.planes[:5]:  # Only use up to MAX_PLANES
        obs.extend(env.get_obs(plane))

    # Pad or trim to exactly STATE_DIM
    if len(obs) < STATE_DIM:
        obs += [0.0] * (STATE_DIM - len(obs))
    else:
        obs = obs[:STATE_DIM]

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    # Get joint actions
    with torch.no_grad():
        q_values = model(obs_tensor).squeeze(0)
    actions = []
    for i in range(len(env.planes)):
        action_slice = q_values[i * 13: (i + 1) * 13]
        action = torch.argmax(action_slice).item()
        actions.append(action)
    
        # Track how often each action is used
    for action in actions:
        action_counter[action] += 1

    # Print action distribution every ~3 seconds
    if pygame.time.get_ticks() % 3000 < 30:  # every ~3000 ms
        print("Action counts so far:", dict(action_counter))

    env.step(actions)

    # Draw planes
    for i, plane in enumerate(env.planes):
        pygame.draw.circle(screen, plane.color, (int(plane.x), int(plane.y)), plane.size)
        end_x = int(plane.x + 15 * plane.dx)
        end_y = int(plane.y + 15 * plane.dy)
        pygame.draw.line(screen, RED, (int(plane.x), int(plane.y)), (end_x, end_y), 2)

        # Speed info
        speed_text = font.render(f"Speed: {int(plane.speed)}", True, WHITE)
        screen.blit(speed_text, (int(plane.x + 10), int(plane.y + 10)))
        if hasattr(plane, "gate_target"):
            gx, gy = plane.gate_target
            pygame.draw.circle(screen, YELLOW, (int(gx), int(gy)), 5)  # Yellow dot = assigned gate
            gate_label = font.render(f"G{i}", True, WHITE)
            screen.blit(gate_label, (int(plane.x + 10), int(plane.y + 10)))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
sys.exit()
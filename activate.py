from bradleyenv import BradleyAirportEnv
from vis_bradleyenv import BradleyAirportGUI
import pygame
from aircraft_1 import Aircraft

env = BradleyAirportEnv()
env.planes = [Aircraft(800, 600) for _ in range(5)]
gui = BradleyAirportGUI(env)

running = True
obs = env.reset()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)

    for plane in env.planes:
        plane.move()
    env.planes = [p for p in env.planes if not p.is_off_screen()]

    if len(env.planes) < 5:
        env.planes.append(Aircraft(800, 600))

    gui.render()

    pygame.time.delay(500)  # slow down steps for visibility

    if done:
        obs = env.reset()



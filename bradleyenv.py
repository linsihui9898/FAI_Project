import gym
from gym import spaces
import numpy as np
import random
from aircraft import Aircraft
import pygame

class BradleyAirportEnv(gym.Env):
    def __init__(self, screen_width=800, screen_height=800):
        super(BradleyAirportEnv, self).__init__()

        self.num_runways = 2  
        self.num_taxiways = 1  
        self.max_aircraft = 10
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Actions: Choose a runway (0, 1), taxiway (2), or delay (3)
        self.action_space = spaces.Discrete(self.num_runways + self.num_taxiways + 1)

        # Observation Space: (traffic level, weather, runway & taxiway availability)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_runways + self.num_taxiways + 2,), dtype=np.float32)

        # Store planes in environment
        self.planes = []

        self.reset()

    def reset(self):
        self.runways = [True] * self.num_runways  # All runways available
        self.taxiway = True  # Taxiway is available
        self.traffic = 0
        self.planes = []
        self.weather = random.uniform(0, 1)  # Random weather condition
        self.time_step = 0
        return np.array(self.planes + [self.traffic, self.weather] + self.runways + [self.taxiway], dtype=np.float32)

    # Need to change actions to be specific to selected aircraft (since there are multiple planes on screen)
    # Also need to update observations
    def step(self, action):
        reward = 0
        done = False
        self.time_step += 1

        # If a runway or taxiway is selected
        if action < self.num_runways + self.num_taxiways:
            if action < self.num_runways:  # Runway selected
                if self.runways[action]:  
                    self.runways[action] = False  
                    self.traffic -= 1  
                    reward = 5  # Large reward for safe landing/takeoff
                else:
                    reward = -10  # Large penalty for collision (runway occupied)
            else:  # Taxiway selected
                if self.taxiway:
                    self.taxiway = False  
                    self.traffic -= 1  
                    reward = 5  # Successful use of taxiway
                else:
                    reward = -10  # Collision on the taxiway
        else:  
            reward = -1  # Penalty for delaying

        # Simulate new aircraft arrivals
        self.traffic += random.randint(0, 1)

        # Weather impact
        if self.weather < 0.3:
            close_choice = random.choice(["runway", "taxiway"])
            if close_choice == "runway":
                self.runways[random.randint(0, self.num_runways - 1)] = False  
            else:
                self.taxiway = False  
        else:  
            self.runways = [True] * self.num_runways  # Gradually reopen runways
            self.taxiway = True  

        # End condition
        if self.traffic <= 0 or self.time_step >= 50:
            done = True

        return np.array([self.traffic, self.weather] + self.runways + [self.taxiway], dtype=np.float32), reward, done, {}

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((500, 500))
                self.clock = pygame.time.Clock()

            self.screen.fill((0, 0, 0))  # Clear screen

            # Draw runways
            pygame.draw.rect(self.screen, (200, 200, 200), (100, 200, 300, 10))  # Horizontal runway
            pygame.draw.rect(self.screen, (200, 200, 200), (200, 100, 10, 300))  # Intersecting vertical runway

            # Draw taxiway
            pygame.draw.rect(self.screen, (100, 100, 100), (80, 200, 10, 150))  # Parallel taxiway

            # Draw each plane
            for plane in self.planes:
                pygame.draw.circle(self.screen, plane.color, (int(plane.x), int(plane.y)), plane.size)

        print(f"Time Step: {self.time_step} | Traffic: {self.traffic} | Runways: {self.runways} | Taxiway: {self.taxiway} | Weather: {self.weather}")

    # Add a new plane to the environment
    def add_plane(self):
        if self.traffic < self.max_aircraft:
            plane = Aircraft(self.screen_width, self.screen_height)
            self.planes.append(plane)

    # May need to update observations as well every time tick
    def update(self):
        for plane in self.planes:
            plane.move()



# ===========================
# Testing the Environment
# ===========================

if __name__ == "__main__":
    env = BradleyAirportEnv()
    
    num_episodes = 5
    max_steps = 20  

    for episode in range(num_episodes):
        obs = env.reset()
        print(f"\n==== Episode {episode + 1} ====")

        for step in range(max_steps):
            action = env.action_space.sample()  
            next_obs, reward, done, _ = env.step(action)

            print(f"Step {step + 1}: Action={action}, Reward={reward}, Next State={next_obs}")
            env.render()  

            if done:
                print(f"Episode {episode + 1} finished after {step + 1} steps.")
                break

    env.close()

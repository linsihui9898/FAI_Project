import gym
from gym import spaces
import numpy as np
import random

class RunwaySchedulingEnv(gym.Env):
    def __init__(self):
        super(RunwaySchedulingEnv, self).__init__()
        self.num_runways = 3
        self.max_aircraft = 10
        # 3 runways + 1 delay action
        self.action_space = spaces.Discrete(self.num_runways + 1)
        # (traffic, weather, runway0, runway1, runway2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.reset()

    def reset(self):
        # Set initial conditions
        self.runways = [True] * self.num_runways
        self.traffic = random.randint(1, self.max_aircraft)
        self.weather = random.uniform(0, 1)
        self.time_step = 0
        return np.array([self.traffic, self.weather] + [float(r) for r in self.runways], dtype=np.float32)

    def step(self, action):
        # Move time
        self.time_step += 1
        reward = 0
        done = False

        # Pick a runway or delay
        if action < self.num_runways:
            if not self.runways[action]:
                # Collision
                reward = -1000
                done = True
            else:
                # Check weather
                if self.weather < 0.3:
                    reward = -50
                else:
                    reward = 1
                # Use runway
                self.runways[action] = False
                self.traffic -= 1
        else:
            # Delay
            reward = -1

        # New arrival
        self.traffic += random.randint(0, 1)

        # Close runway if weather is bad
        if self.weather < 0.3:
            self.runways[random.randint(0, self.num_runways - 1)] = False

        # End conditions
        if self.traffic <= 0 or self.time_step >= 50:
            done = True

        obs = np.array([self.traffic, self.weather] + [float(r) for r in self.runways], dtype=np.float32)
        return obs, reward, done, {}

    def render(self):
        print(f"Time: {self.time_step}, Traffic: {self.traffic}, Runways: {self.runways}, Weather: {self.weather:.2f}")

if __name__ == "__main__":
    env = RunwaySchedulingEnv()
    for episode in range(3):
        obs = env.reset()
        print(f"\nEpisode {episode + 1}")
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            env.render()
            print(f"Action={action}, Reward={reward}, Obs={obs}, Done={done}")
            if done:
                print(f"Episode ended at step {step + 1}")
                break
    env.close()

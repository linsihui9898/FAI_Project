import gym
from gym import spaces
import numpy as np
import random

class RunwaySchedulingEnv(gym.Env):
    def __init__(self):
        super(RunwaySchedulingEnv, self).__init__()

        self.num_runways = 3  
        self.max_aircraft = 10  # Max waiting aircraft

        # Action Space: Choose a runway (0, 1, or 2) for scheduling, or delay (3)
        self.action_space = spaces.Discrete(self.num_runways + 1)  # +1 for 'delay' action

        # Observation Space: (traffic level, weather, runway availability)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_runways + 2,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.runways = [True] * self.num_runways  # All runways available
        self.traffic = random.randint(1, self.max_aircraft)  # Random aircraft count
        self.weather = random.uniform(0, 1)  # Random weather condition
        self.time_step = 0
        return np.array([self.traffic, self.weather] + self.runways, dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        self.time_step += 1

        if action < self.num_runways:  # If a runway is selected
            if self.runways[action]:  # Runway is available
                self.runways[action] = False  # Occupy the runway
                self.traffic -= 1  # Aircraft takes off/lands
                reward = 1
            else:
                reward = -2  # Penalty for choosing an occupied runway
        else:
            reward = -1  # Penalty for delaying an aircraft

        # Simulate aircraft arrival & weather impact
        self.traffic += random.randint(0, 1)  # New aircraft arrival
        if self.weather < 0.3:  # Bad weather affects scheduling
            self.runways[random.randint(0, self.num_runways - 1)] = False  # Close a random runway

        # If traffic is cleared or after 50 steps, stop the episode
        if self.traffic <= 0 or self.time_step >= 50:
            done = True

        return np.array([self.traffic, self.weather] + self.runways, dtype=np.float32), reward, done, {}

    def render(self):
        print(f"Time Step: {self.time_step} | Traffic: {self.traffic} | Runways: {self.runways} | Weather: {self.weather}")

# ===========================
# Testing the Environment
# ===========================

if __name__ == "__main__":
    env = RunwaySchedulingEnv()
    
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


# Action	                            Outcome
# Runway is available (0, 1, 2)	         Aircraft takes off (-1 traffic, +1 reward)
# Runway is occupied (0, 1, 2)	         Penalty (-2 reward, no change)
# Delay (3)	                             Penalty (-1 reward, possible traffic increase)
# Bad weather (<0.3)	                 Random runway closes
# Good weather (>0.3)	                 Runways reopen
import numpy as np
import pickle
from bradleyenv import *
from ATC_agent import ATC_agent
import matplotlib.pyplot as plt

num_episodes = 5000
max_steps = 100

# Initialize environment and agent
env = BradleyAirportEnv()
num_actions = env.action_space.n
agent = ATC_agent(num_actions)

episode_rewards = []
# Training
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        action = agent.choose(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

    print(f"Episode {episode + 1} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.2f}")

print("Training finished")

env.close()


'''
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Trends")
plt.show()
'''
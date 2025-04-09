# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import random
# from collections import deque
# from bradleyenv import BradleyAirportEnv

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # === Hyperparameters ===
# GAMMA = 0.99
# LR = 1e-4
# BATCH_SIZE = 64
# BUFFER_SIZE = 100_000
# TARGET_UPDATE_FREQ = 1000
# EPISODES = 1000
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 5000
# MAX_STEPS = 300

# MAX_PLANES = 5
# STATE_DIM = 6 * MAX_PLANES  # 6 features per plane
# ACTION_DIM = 13 * MAX_PLANES  # 13 actions per plane

# # === Replay Buffer ===
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     # In ReplayBuffer class:
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done, len(action)))  # ✅ store number of planes

#     def sample(self, batch_size):
#         samples = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done, num_planes = zip(*samples)
#         return (
#             torch.FloatTensor(np.array(state)).to(device),
#             action,
#             torch.FloatTensor(reward).to(device),
#             torch.FloatTensor(np.array(next_state)).to(device),
#             torch.FloatTensor(done).to(device),
#             num_planes
#         )


#     def __len__(self):
#         return len(self.buffer)

# # === DQN Network ===
# class DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, output_dim)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# # === DQN Agent ===
# class CentralizedDQNAgent:
#     def __init__(self):
#         self.env = BradleyAirportEnv()
#         self.policy_net = DQN(STATE_DIM, ACTION_DIM).to(device)
#         self.target_net = DQN(STATE_DIM, ACTION_DIM).to(device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
#         self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
#         self.steps_done = 0
#         self.epsilon = EPSILON_START

#     def get_flat_obs(self):
#         obs = []
#         for plane in self.env.planes:
#             obs.extend(self.env.get_obs(plane))
#         while len(obs) < STATE_DIM:  # pad if < MAX_PLANES
#             obs.extend([0.0] * 6)
#         return np.array(obs, dtype=np.float32)

#     def select_action(self, state):
#         self.epsilon = max(EPSILON_END, EPSILON_START - self.steps_done / EPSILON_DECAY)
#         self.steps_done += 1
#         if random.random() < self.epsilon:
#             actions = [random.randint(0, 12) for _ in range(len(self.env.planes))]
#         else:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 q_values = self.policy_net(state_tensor).squeeze(0)
#             actions = []
#             for i in range(len(self.env.planes)):
#                 start = i * 13
#                 end = start + 13
#                 actions.append(torch.argmax(q_values[start:end]).item())
#         return actions

#     def train_step(self):
#         if len(self.replay_buffer) < BATCH_SIZE:
#             return

#         state, action_batch, reward, next_state, done, num_planes = self.replay_buffer.sample(BATCH_SIZE)

#         q_values = self.policy_net(state)
#         next_q_values = self.target_net(next_state).detach()

#         target_q = q_values.clone()
#         for i in range(BATCH_SIZE):
#             for j in range(num_planes[i]):  # ✅ use actual number of planes in this sample
#                 idx = j * 13 + action_batch[i][j]
#                 target_q[i, idx] = reward[i] + GAMMA * next_q_values[i, idx] * (1 - done[i])

#         loss = F.mse_loss(q_values, target_q)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


#     def update_target(self):
#         self.target_net.load_state_dict(self.policy_net.state_dict())

#     def train(self):
#         rewards = []
#         for episode in range(EPISODES):
#             state, _, _, _ = self.env.reset()
#             self.env.add_plane()
#             episode_reward = 0
#             done = False

#             for _ in range(MAX_STEPS):
#                 state_flat = self.get_flat_obs()
#                 actions = self.select_action(state_flat)
#                 next_state, reward, done, _ = self.env.step(actions)
#                 next_state_flat = self.get_flat_obs()

#                 self.replay_buffer.push(state_flat, actions, reward, next_state_flat, done)
#                 self.train_step()

#                 episode_reward += reward
#                 if done:
#                     break

#             if episode % TARGET_UPDATE_FREQ == 0:
#                 self.update_target()

#             rewards.append(episode_reward)
#             if episode % 10 == 0:
#                 print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}")

#         torch.save(self.policy_net.state_dict(), "centralized_dqn_airport.pth")
#         print("Training completed and model saved.")

# if __name__ == "__main__":
#     agent = CentralizedDQNAgent()
#     agent.train()


# === train.py ===
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from bradleyenv import BradleyAirportEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 100_000
TARGET_UPDATE_FREQ = 1000
EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5000
MAX_STEPS = 300

MAX_PLANES = 5
STATE_DIM = 6 * MAX_PLANES
ACTION_DIM = 13 * MAX_PLANES

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done, len(action)))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, num_planes = zip(*samples)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            action,
            torch.FloatTensor(reward).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(done).to(device),
            num_planes
        )

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class CentralizedDQNAgent:
    def __init__(self):
        self.env = BradleyAirportEnv()
        self.policy_net = DQN(STATE_DIM, ACTION_DIM).to(device)
        self.target_net = DQN(STATE_DIM, ACTION_DIM).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0
        self.epsilon = EPSILON_START

    def get_flat_obs(self):
        obs = []
        for plane in self.env.planes:
            obs.extend(self.env.get_obs(plane))
        while len(obs) < STATE_DIM:
            obs.extend([0.0] * 6)
        return np.array(obs[:STATE_DIM], dtype=np.float32)  # truncate to STATE_DIM


    def select_action(self, state):
        self.epsilon = max(EPSILON_END, EPSILON_START - self.steps_done / EPSILON_DECAY)
        self.steps_done += 1
        if random.random() < self.epsilon:
            return [random.randint(0, 12) for _ in range(len(self.env.planes))]
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze(0)
            actions = []
            for i in range(len(self.env.planes)):
                start = i * 13
                end = start + 13
                actions.append(torch.argmax(q_values[start:end]).item())
            return actions

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        state, action_batch, reward, next_state, done, num_planes = self.replay_buffer.sample(BATCH_SIZE)
        q_values = self.policy_net(state)
        next_q_values = self.target_net(next_state).detach()
        target_q = q_values.clone()

        for i in range(BATCH_SIZE):
            for j in range(num_planes[i]):
                idx = j * 13 + action_batch[i][j]
                target_q[i, idx] = reward[i] + GAMMA * next_q_values[i, idx] * (1 - done[i])

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        for episode in range(EPISODES):
            self.env.reset()
            self.env.add_plane()
            episode_reward = 0
            done = False

            for _ in range(MAX_STEPS):
                state_flat = self.get_flat_obs()
                actions = self.select_action(state_flat)
                _, reward, done, _ = self.env.step(actions)
                next_state_flat = self.get_flat_obs()

                self.replay_buffer.push(state_flat, actions, reward, next_state_flat, done)
                self.train_step()

                episode_reward += reward
                if done:
                    break

            if episode % TARGET_UPDATE_FREQ == 0:
                self.update_target()

            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        torch.save(self.policy_net.state_dict(), "centralized_dqn_airport.pth")
        print("Training completed and model saved.")

if __name__ == "__main__":
    agent = CentralizedDQNAgent()
    agent.train()

import numpy as np
import random
from bradleyenv import *

class ATC_agent:
    def __init__(self, num_actions, gamma=0.9, epsilon=1, decay_rate=0.995, epsilon_min=0.01):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min
        self.Q_table = {} # Dict to record Q values
        self.num_updates = {} # Dict to record the update times for each state and action pair 

    # Get Q values
    def get_QVal(self, state):
        # Convert state to a hashable type 
        current_state = tuple(state)
        # Check whether the state already in the table 
        if current_state not in self.Q_table:
            self.Q_table[current_state] = np.zeros(self.num_actions) # Initialize the state
        return self.Q_table[current_state]
    
    # Choose actions using ε-greedy policy
    def choose(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.get_QVal(state)) # Choose the action with the highest Q value
    
    # Update Q values
    def update(self, state, action, reward, next_state, done):
        current_state = tuple(state)
        # Get Q values
        current_Q = self.get_QVal(state)
        next_Q = self.get_QVal(next_state)
        action_Q = current_Q[action] 
        best_next_Q = np.max(next_Q)

        # Calculate dynamic learning rate
        updates = (current_state, action)
        if updates not in self.num_updates:
            self.num_updates[updates] = 0
        self.num_updates[updates] += 1
        eta = 1 / (1 + self.num_updates[updates])

        # Update Q value
        best_reward = reward if done else reward + self.gamma * best_next_Q
        new_Q = action_Q + eta * (best_reward - action_Q)
        self.Q_table[current_state][action] = new_Q 

        # Decay exploration rate
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay_rate 

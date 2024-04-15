import random
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        # self.Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    def append(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample_and_split(self, batch_size):
        transitions = self.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch=np.array(batch.state).reshape(batch_size,-1)
        action_batch= np.array(batch.action).reshape(batch_size,-1)
        reward_batch=np.array(batch.reward).reshape(batch_size,-1)
        next_state_batch=np.array(batch.next_state).reshape(batch_size,-1)
        done_batch =np.array(batch.done).reshape(batch_size,-1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.memory)
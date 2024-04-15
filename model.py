import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

def he_init(weight_shape):
    fan_in = weight_shape[1]  # weight_shape : [현재층 뉴런수, 이전층 뉴런수]
    variance = 2.0 / fan_in

    return torch.randn(weight_shape) * math.sqrt(variance)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden1, hidden2, init_w):
        super(Actor, self).__init__()
        # 이게 원래꺼
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_size)

        
        # self.fc1 = nn.Linear(state_size, hidden1)
        # self.fc2 = nn.Linear(hidden1, action_size)
        # self.fc3 = nn.Linear(hidden2, action_size)

        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = he_init(self.fc1.weight.data.size())
        self.fc2.weight.data = he_init(self.fc2.weight.data.size())
        # self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        out = F.leaky_relu(self.fc1(x))
        out = F.leaky_relu(self.fc2(out))
        # out = self.fc2(out)
        out = self.fc3(out)
        out = torch.tanh(out) 
        return out

class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden1, hidden2, init_w):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden1)
        self.fc2 = nn.Linear(hidden1 + action_size, hidden2)
        # self.fc2 = nn.Linear(hidden1 + action_size, 1)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = he_init(self.fc1.weight.data.size())
        self.fc2.weight.data = he_init(self.fc2.weight.data.size())
        # self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = F.leaky_relu(self.fc1(x)) 
        out = torch.cat([out, a], 1)     # 상태 + 행동
        # out = self.fc2(out)
        out = F.leaky_relu(self.fc2(out)) 
        out = self.fc3(out)
        return out

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from model import (Actor, Critic)
from buffer import ReplayBuffer
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2, sigma_decay = 0.99999):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        # print(f"{abs(np.mean(self.state)):.5f}")
        # print(self.sigma)
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        # self.decrease_sigma()
        return self.state
    
    def decrease_sigma(self):
        self.sigma = self.sigma * self.sigma_decay

class DDPG(object):
    def __init__(self, state_size, action_size, args):
        
        if args['seed'] > 0:
            self.seed(args['seed'])

        self.nb_states = state_size
        self.nb_actions= action_size
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args['hidden1'], 
            'hidden2':args['hidden2'], 
            'init_w':args['init_w']
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args['prate'])

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args['rate'])

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.replay_buffer = ReplayBuffer(args['rmsize'])
        self.random_process = OrnsteinUhlenbeckNoise(action_dimension=action_size, theta=args['ou_theta'], mu=args['ou_mu'], sigma=args['ou_sigma'])
        self.load_sigma = args['load_sigma']

        # Hyper-parameters
        self.batch_size = args['bsize']
        self.tau = args['tau']
        self.discount = args['discount']
        self.depsilon = 1.0 / args['epsilon']

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        #
        self.cur_episode = args['cur_episode']
        self.total_episode = args['total_episode']
        self.mode = args['mode']

        # 
        if USE_CUDA: self.cuda()

    def update_policy(self):
        if len(self.replay_buffer.memory) < self.batch_size:
            return
        # Sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.replay_buffer.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        with torch.no_grad():
            next_q_values = self.critic_target([
                to_tensor(next_state_batch),
                self.actor_target(to_tensor(next_state_batch)),
            ])

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float64))*next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([ to_tensor(state_batch), to_tensor(action_batch) ])
     
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        policy_loss = -self.critic([
            to_tensor(state_batch),
            self.actor(to_tensor(state_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        
        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()


    def select_action(self, s_t, decay_epsilon=True, noise= True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)

        # action += self.is_training * max(self.epsilon, 0) * self.random_process.noise()
        if noise:
            noise = self.random_process.noise()
            action += noise
            action = np.clip(action, -1., 1.)
            # action *= noise

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        # print(action)
        self.a_t = action
        # print(action)
        return action

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(torch.load(f'{output}/actor.pkl'))
        self.critic.load_state_dict(torch.load(f'{output}/critic.pkl'))

        buffer_data = torch.load(f'{output}/replay_buffer.pkl')
        for item in buffer_data:
            self.replay_buffer.memory.append(item)

        additional_info = torch.load(f'{output}/additional_info.pkl')
        self.epsilon = additional_info['epsilon']
        self.random_process.state = additional_info['random_process_state']
        if self.load_sigma:
            self.random_process.sigma = additional_info['random_process_sigma']
        self.cur_episode = additional_info['cur_episode']

    def save_model(self, output):
        torch.save(self.actor.state_dict(), f'{output}/actor.pkl')
        torch.save(self.critic.state_dict(), f'{output}/critic.pkl')

        # ReplayBuffer 저장
        buffer_data = list(self.replay_buffer.memory)
        torch.save(buffer_data, f'{output}/replay_buffer.pkl')

        # 추가 정보 저장 (epsilon, random_process.state, random_process.sigma, cur_episode)
        torch.save({
            'epsilon': self.epsilon,
            'random_process_state': self.random_process.state,
            'random_process_sigma': self.random_process.sigma,
            'cur_episode': self.cur_episode
        }, f'{output}/additional_info.pkl')


    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
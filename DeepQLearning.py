# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 13:16:11 2019

@author: mille
"""

import numpy as np
import torch

class DeepQNetwork(torch.nn.Module):
    def __init__(self, alpha):
        super(DeepQNetwork, self).__init__()
        self.lin1 = torch.nn.Linear(11, 120)
        self.lin2 = torch.nn.Linear(120, 120)
        self.lin3 = torch.nn.Linear(120, 120)
        self.lin4 = torch.nn.Linear(120, 3)
        
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=alpha)
        self.loss = torch.nn.MSELoss(reduction="sum")
        self.lr = 1e-4
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Running on CUDA")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")
        
        self.to(self.device)
        
    def forward(self, x):
        obs = torch.Tensor(x).to(self.device)
        
        #obs = obs.view()
        obs = torch.nn.functional.relu(self.lin1(obs))
        obs = torch.nn.functional.relu(self.lin2(obs))
        obs = torch.nn.functional.relu(self.lin3(obs))
        actions = self.lin4(obs)
        
        return actions


class DeepQNAgent(object):
    def __init__(self, gamma, epsilon, alpha, max_memory, eps_end=0,
                 replace=10000, action_space=[0,1,2]):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = eps_end
        self.mem_size = max_memory
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.mem_cntr = 0
        self.replace_target_cnt = replace
        self.action_space = action_space
        self.Q_eval = DeepQNetwork(alpha)
        self.Q_next = DeepQNetwork(alpha)
    
    def store_transition(self, state, action, reward, state_):
        if self.mem_cntr < self.mem_size:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.mem_cntr % self.mem_size] = \
            [state, action, reward, state_]
        self.mem_cntr += 1
        
    def choose_action(self, x):
        rand = np.random.random()
        actions = self.Q_eval.forward(x)
        if rand < 1 - self.EPSILON:
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        self.steps += 1
        return action
    
    def learn(self, batch_size):
        self.Q_eval.optimizer.zero_grad()
        if self.replace_target_cnt is not None and \
        self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())
            
        if self.mem_cntr + batch_size < self.mem_size:
            mem_start = int(np.random.choice(range(self.mem_cntr)))
        else:
            mem_start = int(np.random.choice(
                    range(self.mem_size-batch_size-1)))
        
        mini_batch = self.memory[mem_start:mem_start+batch_size]
        memory = np.array(mini_batch)
        
        Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(
                self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(
                self.Q_next.device)

        maxA = torch.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = torch.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        
        Qtarget = Qpred
        Qtarget[:,maxA] = rewards + self.GAMMA*torch.max(Qnext[1])
        
        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END
        
        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1
        
    def set_reward(self, snake):
        if snake.dead:
            return -10
        if snake.just_ate:
            return 10
        else:
            return 0

        
        
        
            
            
            
            
            
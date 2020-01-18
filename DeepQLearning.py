# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F

# Since input sizes don't vary
torch.backends.cudnn.benchmark = True


class DeepQNetwork(torch.nn.Module):
    def __init__(self, alpha):
        super(DeepQNetwork, self).__init__()
        self.lin1 = torch.nn.Linear(12, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, 4)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=alpha)
        self.loss = torch.nn.MSELoss()
        self.lr = 3e-4

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Running on CUDA")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")

        self.to(self.device)

    def forward(self, x):
        obs = torch.Tensor(x).to(self.device)
        obs = F.relu(self.lin1(obs))
        obs = F.relu(self.lin2(obs))
        actions = self.lin3(obs)
        return actions


class DeepQNAgent(object):
    def __init__(self, gamma, epsilon, alpha, max_memory, eps_end=0,
                 replace=10000, action_space=[0, 1, 2, 3]):
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

    def learn(self, batch_size, debug=False):
        self.Q_eval.optimizer.zero_grad()
        if (self.replace_target_cnt is not None and
                (self.learn_step_counter % self.replace_target_cnt) == 0):
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

        Qtarget = Qpred.clone()
        # Qtarget[:, maxA] = rewards + self.GAMMA*Qnext[:, maxA]

        for i in range(len(maxA)):
            if rewards[i] != 0:
                Qtarget[i, maxA[i]] = rewards[i] + self.GAMMA*Qnext[i, maxA[i]]

        if self.steps > 500:
            if self.EPSILON - self.Q_eval.lr > self.EPS_END:
                self.EPSILON -= self.Q_eval.lr
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)

        loss.backward()

        if debug:
            print(loss)

        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

# -*- coding: utf-8 -*-
import numpy as np
import torch

# Since input sizes don't vary
torch.backends.cudnn.benchmark = True


class PolicyNetwork(torch.nn.Module):
    """
    Neural network which will learn the optimal policy for each state i.e.
    what action will produce most return.
    """

    def __init__(self, alpha):
        super(PolicyNetwork, self).__init__()
        self.lin_seq = torch.nn.Sequential(
            torch.nn.Linear(12, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 3)
            )

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=alpha)
        self.loss = torch.nn.MSELoss()

        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Running on CUDA")
        else:
            self.device = torch.device("cpu")
            print("Running on CPU")
        """

        self.device = torch.device("cpu")

    def forward(self, state):
        obs = torch.Tensor(state.float())
        return self.lin_seq(obs)


class Policy(object):
    """
    The idea is to implement an optimal policy by learning at the end of games.
    We will get discounted rewards, so that the snake gets rewards when its
    steps make it get fruit in the close future.
    We need an optimal discount rewarding system so that the snake does
    understand that it has to go towards the fruit, but doesn't put itself in
    a situation where it is going to die.
    """

    def __init__(self, epsilon, alpha, discount=0.5, eps_end=0.05,
                 action_space=[0, 1, 2], lr=1e-3):
        self.epsilon = epsilon
        self.discount = discount
        self.eps_end = eps_end
        self.action_space = action_space
        self.steps = 0
        self.model = PolicyNetwork(alpha)
        self.memory_states = None
        self.memory_actions = np.array([])
        self.memory_rewards = np.array([])
        self.discounted = False
        self.lr = lr

    def store_transition(self, state, action, direct_reward):
        if self.memory_states is None:
            self.memory_states = state
        else:
            self.memory_states = np.vstack((self.memory_states, state))
        self.memory_actions = np.append(self.memory_actions, action)
        self.memory_rewards = np.append(self.memory_rewards, direct_reward)

    def choose_action(self, state):
        self.model.eval()
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.policy_action(state)
        self.steps += 1
        return action

    def policy_action(self, state):
        self.model.eval()
        x = torch.from_numpy(state)
        actions = self.model(x)
        return torch.argmax(actions).item()

    def discount_reward(self, i, next_reward):
        # Add reward at index i with discount of the next reward
        self.memory_rewards[i] += self.discount * next_reward

    def update_rewards(self):
        if self.discounted:
            return
        i = len(self.memory_actions) - 1
        current_reward = 0
        while i >= 0:
            self.discount_reward(i, current_reward)
            current_reward = self.memory_rewards[i]
            i -= 1
        self.discounted = True

    def learn(self, batch_size=1):
        if not self.discounted:
            self.update_rewards()
        self.model.train()
        # Remap rewards to a vector of 3 dimensions for the action taken
        targets = np.zeros((len(self.memory_actions), 3))
        for i in range(len(self.memory_actions)):
            targets[i][int(self.memory_actions[i])] = self.memory_rewards[i]
        mem_start = 0
        mem_end = 0
        torch_states = torch.from_numpy(
            self.memory_states).to(self.model.device)
        torch_targets = torch.from_numpy(targets)
        while mem_start < len(self.memory_actions):
            mem_end = mem_start + batch_size - 1
            if mem_end >= len(self.memory_actions):
                mem_end = len(self.memory_actions) - 1
            data = torch.from_numpy(self.memory_states[mem_start:mem_end+1])
            data = data.to(self.model.device)
            y_pred = self.model(torch_states[mem_start:mem_end+1])
            loss = self.model.loss(torch_targets[mem_start:mem_end+1],
                                   y_pred.double())
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            mem_start = mem_end + 1

        if self.epsilon > self.eps_end:
            self.epsilon -= self.lr
            if self.epsilon < self.eps_end:
                self.epsilon = self.eps_end

        self.memory_states = None
        self.memory_actions = np.array([])
        self.memory_rewards = np.array([])
        self.discounted = False

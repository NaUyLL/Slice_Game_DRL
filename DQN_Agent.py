"""
Pytorch版DQN
author:Luo Liyuan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self, N_states, N_actions, Hidden_nodes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_states, Hidden_nodes)
        self.out = nn.Linear(Hidden_nodes, N_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, N_states,
                 N_actions,
                 Hidden_nodes,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=100,
                 memory_size=2000,
                 batch_size=32,
                 e_greedy_increment=None):
        self.eval_net, self.target_net = \
            Net(N_states, N_actions, Hidden_nodes), Net(N_states, N_actions, Hidden_nodes)

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((memory_size, N_states * 2 + 2))  # initialize memory
        self.memory_size = memory_size
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

        self.replace_target_iter = replace_target_iter
        self.gamma = reward_decay  # 奖励折扣
        self.lr = learning_rate  # 学习率
        self.N_actions = N_actions  # 动作维度
        self.N_states = N_states  # 状态维度
        self.batch_size = batch_size  # 批数据的大小
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max


    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        is_random = 0
        if np.random.uniform() < self.epsilon:  # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:  # random
            action = np.random.randint(0, self.N_actions)
            is_random = 1
        return action, is_random

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_states])
        b_a = torch.LongTensor(b_memory[:, self.N_states:self.N_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.N_states + 1:self.N_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_states:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()




if __name__ == "__main__":
    pass
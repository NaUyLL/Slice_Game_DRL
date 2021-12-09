"""
author:Luo Liyuan
"""
import numpy as np
import torch
import gym
from DQN_Agent import DQN

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_actions = env.action_space.n
N_states = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


if __name__ == "__main__":
    dqn = DQN(N_states,
              N_actions,
              50,
              learning_rate=0.01,
              reward_decay=0.9,
              e_greedy=0.9,
              replace_target_iter=100,
              memory_size=2000,
              batch_size=32,
              e_greedy_increment=None)



    print('\nCollecting experience...')
    for i_episode in range(400):
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            a, is_random = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            s = s_
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import UAVsolar17test
# import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pandas as pd
# 存数据至文档
#
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger('充电battery9.30_17UAV+excel+.txt')


#####################  hyper parameters  ####################
EPISODES = 60000
EP_STEPS = 200
LR_ACTOR = 0.0001
LR_CRITIC = 0.0006
GAMMA = 0.8
TAU = 0.1
MEMORY_CAPACITY = 50000
BATCH_SIZE = 128
RENDER = False


# def text_create(name):
#     desktop_path = "/Users/wlongx/Desktop/未命名文件夹"
#     # 新创建的txt文件的存放路径
#     full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
#     file = open(full_path, 'w')
#
# filename = 'log'
# text_create(filename)
# output = sys.stdout
# outputfile = open("/Users/wlongx/Desktop/未命名文件夹" + filename + '.txt', 'w')
# sys.stdout = outputfile
########################## DDPG Framework ######################
class ActorNet(nn.Module):  # define the network structure for actor and critic
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 600)
        self.fc1.weight.data.normal_(0, 0.01)  # initialization of FC1
        self.fc2 = nn.Linear(600, 600)
        self.fc2.weight.data.normal_(0, 0.01)  # initialization of FC1
        self.out = nn.Linear(600, a_dim)
        self.out.weight.data.normal_(0, 0.01)  # initilizaiton of OUT

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        actions = torch.tanh(x)
        # for the game "Pendulum-v0", action range is [-2, 2]
        return actions


# class ActorNet(nn.Module): # define the network structure for actor and critic
#     def __init__(self, s_dim, a_dim):
#         super(ActorNet, self).__init__()
#         self.fc1 = nn.Linear(s_dim, 128)
#         self.fc1.weight.data.normal_(0, 0.02) # initialization of FC1
#         self.out = nn.Linear(128, a_dim)
#         self.out.weight.data.normal_(0, 0.02) # initilizaiton of OUT
#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.out(x)
#         actions = torch.tanh(x)
#         # for the game "Pendulum-v0", action range is [-2, 2]
#         return actions


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 600)
        self.fcs.weight.data.normal_(0, 0.01)
        self.fca = nn.Linear(a_dim, 600)
        self.fca.weight.data.normal_(0, 0.01)
        self.fcas = nn.Linear(600, 600)
        self.fcas.weight.data.normal_(0, 0.01)
        self.out = nn.Linear(600, 1)
        self.out.weight.data.normal_(0, 0.01)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        z = self.fcas(F.relu(x + y))
        actions_value = self.out(F.relu(z))
        return actions_value


# def fanin_init(size, fanin=None):
#     fanin = fanin or size[0]
#     v = 1. / np.sqrt(fanin)
#     return torch.Tensor(size).uniform_(-v, v)
# class CriticNet(nn.Module):
#     def __init__(self, s_dim, a_dim, hidden1=400, hidden2=300, init_w=3e-3):
#         super(CriticNet, self).__init__()
#         self.fc1 = nn.Linear(s_dim, hidden1)
#         self.fc2 = nn.Linear(hidden1 + a_dim, hidden2)
#         self.fc3 = nn.Linear(hidden2, 1)
#         self.relu = nn.ReLU()
#         self.init_weights(init_w)
#
#     def init_weights(self, init_w):
#         self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
#         self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
#         self.fc3.weight.data.uniform_(-init_w, init_w)
#
#     def forward(self, xs):
#         x, a = xs
#         out = self.fc1(x)
#         out = self.relu(out)
#         # debug()
#         out = self.fc2(torch.cat([out, a], 1))
#         out = self.relu(out)
#         out = self.fc3(out)
#         return out

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0  # serves as updating the memory data
        # Create the 4 network objects
        self.actor_eval = ActorNet(s_dim, a_dim)
        self.actor_target = ActorNet(s_dim, a_dim)
        self.critic_eval = CriticNet(s_dim, a_dim)
        self.critic_target = CriticNet(s_dim, a_dim)
        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):  # how to store the episodic data to buffer
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old data with new data
        self.memory[index, :] = transition
        self.pointer += 1

    def choose_action(self, s):
        # print(s)
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.actor_eval(s)[0].detach()

    def learn(self):
        # softly update the target networks
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')
            # sample from buffer a mini-batch data
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_trans = self.memory[indices, :]
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])
        # make action and evaluate its action values
        a = self.actor_eval(batch_s)
        q = self.critic_eval(batch_s, a)
        actor_loss = -torch.mean(q)
        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        # (~np.array([batch_s[i] == batch_s_[i] for i in range(len(batch_s))])
        q_target = batch_r + GAMMA * q_tmp

        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)
        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()


############################### Training ######################################
# Define the env in gym

env = UAVsolar17test.Maze()

s_dim = 35
a_dim = 17
# 61,30
# [0,1,-1,1,1,-1,1]
a_bound = 1
a_low_bound = -1

ddpg = DDPG(a_dim, s_dim, a_bound)
var = 1.5  # the controller of exploration which will decay during training process
t1 = time.time()
all_reward = []
all_reward1 = []
for i in range(EPISODES):
    env.reset()
    s = env.getstate()
    ep_r = 0
    for j in range(EP_STEPS):

        a = ddpg.choose_action(s)

        a = np.clip(np.random.normal(a, var), a_low_bound, a_bound)
        s_, r, done = env.step(a)
        s_ = env.getstate()
        # 离散化a +-0.5
        if done:
            s_ = s
        ddpg.store_transition(s, a, r, s_)  # store the transition to memory
        # 存离散的动作值会导致收敛的更快
        if ddpg.pointer > MEMORY_CAPACITY:
            #     var *= 0.999 # decay the exploration controller factor
            ddpg.learn()
        s = s_
        ep_r += r

        if done:
            var = max(var * 0.99998, 0.5)  # decay the exploration controller factor
            print('探索：', var, 'episode_reward:', ep_r, 'step', j, 'epoch', i)
            rewards = [ep_r]
            all_reward = np.concatenate((all_reward, rewards))
            # print(a)
            break

    if ep_r >= 1000:
        ep_r = 0
        env.reset()
        s = env.getstate()
        for j in range(EP_STEPS):
            a = ddpg.choose_action(s)
            var1 = 0
            a = np.clip(np.random.normal(a, var1), a_low_bound, a_bound)
            s_, r, done = env.step(a)
            s_ = env.getstate()
            s = s_
            ep_r += r
            if done:
                s = s_
                print('real', var1, ep_r, 'step', j, 'epoch', i)

                rewards1 = [ep_r]
                all_reward1 = np.concatenate((all_reward1, rewards1))
                break
# sns.set()
# sns.lineplot(x=range(len(all_reward)),y=all_reward)
print(all_reward)
ALL_reward = pd.DataFrame(all_reward)
ALL_reward.to_csv("reward17_excel")
plt.plot(all_reward)
plt.plot(all_reward1)
plt.xlabel("episode")
plt.ylabel("reward")
plt.show()
print('Running time: ', time.time() - t1)


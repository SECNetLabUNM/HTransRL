import copy

import numpy as np
import torch
import torch.nn.functional as F

from rl_multi_3d_trans import (net_sac_fc_0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_models = {

    'fc0': net_sac_fc_0,

    # add more mappings as needed
}


class SAC_countinuous():
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005

        self.actor = net_models[self.net_model].Actor(s1_dim=self.state_dim,
                                                      s2_dim=self.d2_dim,
                                                      action_dim=self.action_dim,
                                                      net_width=self.net_width).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = net_models[self.net_model].Double_Q_Critic(s1_dim=self.state_dim,
                                                                   s2_dim=self.d2_dim,
                                                                   action_dim=self.action_dim,
                                                                   net_width=self.net_width).to(self.device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.state_dim, self.d2_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

    def select_action(self, s1,s2, deterministic):
        # only used when interact with the env
        # with torch.no_grad():
        # 	state = torch.FloatTensor(state[np.newaxis,:]).to(self.dvc)
        # 	a, _ = self.actor(state, deterministic, with_logprob=False)
        # return a.cpu().numpy()[0]

        self.actor.eval()
        with torch.no_grad():
            s1 = np.array(s1)
            s1 = torch.FloatTensor(s1).to(device)
            try:
                s2 = np.array(s2)
            except:
                print(1)
            s2 = torch.FloatTensor(s2).to(device)

            dist, alpha, beta, nan_event = self.actor.get_dist(s1, s2, self.logger)

            assert torch.all((0 <= alpha))
            assert torch.all((0 <= beta))
            a = dist.sample()
            assert torch.all((0 <= a)) and torch.all(a <= 1)
            a = torch.clamp(a, 0, 1)
            logprob_a = dist.log_prob(a).cpu().numpy()
            return a.cpu().numpy(), logprob_a, alpha, beta

    def train(self, ):
        s1, s2, a, r, s1_next, s2_next, dw = self.replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(s1_next, s2_next, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(s1_next, s2_next, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * (
                    target_Q - self.alpha * log_pi_a_next)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s1, s2, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters(): params.requires_grad = False

        a, log_pi_a = self.actor(s1, s2, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s1, s2, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters(): params.requires_grad = True

        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, EnvName, timestep):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, timestep))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s1 = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.s2 = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s1_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.s2_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, s1, s2, a, r, s1_next, s2_next, dw):
        # 每次只放入一个时刻的数据
        self.s1[self.ptr] = torch.from_numpy(s1).to(self.dvc)
        self.s2[self.ptr] = torch.from_numpy(s2).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)  # Note that a is numpy.array
        self.r[self.ptr] = r
        self.s1_next[self.ptr] = torch.from_numpy(s1_next).to(self.dvc)
        self.s2_next[self.ptr] = torch.from_numpy(s2_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size  # 存满了又重头开始存
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s1[ind], self.s2[ind], self.a[ind], self.r[ind], self.s1_next[ind], self.s2_next[ind], self.dw[ind]

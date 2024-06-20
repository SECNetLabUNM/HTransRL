import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
# import sys
# sys.path.append('/home/user_name/PycharmProjects/air-corridor_ncfo/')
from rl_multi_3d_trans import (net_nn_fc_10_3e,
                               net_nn_fc_12,
                               net_nn_dec, )

net_models = {


    'fc10_3e': net_nn_fc_10_3e,
    'fc12': net_nn_fc_12,
    'dec': net_nn_dec,

    # add more mappings as needed
}


class MyDataset(Dataset):
    def __init__(self, data, env_with_Dead=True):
        self.data = data
        self.env_with_Dead = env_with_Dead

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transition = self.data[idx]
        s1, s2, a, r, s1_prime, s2_prime, logprob_a, done, dw, td_target, adv = transition

        # If your environment does not include Dead, modify dw here
        if self.env_with_Dead:  # Replace with your condition
            dw = False

        return {
            's1': torch.tensor(s1, dtype=torch.float),
            's2': torch.tensor(s2, dtype=torch.float),
            'a': torch.tensor(a, dtype=torch.float),
            'r': torch.tensor([r], dtype=torch.float),
            's1_prime': torch.tensor(s1_prime, dtype=torch.float),
            's2_prime': torch.tensor(s2_prime, dtype=torch.float),
            'logprob_a': torch.tensor(logprob_a, dtype=torch.float),
            'done': torch.tensor([done], dtype=torch.float),
            'dw': torch.tensor([dw], dtype=torch.float),
            'td_target': torch.tensor(td_target, dtype=torch.float),
            'adv': torch.tensor(adv, dtype=torch.float),

        }


class PPO(object):
    def __init__(
            self,
            state_dim=26,
            s2_dim=22,
            action_dim=3,
            env_with_Dead=True,
            gamma=0.99,
            lambd=0.95,
            # gamma=0.89,
            # lambd=0.88,
            clip_rate=0.2,
            K_epochs=10,
            net_width=256,
            a_lr=3e-4,
            c_lr=3e-4,
            l2_reg=1e-3,
            dist='Beta',
            a_optim_batch_size=64,
            c_optim_batch_size=64,
            entropy_coef=0,
            entropy_coef_decay=0.9998,
            writer=None,
            activation=None,
            share_layer_flag=True,
            anneal_lr=True,
            totoal_steps=0,
            with_position=False,
            token_query=False,
            num_enc=5,
            num_dec=5,
            logger=None,
            dir=None,
            test=False,
            net_model='fc1',
            beta_base=1e-5
    ):

        self.dir = dir
        self.logger = logger
        self.share_layer_flag = share_layer_flag
        shared_layers_actor = net_models[net_model].MergedModel(s1_dim=state_dim, s2_dim=s2_dim, net_width=net_width,
                                                                with_position=with_position, token_query=token_query,
                                                                num_enc=num_enc, num_dec=num_dec)
        if share_layer_flag:
            shared_layers_critic = shared_layers_actor
        else:
            shared_layers_critic = net_models[net_model].MergedModel(s1_dim=state_dim, s2_dim=s2_dim,
                                                                     net_width=net_width,
                                                                     with_position=with_position,
                                                                     token_query=token_query,
                                                                     num_enc=num_enc, num_dec=num_dec)
        self.dist = dist
        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.data = {}
        self.l2_reg = l2_reg
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.writer = writer
        self.anneal_lr = anneal_lr
        self.totoal_steps = totoal_steps
        self.a_lr = a_lr
        self.c_lr = c_lr
        # if not test:
        self.actor = net_models[net_model].BetaActorMulti(state_dim, s2_dim, action_dim, net_width,
                                                          shared_layers_actor, beta_base).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic = net_models[net_model].CriticMulti(state_dim, s2_dim, net_width, shared_layers_critic).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

    def load_pretrained(self):
        pass

    def select_action(self, s1, s2):  # only used when interact with the env
        self.actor.eval()
        with torch.no_grad():
            s1 = np.array(s1)
            s1 = torch.FloatTensor(s1).to(device)
            s2 = np.array(s2)
            s2 = torch.FloatTensor(s2).to(device)

            dist, alpha, beta = self.actor.get_dist(s1, s2)

            assert torch.all((0 <= alpha))
            assert torch.all((0 <= beta))
            a = dist.sample()
            assert torch.all((0 <= a)) and torch.all(a <= 1)
            a = torch.clamp(a, 0, 1)
            logprob_a = dist.log_prob(a).cpu().numpy()
            return a.cpu().numpy(), logprob_a, alpha, beta

    def evaluate(self, s1, s2,
                 deterministic=True):  # only used when evaluate the policy.Making the performance more stable
        self.actor.eval()
        with torch.no_grad():
            s1 = np.array(s1)
            s1 = torch.FloatTensor(s1).to(device)
            s2 = np.array(s2)
            s2 = torch.FloatTensor(s2).to(device)

            if deterministic:
                action_with_highest_probability = self.actor.dist_mode(s1, s2)
                chosen_action = action_with_highest_probability
            else:
                dist, alpha, beta = self.actor.get_dist(s1, s2)
                chosen_action = dist.sample()
            return chosen_action.cpu().numpy(), 0.0

    def train(self, global_step, epoches=None, anneal_mode='exponential'):

        if self.anneal_lr:
            if anneal_mode == 'linear':
                frac = 1.0 - global_step / self.totoal_steps
            if anneal_mode == 'exponential':
                frac = 0.995 ** (global_step / self.totoal_steps * 1000)
            print(f"learning discount: {round(frac * 100, 2)}%")
            alrnow = frac * self.a_lr
            clrnow = frac * self.c_lr
            # print(alrnow,clrnow)
            self.actor_optimizer.param_groups[0]["lr"] = alrnow
            self.critic_optimizer.param_groups[0]["lr"] = clrnow

        self.entropy_coef *= self.entropy_coef_decay

        transitions = self.gae()
        dataset = MyDataset(transitions)

        dataloader = DataLoader(dataset, batch_size=self.a_optim_batch_size, shuffle=True, drop_last=True)

        clipfracs = []
        for i in range(epoches):

            '''update the actor-critic'''
            self.actor.train()
            self.critic.train()
            # for i in range(a_optim_iter_num):
            for batch in dataloader:
                s1 = batch['s1'].to(device)
                s2 = batch['s2'].to(device)
                a = batch['a'].to(device)
                logprob_a = batch['logprob_a'].to(device)
                adv = batch['adv'].to(device)
                td_target = batch['td_target'].to(device)

                '''derive the actor loss'''
                # distribution, _, _, nan_event = self.actor.get_dist(s1, s2, self.logger)
                distribution, alpha, beta = self.actor.get_dist(s1, s2)
                # if nan_event:
                #     self.save('nan')
                dist_entropy = distribution.entropy().sum(1, keepdim=True)
                logprob_a_now = distribution.log_prob(a)

                logratio = logprob_a_now.sum(1, keepdim=True) - logprob_a.sum(1, keepdim=True)
                ratio = torch.exp(logratio)  # a/b == exp(log(a)-log(b))

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [torch.mean((ratio - 1.0).abs() > self.clip_rate, dtype=torch.float32).item()]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv
                pg_loss = -torch.min(surr1, surr2)
                a_loss = pg_loss - self.entropy_coef * dist_entropy

                '''derive the critic loss'''
                # index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s1.shape[0]))
                c_loss = (self.critic(s1, s2) - td_target).pow(2).mean()
                for name, param in self.critic.named_parameters():
                    if 'weight' in name:
                        c_loss += param.pow(2).sum() * self.l2_reg

                '''updata parameters'''
                self.actor_optimizer.zero_grad()
                a_loss.mean().backward(retain_graph=self.share_layer_flag)
                # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
                total_actor_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in self.actor.parameters() if
                                 p.grad is not None]), 2)

                total_actor_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 20)

                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                total_critic_norm_before = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in self.critic.parameters() if
                                 p.grad is not None]), 2)
                total_critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1000)
                # was_clipped = total_actor_norm > 20 or total_critic_norm > 1000
                # print(
                #     f" Total norm before clipping: {total_actor_norm_before:.4f}, After clipping: {total_critic_norm_before:.4f}, Was clipped: {was_clipped}")
                self.critic_optimizer.step()

        # y_pred, y_true = vs.cpu().numpy(), td_target.cpu().numpy()
        # var_y = np.var(y_true)
        # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # self.actor_cpu = copy.deepcopy(self.actor).to('cpu')
        # self.writer.add_scalar("charts/averaged_accumulated_reward", sum(accumulated_reward) / len(accumulated_reward),
        #                   global_step)
        self.writer.add_scalar("weights/critic_learning_rate", self.critic_optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", c_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
        self.writer.add_scalar("losses/entropy", dist_entropy.mean().item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        del a_loss, c_loss, pg_loss, dist_entropy, old_approx_kl, approx_kl, logprob_a_now, logratio  # , perm
        del surr1, surr2
        torch.cuda.empty_cache()
        self.data = {}

    def make_batch(self, agent):
        s1_lst = []
        s2_lst = []
        a_lst = []
        r_lst = []
        s1_prime_lst = []
        s2_prime_lst = []
        logprob_a_lst = []
        done_lst = []
        dw_lst = []
        for transition in self.data[agent]:
            s1, s2, a, r, s1_prime, s2_prime, logprob_a, done, dw = transition
            s1_lst.append(s1)
            s2_lst.append(s2)
            a_lst.append(a)
            logprob_a_lst.append(logprob_a)
            r_lst.append([r])
            s1_prime_lst.append(s1_prime)
            s2_prime_lst.append(s2_prime)
            done_lst.append([done])
            dw_lst.append([dw])

        if not self.env_with_Dead:
            '''Important!!!'''
            # env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
            # env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
            dw_lst = (np.array(dw_lst) * False).tolist()

        # self.data = []  # Clean history trajectory

        '''list to tensor'''
        with torch.no_grad():
            s1, s2, a, r, s1_prime, s2_prime, logprob_a, done_mask, dw_mask = \
                torch.tensor(np.array(s1_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s2_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(a_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(r_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s1_prime_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s2_prime_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(logprob_a_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(done_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(dw_lst), dtype=torch.float).to(device),
        return s1, s2, a, r, s1_prime, s2_prime, logprob_a, done_mask, dw_mask

    def gae(self, unification=True):
        transitions = []
        collect_adv = []
        for agent in self.data:
            s1, s2, _, r, s1_prime, s2_prime, _, done_mask, dw_mask = self.make_batch(agent)
            ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
            self.critic.eval()
            with torch.no_grad():
                vs = self.critic(s1, s2)
                vs_ = self.critic(s1_prime, s2_prime)
                '''dw for TD_target and Adv'''
                deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs
                deltas = deltas.cpu().flatten().numpy()
                adv = [0]
                '''done for GAE'''
                for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
                    advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                    adv.append(advantage)
                adv.reverse()
                adv = copy.deepcopy(adv[0:-1])
                collect_adv += adv
                td_target = np.array(adv) + np.array(vs.to('cpu').squeeze(1))
            for i, single_transition in enumerate(self.data[agent]):
                transitions.append(single_transition + [[td_target[i]], adv[i]])
        adv_mean = np.mean(collect_adv)
        adv_std = np.std(collect_adv)
        transitions = [tuple(tran[0:-1] + [[(tran[-1] - adv_mean) / (adv_std + 1e-6)]]) for tran in transitions]
        return transitions

    def put_data(self, agent, transition):
        if agent in self.data:
            self.data[agent].append(transition)
        else:
            self.data[agent] = [transition]

    def save(self, global_step, index=None):
        # global_step is usually interger, but also could be string for some events
        diff = f"_{index}" if index else ''
        if isinstance(global_step, str):
            global_step = global_step
            seq_name = f"{global_step}{diff}"
        else:
            global_step /= 1e6
            seq_name = f"{global_step}m{diff}"
        torch.save(self.actor.state_dict(), f"{self.dir}/ppo_actor_{seq_name}.pth")
        torch.save(self.critic.state_dict(), f"{self.dir}/ppo_critic_{seq_name}.pth")


    def load(self, folder, global_step, dir=None):
        if isinstance(global_step, float) or isinstance(global_step, int):
            global_step = str(global_step / 1000000) + 'm'
        if dir is not None:
            if global_step:
                self.critic.load_state_dict(torch.load(f"{dir}/ppo_critic_{global_step}.pth"), strict=False)
                self.actor.load_state_dict(torch.load(f"{dir}/ppo_actor_{global_step}.pth"), strict=False)
            else:
                self.critic.load_state_dict(torch.load(f"{dir}/ppo_critic.pth"), strict=False)
                self.actor.load_state_dict(torch.load(f"{dir}/ppo_actor.pth"), strict=False)
        else:
            if folder.startswith('/'):
                self.critic.load_state_dict(torch.load(f"{folder}/ppo_critic_{global_step}.pth"), strict=False)
                self.actor.load_state_dict(torch.load(f"{folder}/ppo_actor_{global_step}.pth"), strict=False)
            else:
                self.critic.load_state_dict(torch.load(f"./{folder}/ppo_critic_{global_step}.pth"), strict=False)
                self.actor.load_state_dict(torch.load(f"./{folder}/ppo_actor_{global_step}.pth"), strict=False)

    def load_and_copy(self, folder, global_step, a_lr, c_lr):
        if folder.startswith('/'):
            temp_critic = torch.load(f"{folder}/ppo_critic{global_step}.pth")
            temp_actor = torch.load(f"{folder}/ppo_actor{global_step}.pth")
        else:
            temp_critic = torch.load(f"./{folder}/ppo_critic{global_step}.pth")
            temp_actor = torch.load(f"./{folder}/ppo_actor{global_step}.pth")
        for name, param in self.critic.named_parameters():
            if name in temp_critic:
                param.data.copy_(temp_critic[name].data)
        for name, param in self.actor.named_parameters():
            if name in temp_actor:
                param.data.copy_(temp_actor[name].data)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

    def weights_track(self, global_step):
        total_sum = 0.0
        for param in self.actor.parameters():
            total_sum += torch.sum(param)
        self.writer.add_scalar("weights/actor_sum", total_sum, global_step)
        total_sum = 0.0
        for param in self.critic.parameters():
            total_sum += torch.sum(param)
        self.writer.add_scalar("weights/critic_sum", total_sum, global_step)

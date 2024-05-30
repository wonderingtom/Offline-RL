import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path

import utils
from dm_control.utils import rewards
from agents.agent_example import Agent, load_data

def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint>0:
            w = module.weight
            norm = torch.norm(w)
            w.mul_(torch.clip(constraint/norm, max=1))
    return fn

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=1e-3):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU())

        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        f = self.policy(obs)
        mu = self.fc_mu(f)
        log_std = self.fc_logstd(f)

        mu = torch.clamp(mu, -5, 5)

        std = log_std.clamp(-5, 2).exp()
        dist = utils.SquashedNormal2(mu, std)
        return dist

class C51Q_network(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, atom_size, support, device = 'cuda'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1 * atom_size)).to(device)
        self.support = support.to(device)
        self.atom_size = atom_size
    
    def forward(self, x):
        dist = self.dist(x)
        q = (dist * self.support).sum(-1)
        return q
    
    def dist(self, x):
        q_atoms = self.net(x).view(-1, 1, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=float(1e-3))  # 避免 nan
        return dist
    
    def dist2q(self, dist):
        return (dist * self.support).sum(-1)

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, atom_dim = 51, v_min = -50., v_max = 50. ,init_w=1e-3, device = 'cuda'):
        super().__init__()
        self.device = device
        self.atom_dim= atom_dim
        self.support = torch.linspace(v_min, v_max, atom_dim).to(device)
        self.delta_z = (v_max - v_min) / (atom_dim - 1)
        self.v_min = v_min
        self.v_max = v_max
        self.q1 = C51Q_network(obs_dim, action_dim, hidden_dim, atom_dim, self.support, self.device)
        self.q2 = C51Q_network(obs_dim, action_dim, hidden_dim, atom_dim, self.support, self.device)
        
    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        Q1 = self.q1(obs_action)
        Q2 = self.q2(obs_action)
        return Q1, Q2
    
    def dist(self, obs, action):
        obs_action = torch.cat([obs, action], dim=-1)
        Q1_dist = self.q1.dist(obs_action)
        Q2_dist = self.q2.dist(obs_action)
        return Q1_dist, Q2_dist

    def dist2q(self, Q1_dist, Q2_dist):
        return self.q1.dist2q(Q1_dist), self.q2.dist2q(Q2_dist)
    
    def dist_projection(self, optimal_dist, rewards, gamma):
        batch_size = rewards.shape[0]
        m = torch.zeros(batch_size, self.atom_dim).to(self.device)
        Tz = rewards.repeat(1, self.atom_dim) + gamma * self.support # [batch_size, self.atom_dim]
        Tz = torch.clamp(Tz, self.v_min, self.v_max).to(self.device)
        b = ((Tz - self.v_min) / self.delta_z).to(self.device)
        b -= 1e-4
        l = torch.floor(b).long().to(self.device) # [batch_size, self.atom_dim]
        u = torch.ceil(b).long().to(self.device) # [batch_size, self.atom_dim]
        idx = torch.arange(batch_size).repeat(self.atom_dim, 1).t().to(self.device)
        m[idx, l] += optimal_dist * (u - b)
        m[idx, u] += optimal_dist * (b - l)

        #print(m)
        return m / m.sum(-1).unsqueeze(-1)


class CDSAgent(Agent):
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 device,
                 actor_lr,
                 critic_lr,
                 hidden_dim,
                 critic_target_tau,
                 nstep,
                 batch_size,
                 use_tb,
                 alpha,
                 n_samples,
                 target_cql_penalty,
                 use_critic_lagrange,
                 num_expl_steps,
                 has_next_action=False):
        self.num_expl_steps = num_expl_steps
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.use_critic_lagrange = use_critic_lagrange
        self.target_cql_penalty = target_cql_penalty

        self.alpha = alpha
        self.n_samples = n_samples

        state_dim = obs_shape[0]
        action_dim = action_shape[0]

        # models
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim,device=device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim,device=device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # lagrange multipliers
        self.target_entropy = -self.action_dim
        self.log_actor_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.log_critic_alpha = torch.zeros(1, requires_grad=True, device=device)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_alpha_opt = torch.optim.Adam([self.log_actor_alpha], lr=actor_lr)
        self.critic_alpha_opt = torch.optim.Adam([self.log_critic_alpha], lr=actor_lr)

        self.train()
        # self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        policy = self.actor(obs)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample()
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def _repeated_critic_apply(self, obs, actions):
        """
            obs is (batch_size, obs_dim)
            actions is (n_samples, batch_size, action_dim)

            output tensors are (n_samples, batch_size, 1)
        """
        batch_size = obs.shape[0]
        n_samples = actions.shape[0]

        reshaped_actions = actions.reshape((n_samples * batch_size, -1))
        repeated_obs = obs.unsqueeze(0).repeat((n_samples, 1, 1))
        repeated_obs = repeated_obs.reshape((n_samples * batch_size, -1))

        Q1, Q2 = self.critic(repeated_obs, reshaped_actions)
        Q1 = Q1.reshape((n_samples, batch_size, 1))
        Q2 = Q2.reshape((n_samples, batch_size, 1))

        return Q1, Q2

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        # Compute standard SAC loss
        with torch.no_grad():
            dist = self.actor(next_obs)                 # SquashedNormal分布
            sampled_next_action = dist.sample()         # (1024, act_dim)
            # print("sampled_next_action:", sampled_next_action.shape)
            target_Q1_dist, target_Q2_dist = self.critic_target.dist(next_obs, sampled_next_action)  # (1024,1), (1024,1)
            target_Q1, target_Q2 = self.critic_target.dist2q(target_Q1_dist, target_Q2_dist)
              # (1024,1)
            index = torch.cat([target_Q1, target_Q2], dim=-1).argmin(-1)
            batch = torch.arange(index.shape[0])
            target_dist = torch.cat([target_Q1_dist, target_Q2_dist], dim=1)
            target_dist = target_dist[batch, index]
            target_dist = self.critic.dist_projection(target_dist, reward, discount)
            target_Q = reward + (discount * self.critic.q1.dist2q(target_dist))   # (1024,1)
            

        
        Q1_dist, Q2_dist = self.critic.dist(obs, action)
        Q1, Q2 = self.critic.dist2q(Q1_dist, Q2_dist)
        # critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)   # 标量
        critic_loss = F.cross_entropy(Q1_dist.squeeze(1), target_dist) + \
                        F.cross_entropy(Q2_dist.squeeze(1), target_dist)
        # Add CQL penalty
        with torch.no_grad():
            random_actions = torch.FloatTensor(self.n_samples, Q1.shape[0],
                        action.shape[-1]).uniform_(-1, 1).to(self.device)    # (n_samples, 1024, act_dim)
            sampled_actions = self.actor(obs).sample(
                sample_shape=(self.n_samples,))                              # (n_samples, 1024, act_dim)
            # print("sampled_actions:", sampled_actions.shape)
            next_sampled_actions = self.actor(next_obs).sample(
                sample_shape=(self.n_samples,))                              # (n_samples, 1024, act_dim)
            # print("next_sampled_actions:", next_sampled_actions.shape)

        rand_Q1, rand_Q2 = self._repeated_critic_apply(obs, random_actions)  # (n_samples, 1024, 1)
        sampled_Q1, sampled_Q2 = self._repeated_critic_apply(                # (n_samples, 1024, 1)
            obs, sampled_actions)
        next_sampled_Q1, next_sampled_Q2 = self._repeated_critic_apply(      # (n_samples, 1024, 1)
            obs, next_sampled_actions)

        # situation 1
        cat_Q1 = torch.cat([rand_Q1, sampled_Q1, next_sampled_Q1,
                            Q1.unsqueeze(0)], dim=0)                # (1+3*n_samples, 1024, 1)
        cat_Q2 = torch.cat([rand_Q2, sampled_Q2, next_sampled_Q2,
                            Q2.unsqueeze(0)], dim=0)                # (1+3*n_samples, 1024, 1)

        assert (not torch.isnan(cat_Q1).any()) and (not torch.isnan(cat_Q2).any())

        cql_logsumexp1 = torch.logsumexp(cat_Q1, dim=0).mean()
        cql_logsumexp2 = torch.logsumexp(cat_Q2, dim=0).mean()
        cql_logsumexp = cql_logsumexp1 + cql_logsumexp2

        cql_penalty = cql_logsumexp - (Q1 + Q2).mean()  # 标量

        # Update lagrange multiplier
        if self.use_critic_lagrange:
            alpha = torch.clamp(self.log_critic_alpha.exp(), min=0.0, max=1000000.0)
            alpha_loss = -0.5 * alpha * (cql_penalty - self.target_cql_penalty)

            self.critic_alpha_opt.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.critic_alpha_opt.step()
            alpha = torch.clamp(self.log_critic_alpha.exp(),
                                min=0.0,
                                max=1000000.0).detach()
        else:
            alpha = self.alpha

        # Combine losses
        critic_loss = critic_loss + alpha * cql_penalty

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = utils.grad_norm(self.critic.parameters())
        self.critic_opt.step()
        norm_constraint = 100
        self.critic.apply(l2_projection(norm_constraint))

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            # metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['critic_cql'] = cql_penalty.item()
            metrics['critic_cql_logsum1'] = cql_logsumexp1.item()
            # metrics['critic_cql_logsum2'] = cql_logsumexp1.item()
            metrics['rand_Q1'] = rand_Q1.mean().item()
            # metrics['rand_Q2'] = rand_Q2.mean().item()
            metrics['sampled_Q1'] = sampled_Q1.mean().item()
            # metrics['sampled_Q2'] = sampled_Q2.mean().item()
            metrics['critic_grad_norm'] = critic_grad_norm

        return metrics

    def update_actor(self, obs, action, step):
        metrics = dict()

        policy = self.actor(obs)
        sampled_action = policy.rsample()              # (1024, 6)
        # print("sampled_action:", sampled_action.shape)
        log_pi = policy.log_prob(sampled_action)       # (1024, 6)

        # update lagrange multiplier
        alpha_loss = -(self.log_actor_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.actor_alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.actor_alpha_opt.step()
        alpha = self.log_actor_alpha.exp().detach()

        # optimize actor
        Q1, Q2 = self.critic(obs, sampled_action)     # (1024, 1)
        Q = torch.min(Q1, Q2)                         # (1024, 1)
        actor_loss = (alpha * log_pi - Q).mean()      # 标量
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
        actor_grad_norm = utils.grad_norm(self.actor.parameters())
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = -log_pi.mean().item()
            metrics['actor_alpha'] = alpha.item()
            metrics['actor_alpha_loss'] = alpha_loss.item()
            metrics['actor_mean'] = policy.loc.mean().item()
            metrics['actor_std'] = policy.scale.mean().item()
            metrics['actor_action'] = sampled_action.mean().item()
            metrics['actor_atanh_action'] = utils.atanh(sampled_action).mean().item()
            metrics['actor_grad_norm'] = actor_grad_norm

        return metrics

    # TODO: conservative data sharing
    def conservative_data_share(self, batch_main, batch_share):
        obs_all, action_all, next_obs_all, reward_all, discount_all, obs_k_all = [], [], [], [], [], []

        # 1. sample examples from the main task
        # shape = (512, 24), (512, 6), (512, 1), (512, 1), (512, 24)
        obs_m, action_m, reward_m, discount_m, next_obs_m, _ = utils.to_torch(batch_main, self.device)
        # print("main samples:", obs_m.shape, action_m.shape, reward_m.shape, discount_m.shape, next_obs_m.shape, eps_flag_m.shape)
        obs_all.append(obs_m)
        action_all.append(action_m)
        next_obs_all.append(next_obs_m)
        reward_all.append(reward_m)
        discount_all.append(discount_m)

        # 2. sample examples from other tasks
        # sample 10 times samples, and select the top-0.1 samples.   (batch_size_split*10, xx, xx)
        # shape = (5120, 24) (5120, 6) (5120, 1) (5120, 1) (5120, 24)
        if batch_share is not None:
            obs, action, reward, discount, next_obs, _ = utils.to_torch(batch_share, self.device)
            # print("obs:", obs.shape, action.shape, reward.shape, discount.shape, next_obs.shape)
            # calculate the conservative Q value
            with torch.no_grad():
                conservative_q_value, _ = self.critic(obs, action)          # (5120, 1)
            conservative_q_value = conservative_q_value.squeeze().detach().cpu().numpy()
            # choose the top 0.1  top_index.shape=(512,)
            top_index = np.argpartition(conservative_q_value, -obs_m.shape[0])[-obs_m.shape[0]:]  # find the top 0.1 index. (batch_size_split,)
            # extract the samples
            obs_all.append(obs[top_index])
            action_all.append(action[top_index])
            next_obs_all.append(next_obs[top_index])
            reward_all.append(reward[top_index])
            discount_all.append(discount[top_index])

        return torch.cat(obs_all, dim=0), torch.cat(action_all, dim=0), torch.cat(reward_all, dim=0), \
            torch.cat(discount_all, dim=0), torch.cat(next_obs_all, dim=0)

    def update(self, replay_iter_main, replay_iter_share, step, total_step):
        metrics = dict()

        batch_main = next(replay_iter_main)  # len=6, inner_shape=512x24
        if replay_iter_share is not None:
            batch_share = next(replay_iter_share)
        else:
            batch_share = None

        # print("conservative data sharing...")   # obs.shape=(1024, 24), action.shape=(1024, 6) reward.shape=(1024, 1)
        obs, action, reward, discount, next_obs = self.conservative_data_share(batch_main, batch_share)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, action, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics

    def save(self, dir: Path) -> None:
        torch.save(self.actor, dir / "actor.pth")
        torch.save(self.critic, dir / "critic.pth")
    
    def load(self, load_path: Path) -> None:
        self.actor = torch.load(load_path / "actor.pth")
        self.critic = torch.load(load_path / "critic.pth")
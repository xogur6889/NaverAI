import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class PPO:
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 value_loss_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.actor_critic_optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        # with torch.no_grad():
        #     advantages = rollouts.returns - self.actor_critic.get_value(rollouts.obs[:-1])
        returns = rollouts.returns[:-1]
        value_preds = rollouts.value_preds[:-1]
        masks = rollouts.masks[:-1]
        advantages = returns - value_preds
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for e in range(self.ppo_epoch):
            # 10000개를 섞고 5000개를 뽑자.
            indices = list(BatchSampler(SubsetRandomSampler(range(10000)), 10000, drop_last=True))[0]
            random_obs = rollouts.obs[:-1][indices]
            random_value_preds = value_preds[indices]
            random_values = self.actor_critic.get_value(random_obs)
            random_returns = returns[indices]
            random_masks = masks[indices]
            random_actions = rollouts.actions[indices]
            random_old_action_log_probs = rollouts.action_log_probs[indices]
            random_advantages = advantages[indices]

            value_pred_clipped = random_value_preds + (random_values - random_value_preds).clamp(-self.clip_param, self.clip_param)
            value_losses = (random_returns - random_values).pow(2)
            value_losses_clipped = (value_pred_clipped - random_returns).pow(2)
            value_loss = (0.5 * torch.max(value_losses, value_losses_clipped) * random_masks).mean() * self.value_loss_coef
            # value_loss = 0.5 * (random_returns - random_values).pow(2).mean()
            # advantages = random_returns - values
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            action_log_probs = self.actor_critic.evaluate_actions(random_actions, random_obs)
            ratio = torch.exp(action_log_probs - random_old_action_log_probs)
            surr1 = ratio * random_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * random_advantages
            action_loss = (torch.min(surr1, surr2) * random_masks).mean()
            # dist_loss = (dist_entropy * rollouts.masks[:-1]).mean() * self.entropy_coef

            self.actor_critic_optimizer.zero_grad()
            # (value_loss - action_loss - dist_loss).backward()
            (value_loss - action_loss).backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.actor_critic_optimizer.step()

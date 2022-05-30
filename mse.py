import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.optim import RMSprop


class MSE:
    def __init__(self,
                 mac,
                 optim_alpha,
                 optim_eps,
                 lr=None,
                 max_grad_norm=None,
                 target_update_interval=None):

        self.mac = mac
        self.max_grad_norm = max_grad_norm
        self.parameters = list(self.mac.agent.parameters())
        self.mixer = VDNMixer()
        self.parameters += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)
        self.optimiser = RMSprop(params=self.parameters, lr=lr, alpha=optim_alpha, eps=optim_eps)
        self.target_mac = copy.deepcopy(mac)

        self.last_target_update_episode = 0
        self.target_update_interval = target_update_interval

    def update(self, rollouts, indices, episode_num):
        loss = 0
        for index in indices:
            obs_history = torch.FloatTensor(rollouts.obs_buffer[index])
            action_history = torch.LongTensor(rollouts.action_buffer[index])[:-1]
            # -1 하면 마지막 리워드 없음.
            reward_history = torch.FloatTensor(rollouts.reward_buffer[index])[1:]
            mask_history = torch.FloatTensor(rollouts.mask_buffer[index])

            mac_out = []
            self.mac.init_hidden()
            for obs in obs_history:
                agent_outs = self.mac.forward(obs)
                mac_out.append(agent_outs)
            mac_out = torch.stack(mac_out, dim=0)  # Concat over time

            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = torch.gather(mac_out[:-1], dim=2, index=action_history).squeeze(2)  # Remove the last dim
            # chosen_action_qvals[mask_history[:-1] == 0.0] = -9999999
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden()
            for obs in obs_history:
                target_agent_outs = self.target_mac.forward(obs)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            # r + maxQ에서 t+1 꺼씀.
            target_mac_out = torch.stack(target_mac_out[1:], dim=0)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=2)[0]
            # target_max_qvals[mask_history[1:] == 0.0] = -9999999

            chosen_action_qvals = self.mixer(chosen_action_qvals, None)
            target_max_qvals = self.target_mixer(target_max_qvals, None)

            # Mask out unavailable actions
            # targets = reward_history + 0.99 * target_max_qvals * mask_history
            targets = reward_history + 0.99 * target_max_qvals
            # Td-error
            td_error = (chosen_action_qvals - targets.detach())
            loss += (td_error ** 2).sum() / len(td_error)

        loss /= len(indices)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
            self.save_models(".")
            print('Save')

    def _update_targets(self):
        self.target_mac.load_state(self.mac)

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.optimiser.load_state_dict(torch.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs, batch):
        return torch.sum(agent_qs, dim=1, keepdim=True)

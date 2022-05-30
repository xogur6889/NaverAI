import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_size, hid_size, num_outputs):
        super(ActorCritic, self).__init__()
        self.dist = Categorical(hid_size, num_outputs)
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, hid_size),
            nn.Tanh(),
            nn.Linear(hid_size, 1)
        )
        # self.gru = nn.GRU(obs_size, hid_size)
        self.train()

    def th_act(self, inputs):
        actor_features = self.actor(inputs)
        dist = self.dist(actor_features)
        action = dist.mode()
        return action

    def act(self, obs):
        actor_features = self.actor(obs)
        dist = self.dist(actor_features)
        action = dist.sample()
        action_log_prob = dist.log_probs(action)
        return action, action_log_prob, self.get_value(obs)

    def evaluate_actions(self, actions, obs_s):
        actor_features = self.actor(obs_s)
        dist = self.dist(actor_features)
        action_log_probs = dist.th_log_probs(actions)
        return action_log_probs

    def _forward_gru(self, x, hxs, masks):
        x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
        x = x.squeeze(0)
        hxs = hxs.squeeze(0)
        return x, hxs

    def get_value(self, inputs):
        return self.critic(inputs)


class QNet(nn.Module):
    def __init__(self, n_agents, input_shape, rnn_hidden_dim, n_actions):
        super(QNet, self).__init__()
        self.n_agents = n_agents
        self.agent = RNNAgent(input_shape, rnn_hidden_dim, n_actions)
        self.action_selector = EpsilonGreedyActionSelector()
        self.hidden_states = None

    def act(self, obs, t):
        agent_outs = self.forward(obs)
        chosen_actions = self.action_selector.select_action(agent_outs, t)
        return chosen_actions

    def forward(self, agent_inputs):
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return agent_outs

    def init_hidden(self):
        self.hidden_states = self.agent.init_hidden().expand(self.n_agents, -1)  # bav

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))


class RNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNNAgent, self).__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_actions = n_actions
        self.fc1 = nn.Linear(input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class EpsilonGreedyActionSelector:
    def __init__(self):
        # self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
        self.schedule = DecayThenFlatSchedule(1.0, 0.05, 50000, decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, t):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t)

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        random_numbers = torch.rand_like(agent_inputs[:, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(torch.ones_like(agent_inputs[:, ]).float()).sample().long()
        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=1)[1]

        return picked_actions.unsqueeze(-1).tolist()


class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / numpy.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, numpy.exp(- T / self.exp_scaling)))

    pass

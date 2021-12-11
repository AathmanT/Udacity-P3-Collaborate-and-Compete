import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.actor_fc1 = nn.Linear(state_size, fc1_units)
        self.actor_bn1 = nn.BatchNorm1d(fc1_units)
        self.actor_fc2 = nn.Linear(fc1_units, fc2_units)
        self.actor_bn2 = nn.BatchNorm1d(fc2_units)
        self.actor_fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.actor_fc1.weight.data.uniform_(*hidden_init(self.actor_fc1))
        self.actor_fc2.weight.data.uniform_(*hidden_init(self.actor_fc2))
        self.actor_fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.actor_bn1(self.actor_fc1(state))
        x = F.relu(x)
        x = self.actor_bn2(self.actor_fc2(x))
        x = F.relu(x)

        return F.tanh(self.actor_fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.critic_fcs1 = nn.Linear(state_size * 2, fcs1_units)
        self.critic_bn1 = nn.BatchNorm1d(fcs1_units)
        self.critic_fc2 = nn.Linear(fcs1_units + action_size * 2, fc2_units)
        self.critic_fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.critic_fcs1.weight.data.uniform_(*hidden_init(self.critic_fcs1))
        self.critic_fc2.weight.data.uniform_(*hidden_init(self.critic_fc2))
        self.critic_fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        xs = self.critic_bn1(self.critic_fcs1(state))
        xs = F.relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.critic_fc2(x))

        return self.critic_fc3(x)

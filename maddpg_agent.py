import numpy as np
import random
import torch
import torch.nn.functional as F

from collections import namedtuple, deque
from ddpg_agent import Agent
from tensorboardX import SummaryWriter


BUFFER_SIZE = int(1e6)    # reply buffer size
BATCH_SIZE = 128          # minibatch size
GAMMA = 0.98              # discount factor
UPDATE_EVERY = 1          # how often to update the network
LEARN_TIMES = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
comment = " LR_ACTOR=1e-4 LR_CRITIC=1e-3 sigma=0.2 theta=0.1 grad clipping 2 samples [128,128]"


class MultiAgent():

    def __init__(self, num_agents=2, seed=0, state_size=24, action_size=2):

        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.update_every = UPDATE_EVERY
        self.learn_times = LEARN_TIMES
        self.num_agents = num_agents
        self.t_step = 0
        self.writer = SummaryWriter(comment=comment)
        self.train_epochs = 0

        # Initialize both DDPG agents
        self.agents = [Agent(state_size, action_size, seed) for i in range(num_agents)]

        # Replay buffer
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

    @staticmethod
    def reshape_array(input_array):
        return input_array.reshape(1, -1)

    def step(self, both_states, both_actions, both_rewards, both_next_states, both_dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        both_states = MultiAgent.reshape_array(both_states)
        both_next_states = MultiAgent.reshape_array(both_next_states)

        self.memory.add(both_states, both_actions, both_rewards, both_next_states, both_dones)

        self.t_step = (self.t_step + 1) % self.update_every
        # Learn every UPDATE_EVERY time steps.
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > self.batch_size:
                for i in range(self.learn_times):
                    self.learn(self.gamma)

    def act(self, all_states, add_noise=True):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, add_noise=add_noise)
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)

    @staticmethod
    def get_relevant_state(input_states_tensor, i):
        return torch.index_select(input_states_tensor.reshape(-1, 2, 24), 1, torch.tensor([i])).squeeze(1)

    @staticmethod
    def get_actor_predictions(actor_network, states_to_pred, i):
        relevant_state = MultiAgent.get_relevant_state(states_to_pred, i)
        actor_network_pred = actor_network(relevant_state)
        return actor_network_pred

    def learn(self, gamma):
        self.train_epochs += 1
        # Get separate samples for each agent to learn. This will give more stable training
        experiences = [self.memory.sample() for i in range(self.num_agents)]

        both_actions_next = []
        both_actions_pred = []
        for i, agent in enumerate(self.agents):
            states, actions, rewards, next_states, dones = experiences[i]
            both_actions_next.append(MultiAgent.get_actor_predictions(agent.actor_target, next_states, i))
            both_actions_pred.append(MultiAgent.get_actor_predictions(agent.actor_local, states, i))

        for i, agent in enumerate(self.agents):

            states, actions, rewards, next_states, dones = experiences[i]

            # ---------------------------- update critic ---------------------------- #
            # Get predicted next-state actions and Q values from target models

            actions_next = torch.cat(both_actions_next, dim=1).to(device)
            with torch.no_grad():
                Q_targets_next = agent.critic_target(next_states, actions_next)
            Q_expected = agent.critic_local(states, actions)
            # Compute Q targets for current states (y_i)
            Q_targets = torch.index_select(rewards, 1, torch.tensor([i])) + (gamma * Q_targets_next * (1 - torch.index_select(dones, 1, torch.tensor([i]))))
            # compute critic loss
            critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
            self.writer.add_scalar(f"Critic{i}_loss", critic_loss, self.train_epochs)

            # Minimize loss
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
            agent.critic_optimizer.step()

            # for name, param in agent.critic_local.named_parameters():
            #     self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.train_epochs)
            #     self.writer.add_histogram(name + "_grad", param.grad.data.cpu().numpy(), self.train_epochs)

            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            agent.actor_optimizer.zero_grad()
            actions_pred = []
            for j, actions in enumerate(both_actions_pred):
                if j != i:
                    actions_pred.append(actions.detach())
                else:
                    actions_pred.append(actions)

            actions_pred = torch.cat(actions_pred, dim=1).to(device)
            actor_loss = -agent.critic_local(states, actions_pred).mean()
            self.writer.add_scalar(f"Actor{i}_loss", actor_loss, self.train_epochs)

            # Minimize loss
            actor_loss.backward()
            agent.actor_optimizer.step()

            # for name, param in agent.actor_local.named_parameters():
            #     self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.train_epochs)
            #     self.writer.add_histogram(name + "_grad", param.grad.data.cpu().numpy(), self.train_epochs)

            # ----------------------- update target networks ----------------------- #
            agent.soft_update(agent.critic_local, agent.critic_target, agent.tau)
            agent.soft_update(agent.actor_local, agent.actor_target, agent.tau)

    def reset(self):
        for i, agent in enumerate(self.agents):
            agent.noise.reset()

    def save_agents(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), "checkpoint_actor_agent_"+str(i)+".pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic_agent_"+str(i)+".pth")

    def write_score(self, score, episode):
        self.writer.add_scalar("Mean_score", score, episode)

    def write_mean_score(self, moving_score, episode):
        self.writer.add_scalar("Moving_Mean_score", moving_score, episode)

    def finalize_log_writing(self):
        self.writer.flush()
        self.writer.close()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

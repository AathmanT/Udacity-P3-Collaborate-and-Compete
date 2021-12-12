#!/usr/bin/env python
# coding: utf-8

# # Collaboration and Competition
# 
# ---
# 
# In this notebook, you will learn how to use the Unity ML-Agents environment for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.
# 
# ### 1. Start the Environment
# 
# We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

# In[1]:


from unityagents import UnityEnvironment
import numpy as np


# Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.
# 
# - **Mac**: `"path/to/Tennis.app"`
# - **Windows** (x86): `"path/to/Tennis_Windows_x86/Tennis.exe"`
# - **Windows** (x86_64): `"path/to/Tennis_Windows_x86_64/Tennis.exe"`
# - **Linux** (x86): `"path/to/Tennis_Linux/Tennis.x86"`
# - **Linux** (x86_64): `"path/to/Tennis_Linux/Tennis.x86_64"`
# - **Linux** (x86, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86"`
# - **Linux** (x86_64, headless): `"path/to/Tennis_Linux_NoVis/Tennis.x86_64"`
# 
# For instance, if you are using a Mac, then you downloaded `Tennis.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
# ```
# env = UnityEnvironment(file_name="Tennis.app")
# ```

# In[2]:


env = UnityEnvironment(file_name="./Tennis_Windows_x86_64/Tennis.exe")


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[3]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# ### 2. Examine the State and Action Spaces
# 
# In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
# 
# The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
# 
# Run the code cell below to print some information about the environment.

# In[4]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# ### 3. Take Random Actions in the Environment
# 
# In the next code cell, you will learn how to use the Python API to control the agents and receive feedback from the environment.
# 
# Once this cell is executed, you will watch the agents' performance, if they select actions at random with each time step.  A window should pop up that allows you to observe the agents.
# 
# Of course, as part of the project, you'll have to change the code so that the agents are able to use their experiences to gradually choose better actions when interacting with the environment!

# In[5]:


# for i in range(1, 6):                                      # play game for 5 episodes
#     env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
#     states = env_info.vector_observations                  # get the current state (for each agent)
#     scores = np.zeros(num_agents)                          # initialize the score (for each agent)
#     while True:
#         actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
#         actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
#         env_info = env.step(actions)[brain_name]           # send all actions to tne environment
#         next_states = env_info.vector_observations         # get next state (for each agent)
#         rewards = env_info.rewards                         # get reward (for each agent)
#         dones = env_info.local_done                        # see if episode finished
#         scores += env_info.rewards                         # update the score (for each agent)
#         states = next_states                               # roll over states to next time step
#         if np.any(dones):                                  # exit loop if episode finished
#             break
#     print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))


# ## 4. Import

# In[6]:


import torch
import random
import matplotlib.pyplot as plt
from collections import deque

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 5. Agent implementation
# This notebook implements the Multi Agent Deep Deterministic Policy Gradients (MADDPG) method for training the agents.
# <br>Please refer the maddpg_agent script for MultiAgent class.

# In[7]:


import maddpg_agent 
import time
import importlib

importlib.reload(maddpg_agent)
seed = 0
agent = maddpg_agent.MultiAgent(num_agents, seed, state_size, action_size)


# In[8]:


def maddpg(n_episodes=4000, max_t=1000, print_every=100):
    
    scores = []
    scores_deque = deque(maxlen=print_every)
    avg_scores = []
    training_start_time = time.time()

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
#         agent.reset()
        state = env_info.vector_observations
        rewards = []

        start_time = time.time()
        for t in range(max_t):
            print('\rt {}'.format(t), end="")
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            rewards_vec = env_info.rewards
            done = env_info.local_done
            agent.step(state, action, rewards_vec, next_state, done)
            state = next_state
            rewards.append(rewards_vec)
            if any(done):
                break

        max_reward = np.max(np.sum(np.array(rewards), axis=0))

        scores.append(max_reward)
        scores_deque.append(max_reward)
        current_avg_score = np.mean(scores_deque)
        avg_scores.append(current_avg_score)
        duration = time.time() - start_time
        duration_str = time.strftime('%Mm%Ss', time.gmtime(duration))

        print('       Episode {} ({})\tAverage Score: {:.4f}'.format(i_episode, duration_str, current_avg_score))
        agent.write_score(max_reward, i_episode)
        agent.write_mean_score(np.mean(scores_deque),  i_episode)

        if i_episode % print_every == 0:
            print('Saving checkpoint...')
            agent.save_agents()

        if np.mean(scores_deque)>=0.5:
            training_time = time.time() - training_start_time
            training_time_str = time.strftime('%Hh%Mm%Ss', time.gmtime(training_time))
            print('\nEnvironment solved in {:d} episodes! ({}) \tAverage Score: {:.4f}'.format(i_episode, training_time_str, np.mean(scores_deque)))
            agent.save_agents()
            agent.finalize_log_writing()
            break
    return scores, avg_scores


# In[9]:


print(agent.agents[0].actor_local)
print(agent.agents[0].critic_local)
scores, avg_scores = maddpg()


# ### 6. Plot the rewards

# In[10]:


fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, 'b', label='Scores')
plt.plot(np.arange(len(scores)), avg_scores, 'r', linewidth=5, label='Moving avg')
plt.ylabel('Score')
plt.xlabel('Episode #')
ax.legend(fontsize='large')
fig.savefig('assets/rewards_plot.png', dpi=fig.dpi)

plt.show()


# ### 7. Watch the Smart Agents!
# In the next code cell, you will load the trained weights from file to watch the smart agents!

# In[ ]:


from unityagents import UnityEnvironment
import numpy as np
import random
import torch
from collections import deque
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import maddpg_agent 
import importlib

importlib.reload(maddpg_agent)
seed = 2
agent = maddpg_agent.MultiAgent(num_agents, seed, state_size)

for i, agent in enumerate(agent.agents):
    agent.actor_local.load_state_dict(torch.load(f"checkpoint_actor_agent_{i}.pth"))
    agent.critic_local.load_state_dict(torch.load(f"checkpoint_critic_agent_{i}.pth"))
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in range(1, 6):                                      # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = agent.act(states)                        # select an action (for each agent)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Episode {}: Agents score {}'.format(i, np.max(scores)))


# When finished, you can close the environment.

# In[ ]:


env.close()


# In[ ]:





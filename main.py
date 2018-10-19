import argparse, os
import math
from collections import namedtuple
from itertools import count

import gym
import numpy as np
from gym import wrappers

import torch
from ddpg import DDPG
from naf import NAF
from normalized_actions import NormalizedActions
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
from rover_domain import Task_Rovers
import utils as utils
from torch.autograd import Variable


class Parameters:
    def __init__(self):

        #Agent specific
        self.algo = 'DDPG' #DDPG | NAF
        self.gamma = 0.99
        self.tau = 0.001
        self.noise_scale = 0.5
        self.final_noise_scale = 0.3
        self.exploration_end = 10000 #Num of episodes with noise
        self.seed = 4

        #NN specifics
        self.num_hnodes = self.num_mem = 200
        self.is_dpp = True

        # Train data
        self.batch_size = 10000


        #Rover domain
        self.dim_x = self.dim_y = 15; self.obs_radius = 15; self.act_dist = 1.5; self.angle_res = 20
        self.num_poi = 10; self.num_rover = 4; self.num_timestep = 25
        self.poi_rand = 1; self.coupling = 2; self.rover_speed = 1
        self.is_homogeneous = True  #False --> Heterogenenous Actors
        self.sensor_model = 2 #1: Density Sensor
                              #2: Closest Sensor


        #Dependents
        self.state_dim = 2*360 / self.angle_res + 5
        self.action_dim = 2
        self.test_frequency = 10

        #Replay Buffer
        self.buffer_size = 1000000

        self.save_foldername = 'R_D++/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

        #Unit tests (Simply changes the rover/poi init locations)
        self.unit_test = 0 #0: None
                           #1: Single Agent
                           #2: Multiagent 2-coupled

        self.num_episodes = 100000
        self.updates_per_step = 1
        self.replay_size = 1000000
        self.render = 'False'


args = Parameters()
tracker = utils.Tracker(args, ['rewards'], '')
env = Task_Rovers(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.algo == "NAF":
    agent = NAF(args.gamma, args.tau, args.num_hnodes, env.observation_space.shape[0], env.action_space, args)
else:
    agent = DDPG(args.gamma, args.tau, args.num_hnodes, env.observation_space.shape[0], env.action_space, args)

memory = ReplayMemory(args.replay_size)
ounoise = OUNoise(env.action_space.shape[0])

for i_episode in range(args.num_episodes):
    joint_state = utils.to_tensor(np.array(env.reset()))

    if i_episode % args.test_frequency != 0:
        ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end - i_episode) / args.exploration_end + args.final_noise_scale
        ounoise.reset()
    episode_reward = 0
    for t in range(args.num_timestep):
        if i_episode % args.test_frequency == 0:
            joint_action = agent.select_action(joint_state)
        else:
            joint_action = agent.select_action(joint_state, ounoise)
        joint_next_state, joint_reward = env.step(joint_action.cpu().numpy())
        joint_next_state = utils.to_tensor(np.array(joint_next_state), volatile=True)
        done = t == args.num_timestep - 1
        episode_reward += np.sum(joint_reward)

        #Add to memory
        for i in range(args.num_rover):
            action = Variable(joint_action[i].unsqueeze(0))
            state = joint_state[i,:].unsqueeze(0)
            next_state = joint_next_state[i, :].unsqueeze(0)
            reward = utils.to_tensor(np.array([joint_reward[i]])).unsqueeze(0)
            memory.push(state, action, next_state, reward)

        state = next_state

        if len(memory) > args.batch_size * 5:
            for _ in range(args.updates_per_step):
                transitions = memory.sample(args.batch_size)
                batch = Transition(*zip(*transitions))
                agent.update_parameters_dpp(batch)

    if i_episode % args.test_frequency == 0:
        env.render()
        tracker.update([episode_reward], i_episode)
        print("Episode: {}, noise: {}, reward: {}, average reward: {}".format(i_episode, ounoise.scale,
                                                                          episode_reward, tracker.all_tracker[0][1]))
    


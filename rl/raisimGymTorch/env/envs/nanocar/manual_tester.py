from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.EleRLRaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import os
import math
import time
import torch
import argparse

from raisimGymTorch.env.bin.nanocar import RaisimGymEnv

import copy
import sys
import datetime
import sys
import numpy as np
from raisimGymTorch.algo.elegantrl.net import ActorSAC
from raisimGymTorch.algo.elegantrl import AgentSAC, AgentModSAC,Config
from raisimGymTorch.algo.elegantrl import ReplayBuffer
from raisimGymTorch.algo.elegantrl import Evaluator
from raisimGymTorch.algo.lattice_planner.lattice import LatticeTrajectoryGenerator


# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
cfg['environment']['num_envs'] = 1

max_step = 200#math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
control_dt = cfg['environment']['control_dt']
env = VecEnv(RaisimGymEnv(home_path + "/data", dump(cfg['environment'], Dumper=RoundTripDumper)), max_step=max_step)

a = np.array([[1, 2]])

state = env.reset()
lattice = LatticeTrajectoryGenerator(1,0.2,0.025)
for step in range(max_step):
    lattice.update(state[:,2],a)
    lattice.generate_trajectory()
    traj = lattice.Trajectory
    for i in range(len(traj[0].w)):
        action = np.array([[traj[0].v[i], traj[0].w[i]]],dtype=np.float32)
        state, reward, done, info_dict = env.step(action)
    # self.analyzer.add_reward_info(env.get_reward_info())


env.turn_off_visualization()
env.close()
print("Finished at the maximum visualization steps")
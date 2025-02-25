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

max_step = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
control_dt = cfg['environment']['control_dt']
env = VecEnv(RaisimGymEnv(home_path + "/data", dump(cfg['environment'], Dumper=RoundTripDumper)), max_step=max_step)

# shortcuts
ob_dim = env.state_dim
act_dim = env.action_dim

weight_path = args.weight
# weight_path = "/home/ws/raisim/raisimProject/3DNav/rl/Nanocar_SAC/2024-07-04-23-34-36/actor__000000002400_-0641.987.pt"
iteration_number = weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
weight_dir = weight_path.rsplit('/', 1)[0] + '/'
config = Config()
net_dims = (256, 1024, 2048, 256)
# agent = AgentSAC(net_dims, env.state_dim, env.action_dim, gpu_id=0, args=config)
# actor = ActorSAC(net_dims, env.state_dim, env.action_dim)



if weight_path == "":
    print("Can't find trained weight, please provide a trained weight with --weight switch\n")
else:
    print("Loaded weight from {}\n".format(weight_path))
    start = time.time()
    state = env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.
    n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    total_steps = n_steps * 1
    start_step_id = 0

    print("Visualizing and evaluating the policy: ", weight_path)

    device = torch.device("cpu")

    if os.path.isfile(weight_path):
        actor = torch.load(weight_path, map_location=lambda storage, loc: storage)
    env.turn_on_visualization()

    for step in range(max_step):
        start = time.time()

        action = actor(torch.from_numpy(state).to(device)).detach().cpu().numpy()
        # print(action)
        state, reward, done, info_dict = env.step(action)

        returns = reward
        dones = done

        reward_ll_sum = reward_ll_sum + returns[0]
        end = time.time()
        step_time = end - start
        if(step_time < control_dt):
            time.sleep(control_dt - step_time)
        if dones or step == max_step - 1:
            print('----------------------------------------------------')
            print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(reward_ll_sum / (step + 1 - start_step_id))))
            print('{:<40} {:>6}'.format("time elapsed [sec]: ", '{:6.4f}'.format((step + 1 - start_step_id) * control_dt)))
            print('----------------------------------------------------\n')
            start_step_id = step + 1
            reward_ll_sum = 0.0

    env.turn_off_visualization()
    env.close()
    print("Finished at the maximum visualization steps")

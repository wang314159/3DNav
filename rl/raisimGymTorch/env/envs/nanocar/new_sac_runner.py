import torch
import torch.nn as nn
import os
# import the skrl components to build the RL system
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.EleRLRaisimGymVecEnv import RaisimGymVecEnv as VecEnv

from raisimGymTorch.env.bin.nanocar import RaisimGymEnv

import copy
import sys
import time

import math
import datetime
import sys
from argparse import ArgumentParser
from raisimGymTorch.algo.elegantrl import Config
from raisimGymTorch.algo.elegantrl import AgentSAC, AgentModSAC
from raisimGymTorch.algo.elegantrl import ReplayBuffer
from raisimGymTorch.algo.elegantrl import Evaluator
import wandb



if __name__ == '__main__':

    task_path = os.path.dirname(os.path.realpath(__file__))
    home_path = task_path + "/../../../../.."
    cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

    agent_class = AgentSAC# DRL algorithm name
    env_class = VecEnv  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    use_wandb = True
    if use_wandb:
        wandb.init(project="nanocar_navigation")
    
    max_step = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
    control_dt = cfg['environment']['control_dt']
    eval_env = VecEnv(RaisimGymEnv(home_path + "/data", dump(cfg['environment'], Dumper=RoundTripDumper)), max_step=max_step)
    env = VecEnv(RaisimGymEnv(home_path + "/data", dump(cfg['environment'], Dumper=RoundTripDumper)), max_step=max_step)
    env.seed(cfg['seed'])
    print("env created")
    env.turn_off_visualization()
    eval_interval = cfg['environment']['eval_every_n']
    epoches = 100000
    
    env_args = {
        'env_name': 'Nanocar',  # the environment name
        'max_step': max_step,  # the max step number of an episode.
        'state_dim': env.state_dim,  # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': env.action_dim,  # the torque applied to free end of the pendulum
        'if_discrete': False,  # continuous action space, symbols → direction, value → force

        'num_envs': env.num_envs,  # the number of sub envs in vectorized env
        'if_build_vec_env': True,
    }
    print(env_args)
    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.net_dims = (256, 1024, 2048, 256)  # the middle layer dimension of MultiLayer Perceptron
    args.batch_size = 1024  # vectorized env need a larger batch_size
    args.gamma = 0.97  # discount factor of future rewards
    args.horizon_len = args.max_step

    args.repeat_times = 1.0  # repeatedly update network using ReplayBuffer to keep critic's loss small
    args.learning_rate = 2e-4
    args.state_value_tau = 0.2  # the tau of normalize for value and state `std = (1-std)*std + tau*std`

    args.eval_per_step = int(4e3)

    args.gpu_id = 0
    args.num_workers = 1
    if_single_process = True
    args.init_before_training()
    torch.set_grad_enabled(False)

    '''init environment'''
    # env = build_env(args.env_class, args.env_args, args.gpu_id)

    '''init agent'''
    agent = AgentSAC(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    '''init agent.last_state'''
    state = env.reset()

    assert state.shape == (args.num_envs, args.state_dim)
    assert isinstance(state, torch.Tensor)
    state = state.to(agent.device)
    agent.last_state = state.detach()

    '''init evaluator'''
    evaluator = Evaluator(cwd=args.cwd, env=eval_env, args=args, if_wandb=use_wandb)

    '''init buffer'''
    buffer = ReplayBuffer(
        gpu_id=args.gpu_id,
        num_seqs=args.num_envs,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        if_use_per=args.if_use_per,
        args=args,
    )
    buffer_items = agent.explore_vec_env(env, args.horizon_len * args.eval_times, if_random=True)
    buffer.update(buffer_items)  # warm up for ReplayBuffer

    '''train loop'''
    cwd = args.cwd
    horizon_len = args.horizon_len
    if_save_buffer = args.if_save_buffer
    del args

    for i in range(epoches):
        # print(i)
        # state = env.reset()
        # agent.last_state = state.to(agent.device).detach()
        buffer_items = agent.explore_vec_env(env, horizon_len)

        exp_r = buffer_items[2].mean().item()
        buffer.update(buffer_items)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)
        # print(f"| epoch: {i:3d} | exp_r: {exp_r:7.2f} | critic_loss: {logging_tuple[0]:7.2f} | actor_loss: {logging_tuple[1]:7.2f} | alpha_loss: {logging_tuple[2]:7.2f} |")
        if(i%eval_interval==0):
            evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len,epoch=i, exp_r=exp_r, logging_tuple=logging_tuple)

    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    env.close()
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, 'save_or_load_history'):
        buffer.save_or_load_history(cwd, if_save=True)

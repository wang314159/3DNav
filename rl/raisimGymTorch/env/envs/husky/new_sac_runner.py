import torch
import torch.nn as nn
import os
# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaac_orbit_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.NewRaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.husky import NormalSampler
from raisimGymTorch.env.bin.husky import RaisimGymEnv

# from raisimGymTorch.algo.sac.lstm import Actor,Critic
from raisimGymTorch.algo.sac.mlp import Actor,Critic
# from raisimGymTorch.algo.sac.rnn import Actor,Critic
# from raisimGymTorch.algo.sac.cnn import Actor,Critic
import copy
import tqdm
import sys
import time
# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed





# create environment from the configuration file
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))
env = VecEnv(RaisimGymEnv(home_path + "/data", dump(cfg['environment'], Dumper=RoundTripDumper)))
env.seed(cfg['seed'])
print("obs",env.num_obs)
print("act",env.num_acts)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=15625, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True, num_envs=env.num_envs)
models["critic_1"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["critic_2"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)

print("model created")
# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 80
cfg["learning_starts"] = 80
cfg["grad_norm_clip"] = 0
cfg["learn_entropy"] = True
cfg["entropy_learning_rate"] = 5e-3
cfg["initial_entropy_value"] = 1.0
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = "runs/torch/husky"
cfg["experiment"]["wandb"] = False

agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

print(env.num_agents)
time.sleep(2)
# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 160000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()

# initial_timestep=0
# timesteps=160000

# states, infos = env.reset()

# for timestep in tqdm.tqdm(range(initial_timestep, timesteps), disable=False, file=sys.stdout):

#             # pre-interaction
#             agent.pre_interaction(timestep=timestep, timesteps=timesteps)

#             # compute actions
#             with torch.no_grad():
#                 actions = agent.act(states, timestep=timestep, timesteps=timesteps)[0]

#                 # step the environments
#                 next_states, rewards, terminated, truncated, infos = env.step(actions)

#                 # render scene
#                 # if not headless:
#                 #     env.render()

#                 # record the environments' transitions
#                 agent.record_transition(states=states,
#                                               actions=actions,
#                                               rewards=rewards,
#                                               next_states=next_states,
#                                               terminated=terminated,
#                                               truncated=truncated,
#                                               infos=infos,
#                                               timestep=timestep,
#                                               timesteps=timesteps)

#             # post-interaction
#             agent.post_interaction(timestep=timestep, timesteps=timesteps)

#             # reset environments
#             if env.num_envs > 1:
#                 states = next_states
#             else:
#                 if terminated.any() or truncated.any():
#                     with torch.no_grad():
#                         states, infos = env.reset()
#                 else:
#                     states = next_states

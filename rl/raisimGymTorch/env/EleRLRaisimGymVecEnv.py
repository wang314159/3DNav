# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//

import numpy as np
import platform
import os
import torch
from torch import Tensor


class RaisimGymVecEnv():

    def __init__(self, impl, normalize_ob=True, seed=0, clip_obs=10.,max_step=1e4):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.state_dim = self.wrapper.getObDim()
        self.action_dim = self.wrapper.getActionDim()
        # self.observation_space = spaces.Box(-1e6,1e6,[self.num_envs,self.num_obs],dtype=np.float32)
        # self.action_space = spaces.Box(-50,50,[self.num_envs,self.num_obs],dtype=np.float32)
        # self.observation_space = spaces.Box(-1e6,1e6,[self.num_obs],dtype=np.float32)
        # self.action_space = spaces.Box(-50,50,[self.num_acts],dtype=np.float32)

        self._observations = np.zeros([self.num_envs,self.state_dim], dtype=np.float32)
        self.device = torch.device(f"cuda:{0}" if (torch.cuda.is_available() ) else "cpu")

        '''the necessary env information when you design a custom env'''
        self.env_name = "3DNav"  # the name of this env.
        self.max_step = max_step  # the max step number in an episode for evaluation
        self.if_discrete = False  # discrete action or continuous action

        self.log_prob = np.zeros(self.num_envs, dtype=np.float32)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=bool)
        self.rewards = [[] for _ in range(self.num_envs)]
        self.wrapper.setSeed(seed)
        self.count = 0.0
        self.mean = np.zeros(self.state_dim, dtype=np.float32)
        self.var = np.zeros(self.state_dim, dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action: Tensor) -> (Tensor, Tensor, Tensor, list[dict]): # type: ignore
        # print("step")
        self.wrapper.step(action.cpu().detach().numpy(), self._reward, self._done)
        states = torch.tensor(self._observations, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self._reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self._done, dtype=torch.bool, device=self.device)
        
        return states, rewards, dones,{}

    def load_scaling(self, dir_name, iteration, count=1e5):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        self.count = count
        self.mean = np.loadtxt(mean_file_name, dtype=np.float32)
        self.var = np.loadtxt(var_file_name, dtype=np.float32)
        self.wrapper.setObStatistics(self.mean, self.var, self.count)

    def save_scaling(self, dir_name, iteration):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"
        self.wrapper.getObStatistics(self.mean, self.var, self.count)
        np.savetxt(mean_file_name, self.mean)
        np.savetxt(var_file_name, self.var)

    def observe(self, update_statistics=True):
        self.wrapper.observe(self._observations, update_statistics)
        # print(self._observations.shape[0]," ",self._observations.shape[1])
        return self._observations

    def get_reward_info(self):
        return self.wrapper.getRewardInfo()

    def reset(self) -> Tensor:
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()
        states = torch.tensor(self.observe(), dtype=torch.float32, device=self.device)
        return states

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()
    
    def render():
        pass

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

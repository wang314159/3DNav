# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
import numpy as np


class EleRLRewardAnalyzer:

    def __init__(self, env,label=""):
        reward_info = env.get_reward_info()
        self.data_tags = list(reward_info[0].keys())
        self.data_size = 0
        self.data_mean = np.zeros(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_square_sum = np.zeros(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_min = np.inf * np.ones(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_max = -np.inf * np.ones(shape=(len(self.data_tags), 1), dtype=np.double)
        self.reward_dict = {}
        self.label = label

    def add_reward_info(self, info):
        self.data_size += len(info)

        for i in range(len(self.data_tags)):
            for j in range(len(info)):
                self.data_square_sum[i] += info[j][self.data_tags[i]]*info[j][self.data_tags[i]]
                self.data_mean[i] += info[j][self.data_tags[i]]
                self.data_min[i] = min(self.data_min[i], info[j][self.data_tags[i]])
                self.data_max[i] = max(self.data_max[i], info[j][self.data_tags[i]])

    def analyze(self):
        self.data_mean /= self.data_size
        data_std = np.sqrt((self.data_square_sum - self.data_size * self.data_mean * self.data_mean) / (self.data_size - 1 + 1e-16))

        for data_id in range(len(self.data_tags)):
            self.reward_dict["reward/"+self.data_tags[data_id]+'_mean'] = self.data_mean[data_id][0]
            self.reward_dict["reward/"+self.data_tags[data_id]+'_std'] = data_std[data_id][0]
            self.reward_dict["reward/"+self.data_tags[data_id]+'_min'] = self.data_min[data_id][0]
            self.reward_dict["reward/"+self.data_tags[data_id]+'_max'] = self.data_max[data_id][0]

        self.data_size = 0
        self.data_mean = np.zeros(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_square_sum = np.zeros(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_min = np.inf * np.ones(shape=(len(self.data_tags), 1), dtype=np.double)
        self.data_max = -np.inf * np.ones(shape=(len(self.data_tags), 1), dtype=np.double)
        return self.reward_dict


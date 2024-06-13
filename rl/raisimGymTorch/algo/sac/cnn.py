import torch
import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin
from skrl.models.torch import Model, DeterministicMixin


# define the model
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",num_envs=1):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        return self.net(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,num_envs=1):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=4),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(1024, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 16),
                                                nn.Tanh())

        self.net = nn.Sequential(nn.Linear(16 + self.num_actions, 64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 nn.Tanh(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        x = self.features_extractor(inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2))
        return self.net(torch.cat([x, inputs["taken_actions"]], dim=1)), {}
from stable_baselines3.common.buffers import BaseBuffer

class ReplayBuffer(BaseBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: torch.device = "cpu",
        n_envs: int = 1,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, handle_timeout_termination)
        self.n_envs = n_envs
        self.observations = np.zeros((buffer_size, n_envs) + observation_space.shape, dtype=observation_space.dtype)
        self.actions = np.zeros((buffer_size, n_envs) + action_space.shape, dtype=action_space.dtype)
        self.next_observations = np.zeros((buffer_size, n_envs) + observation_space.shape, dtype=observation_space.dtype)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done, info):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.pos = 0
            self.full = True

    def sample(self, batch_size):
        max_size = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, max_size, size=batch_size)

        obs = self.observations[batch_inds]
        next_obs = self.next_observations[batch_inds]
        actions = self.actions[batch_inds]
        rewards = self.rewards[batch_inds]
        dones = self.dones[batch_inds]

        return {
            "observations": torch.as_tensor(obs, device=self.device).float(),
            "next_observations": torch.as_tensor(next_obs, device=self.device).float(),
            "actions": torch.as_tensor(actions, device=self.device).float(),
            "rewards": torch.as_tensor(rewards, device=self.device).float(),
            "dones": torch.as_tensor(dones, device=self.device).float(),
        }

    def size(self):
        return self.buffer_size if self.full else self.pos

import cv2
import gym
import numpy as np
from gym.spaces.box import Box


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id):
    env = gym.make(env_id)

    if env_id == 'MountainCar-v0':
        env = MountainCarDiscrete(env)
    else:
        env = AtariRescale42x42(env)
        env = NormalizedEnv(env)
    return env


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class MountainCarDiscrete(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(MountainCarDiscrete, self).__init__(env)
        self.observation_space_orig = self.observation_space
        self.bins = [np.linspace(self.observation_space.low[i],
                                 self.observation_space.high[i],
                                 32) for i in [0,1]]
        self.observation_space = Box(0.0, 1.0, [1, 32,32])
    
    def observation(self, observation):
        """
            Return discretized observation.
            32x32 grid with a 1 at the observation.
        """
        coord = []
        for obs, bins in zip(observation, self.bins):
            ind = np.digitize(obs, bins)
            coord.append(ind)

        x = np.zeros((32, 32), dtype=np.float32)
        x[coord] = 1.0
        x = x.flatten()
        return np.array([x])

class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        obs = (observation - unbiased_mean) / (unbiased_std + 1e-8)
        return obs

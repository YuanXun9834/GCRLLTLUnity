from envs.unity import UnityGCRLLTLWrapper

__all__ = ['UnityGCRLLTLWrapper']

from typing import Any
import random

import torch
from torch.utils.data import Dataset
import numpy as np
from stable_baselines3.common.buffers import RolloutBuffer

from envs.utils import get_zone_vector

class TrajectoryBufferDataset(Dataset):
    def __init__(self, states, goal_values) -> None:
        self.states = states
        self.goal_values = goal_values
        
    def __getitem__(self, index) -> Any:
        s = self.states[index]
        v_omega = self.goal_values[index]
        return s, v_omega

    def __len__(self) -> int:
        return len(self.states)

class TrajectoryBuffer:
    ZONE_OBS_DIM = 24  # Unity zone vector dimension

    def __init__(
        self, 
        traj_length: int = 1000,
        buffer_size: int = 100000,
        obs_dim: int = 100,
        n_envs: int = 1,
        device: str = 'cpu',
    ):
        self.traj_length = traj_length
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.n_envs = n_envs
        self.device = device
        self.zone_vector = get_zone_vector()
        self.reset()
    
    def reset(self) -> None:
        self.buffers = {}
        buffer_length = int(self.buffer_size / self.n_envs)
        for pid in range(self.n_envs):
            self.buffers[pid] = {
                'obs': np.zeros((buffer_length, self.obs_dim), dtype=np.float32),
                'success': np.zeros((buffer_length, 1), dtype=np.bool_), 
                'steps': np.zeros((buffer_length, 1), dtype=np.int32),
            }
        self.pos = 0

    def add_rollouts(self, rollout_buffer: RolloutBuffer) -> None:
        rollout_obs = rollout_buffer.observations['obs'].transpose(1, 0, 2)
        rollout_success = rollout_buffer.observations['success'].transpose(1, 0, 2)
        rollout_steps = rollout_buffer.observations['steps'].transpose(1, 0, 2)

        shape = rollout_obs.shape
        forward_steps = shape[1]

        for pid in range(self.n_envs):
            self.buffers[pid]['obs'][self.pos: self.pos + forward_steps] = rollout_obs[pid]
            self.buffers[pid]['success'][self.pos: self.pos + forward_steps] = rollout_success[pid]
            self.buffers[pid]['steps'][self.pos: self.pos + forward_steps] = rollout_steps[pid]

        self.pos += forward_steps

    def build_dataset(self, policy):
        states, goal_values = [], []
        buffer_length = int(self.buffer_size / self.n_envs)
        with torch.no_grad():
            for pid in range(self.n_envs):
                local_states, local_goal_values = [], []
                pos, forward_steps = 0, 0
                while pos < buffer_length:
                    local_states.append(self.buffers[pid]['obs'][pos])
                    if forward_steps >= self.traj_length - 1 and not self.buffers[pid]['success'][pos]:
                        local_states, local_goal_values = [], []
                        forward_steps = 0
                    elif self.buffers[pid]['success'][pos]:
                        if random.random() > 0.9:
                            _local_states = []
                            for state in local_states:
                                values = self.get_goal_value(state, policy)
                                for g in values:
                                    if values[g]:
                                        _local_states.append(np.concatenate((state, self.zone_vector[g]), axis=0))
                                        local_goal_values.append(values[g])
                            states += _local_states
                            goal_values += local_goal_values
                        local_states, local_goal_values = [], []
                        forward_steps = 0
                    
                    pos += 1
                    forward_steps += 1

        print(f"Collected {len(states)} states and {len(goal_values)} goal values")
        return TrajectoryBufferDataset(states=states, goal_values=goal_values)
    
    def get_goal_value(self, state, policy):
        goal_value = {'green': None, 'red': None, 'yellow': None}
        for zone in self.zone_vector:
            if not np.allclose(state[-self.ZONE_OBS_DIM:], self.zone_vector[zone]):
                with torch.no_grad():
                    obs = torch.from_numpy(
                        np.concatenate((state[:-self.ZONE_OBS_DIM], self.zone_vector[zone]))
                    ).unsqueeze(dim=0).to(self.device)
                    goal_value[zone] = policy.predict_values(obs)[0].item()
        
        return goal_value
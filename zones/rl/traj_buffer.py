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
    ZONE_OBS_DIM = 24

    def __init__(
        self, 
        traj_length: int = 1000,
        buffer_size: int = 100000,
        obs_dim: int = 62,  # Updated to match actual observation size
        n_envs: int = 1,
        device: str = 'cpu',
    ):
        self.traj_length = traj_length
        self.buffer_size = max(buffer_size, 2048 * n_envs)  # Ensure buffer is large enough
        self.obs_dim = obs_dim
        self.n_envs = n_envs
        self.device = device
        self.zone_vector = get_zone_vector()
        print(f"Initializing TrajectoryBuffer with:")
        print(f"- traj_length: {traj_length}")
        print(f"- buffer_size: {self.buffer_size}")
        print(f"- obs_dim: {obs_dim}")
        print(f"- n_envs: {n_envs}")
        self.reset()
    
    def reset(self) -> None:
        self.buffers = {}
        buffer_length = max(int(self.buffer_size / self.n_envs), 2048)  # Ensure minimum size
        print(f"Creating buffers with length: {buffer_length}")
        
        for pid in range(self.n_envs):
            self.buffers[pid] = {
                'obs': np.zeros((buffer_length, self.obs_dim), dtype=np.float32),
                'success': np.zeros((buffer_length, 1), dtype=np.bool_), 
                'steps': np.zeros((buffer_length, 1), dtype=np.int32),
            }
        self.pos = 0
        
    def add_rollouts(self, rollout_buffer: RolloutBuffer) -> None:
        try:
            rollout_obs = rollout_buffer.observations['obs'].transpose(1, 0, 2)
            shape = rollout_obs.shape
            forward_steps = shape[1]
            
            print(f"\nAdding rollouts to trajectory buffer:")
            print(f"- Rollout shape: {shape}")
            print(f"- Current buffer position: {self.pos}")
            print(f"- Forward steps: {forward_steps}")
            print(f"- Buffer size: {self.buffer_size}")
            
            # Check if we need to wrap around or reset
            if self.pos + forward_steps > len(self.buffers[0]['obs']):
                print("Buffer position would exceed size, resetting to 0")
                self.pos = 0
                
            for pid in range(self.n_envs):
                try:
                    # Get current buffer state
                    print(f"\nEnvironment {pid}:")
                    print(f"- Buffer shape: {self.buffers[pid]['obs'].shape}")
                    print(f"- Target slice: [{self.pos}:{self.pos + forward_steps}]")
                    print(f"- Rollout shape for this env: {rollout_obs[pid].shape}")
                    
                    # Copy data
                    self.buffers[pid]['obs'][self.pos: self.pos + forward_steps] = rollout_obs[pid]
                    self.buffers[pid]['success'][self.pos: self.pos + forward_steps] = \
                        rollout_buffer.observations['success'].transpose(1, 0, 2)[pid]
                    self.buffers[pid]['steps'][self.pos: self.pos + forward_steps] = \
                        rollout_buffer.observations['steps'].transpose(1, 0, 2)[pid]
                    
                    print("Successfully copied data")
                    
                except Exception as e:
                    print(f"Error in environment {pid}:")
                    print(f"- Error message: {str(e)}")
                    print(f"- Buffer section shape: {self.buffers[pid]['obs'][self.pos: self.pos + forward_steps].shape}")
                    print(f"- Rollout section shape: {rollout_obs[pid].shape}")
                    raise

            self.pos += forward_steps
            print(f"New buffer position: {self.pos}")
            
        except Exception as e:
            print(f"Error in add_rollouts: {str(e)}")
            print("Full error information:", e)
            raise
        
    def build_dataset(self, policy):
        states, goal_values = [], []
        buffer_length = int(self.buffer_size / self.n_envs)
        print("\nBuilding dataset from buffer:")
        print(f"Buffer length per env: {buffer_length}")
        
        total_successes = 0
        total_trajectories = 0
        
        with torch.no_grad():
            for pid in range(self.n_envs):
                success_count = np.sum(self.buffers[pid]['success'])
                print(f"\nEnvironment {pid}:")
                print(f"Total successes: {success_count}")
                print(f"Buffer observation shape: {self.buffers[pid]['obs'].shape}")
                local_states, local_goal_values = [], []
                pos, forward_steps = 0, 0
                
                
                while pos < buffer_length:
                    local_states.append(self.buffers[pid]['obs'][pos])
                    if forward_steps >= self.traj_length - 1 and not self.buffers[pid]['success'][pos]:
                        print(f"  Resetting trajectory at pos {pos} (no success)")
                        local_states, local_goal_values = [], []
                        forward_steps = 0
                    elif self.buffers[pid]['success'][pos]:
                        if random.random() > 0.9:
                            _local_states = []
                            # compute the goal-value for all possible goals
                            for state in local_states:
                                values = self.get_goal_value(state, policy)
                                for g in values:
                                    if values[g] is not None:
                                        _local_states.append(np.concatenate((state, get_zone_vector()[g]), axis=0))
                                        local_goal_values.append(values[g])
                            states += _local_states
                            goal_values += local_goal_values
                            print(f"  Added successful trajectory at pos {pos}")
                            print(f"  - States added: {len(_local_states)}")
                            print(f"  - Values added: {len(local_goal_values)}")
                        local_states, local_goal_values = [], []
                        forward_steps = 0
                    
                    pos += 1
                    forward_steps += 1
                    total_successes += success_count

        print(f"\nDataset Summary:")
        print(f"Total environments processed: {self.n_envs}")
        print(f"Total successful episodes: {total_successes}")
        print(f"Total states collected: {len(states)}")
        print(f"Total goal values collected: {len(goal_values)}")
        
        return TrajectoryBufferDataset(states=states, goal_values=goal_values)
    
    def get_goal_value(self, state, policy):
        goal_value = {'J': None, 'W': None, 'R': None, 'Y': None}
        for zone in self.zone_vector:
            if not np.allclose(state[-self.ZONE_OBS_DIM:], self.zone_vector[zone]):
                with torch.no_grad():
                    obs = {'obs': torch.from_numpy(np.concatenate((state[:-self.ZONE_OBS_DIM], self.zone_vector[zone]))).unsqueeze(dim=0).to(self.device)}
                    goal_value[zone] = policy.predict_values(obs)[0].item()
        
        return goal_value

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
        self.successful_trajectories = []
        
    def add_rollouts(self, episodes):
        """
        Add successful episodes to the buffer
        
        Args:
            episodes: List of dictionaries containing successful episodes
                     Each dict has 'states' and 'rewards' keys
        """
        print(f"Adding {len(episodes)} episodes to buffer")
        for episode in episodes:
            if len(episode['states']) > 0:  # Only add non-empty episodes
                self.successful_trajectories.append({
                    'states': episode['states'].copy(),
                    'rewards': episode['rewards'].copy(),
                })
                print(f"Added episode with {len(episode['states'])} steps")
                
        # Trim buffer if it exceeds size
        while len(self.successful_trajectories) > self.buffer_size:
            self.successful_trajectories.pop(0)
            
    def build_dataset(self, policy):
        """
        Build a dataset from collected trajectories
        """
        print("\nBuilding dataset from collected trajectories")
        print(f"Number of successful trajectories: {len(self.successful_trajectories)}")
        
        all_states = []
        all_goal_values = []
        
        for trajectory in self.successful_trajectories:
            states = trajectory['states']
            
            if len(states) < 2:  # Skip very short trajectories
                continue
                
            print(f"Processing trajectory with {len(states)} states")
            
            # Sample states from trajectory
            sample_indices = np.random.choice(
                len(states), 
                size=min(self.traj_length, len(states)), 
                replace=False
            )
            
            for idx in sample_indices:
                state = states[idx]
                
                try:
                    # Get value predictions for this state
                    if isinstance(state, dict):
                        obs = state['obs']
                    else:
                        obs = state
                        
                    with torch.no_grad():
                        value = policy.predict_values(
                            torch.as_tensor(obs).unsqueeze(0).to(self.device)
                        )[0].item()
                        
                    all_states.append(obs)
                    all_goal_values.append(value)
                    
                except Exception as e:
                    print(f"Error processing state: {e}")
                    continue
                    
        print(f"\nDataset building complete:")
        print(f"Total states collected: {len(all_states)}")
        print(f"Total values collected: {len(all_goal_values)}")
        
        return TrajectoryBufferDataset(
            states=np.array(all_states),
            goal_values=np.array(all_goal_values)
        )
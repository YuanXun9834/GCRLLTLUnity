import argparse
import random

import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.traj_buffer import TrajectoryBuffer
from rl.callbacks import CollectTrajectoryCallback
from envs.unity import UnityGCRLLTLWrapper 
from envs.utils import get_zone_vector

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            if key == 'obs':
                # Adjust network size based on Unity observation space
                extractors[key] = nn.Linear(subspace.shape[0], 100)
                total_concat_size += 100

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            
        return torch.cat(encoded_tensor_list, dim=1)
    
def make_unity_env():
    """Create Unity environment function"""
    env = UnityGCRLLTLWrapper(
        env_path=args.unity_env_path,
        worker_id=0,  # Will be set automatically for parallel envs
        no_graphics=True
    )
    return env

def main(args):
    device = torch.device(args.device)
    timeout = args.timeout
    total_timesteps = args.total_timesteps
    num_cpus = args.num_cpus
    seed = args.seed
    exp_name = args.exp_name

    # Create vectorized environment
    env = make_vec_env(
        make_unity_env,
        n_envs=num_cpus,
        seed=seed,
        vec_env_cls=SubprocVecEnv
    )
    
    # Setup model
    model = PPO(
        policy='MultiInputPolicy',
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[512, 1024, 256],
            features_extractor_class=CustomCombinedExtractor,
        ),
        env=env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.998,
        n_epochs=10,
        n_steps=int(50000/num_cpus),
        batch_size=1000,
        ent_coef=0.003,
        device=device,
    )

    # Setup logging and evaluation
    log_path = f'logs/ppo/{exp_name}/'
    new_logger = configure(log_path, ['stdout', 'csv'])
    model.set_logger(new_logger)

    eval_env = make_vec_env(make_unity_env)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=100000/num_cpus,
        n_eval_episodes=10,
        deterministic=True,
    )
    
    # Setup trajectory collection
    traj_buffer = TrajectoryBuffer(
        traj_length=1000,
        buffer_size=total_timesteps,
        obs_dim=100,
        n_envs=num_cpus,
        device=device
    )
    traj_callback = CollectTrajectoryCallback(traj_buffer=traj_buffer)

    callback = CallbackList([eval_callback, traj_callback])
    
    # Train model
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save trajectory dataset
    traj_dataset = traj_buffer.build_dataset(model.policy)
    torch.save(traj_dataset, f'./datasets/{exp_name}_traj_dataset.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--timeout', type=int, default=1000)
    parser.add_argument('--total_timesteps', type=int, default=1e7)
    parser.add_argument('--num_cpus', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--exp_name', type=str, default='traj_exp')
    parser.add_argument('--unity_env_path', type=str, required=True,
                      help='Path to Unity environment executable')
    
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
import argparse
import random
import os
from functools import partial

import torch
import torch.nn as nn
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

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for the Unity environment"""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=1)

        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            if key == 'obs':
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], 512),
                    nn.ReLU(),
                    nn.Linear(512, 256)
                )
                total_concat_size += 256

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
            
        return torch.cat(encoded_tensor_list, dim=1)

def make_env(env_path, worker_id=0):
    """Creates a single Unity environment"""
    def _init():
        try:
            env = UnityGCRLLTLWrapper(
                env_path=env_path,
                worker_id=worker_id,
                no_graphics=True
            )
            return env
        except Exception as e:
            print(f"Error creating environment: {str(e)}")
            raise
    return _init

def setup_logging(exp_name):
    """Setup logging directory"""
    log_path = os.path.join('logs', 'ppo', exp_name)
    os.makedirs(log_path, exist_ok=True)
    return log_path

def main(args):
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Setup logging
    log_path = setup_logging(args.exp_name)
    
    try:
        # First test with a single environment
        print("Testing single environment setup...")
        test_env = UnityGCRLLTLWrapper(
            env_path=args.unity_env_path,
            worker_id=0,
            no_graphics=True
        )
        
        # Test environment
        print("Testing environment reset and step...")
        obs = test_env.reset()
        print(f"Observation space: {test_env.observation_space}")
        print(f"Action space: {test_env.action_space}")
        
        # Take a test step
        action = test_env.action_space.sample()
        obs, reward, done, info = test_env.step(action)
        print("Environment test successful!")
        test_env.close()
        
        # Create vectorized environment
        print(f"Creating {args.num_cpus} environments...")
        env = make_vec_env(
            make_env(args.unity_env_path),
            n_envs=args.num_cpus,
            seed=args.seed,
            start_method='spawn'
        )
        
        # Setup PPO model
        print("Setting up PPO model...")
        policy_kwargs = dict(
            activation_fn=nn.ReLU,
            net_arch=[512, 1024, 256],
            features_extractor_class=CustomCombinedExtractor,
        )
        
        model = PPO(
            policy='MultiInputPolicy',
            policy_kwargs=policy_kwargs,
            env=env,
            verbose=1,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_epochs=args.n_epochs,
            n_steps=int(args.n_steps/args.num_cpus),
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            device=device,
        )

        # Setup logger
        new_logger = configure(log_path, ['stdout', 'csv', 'tensorboard'])
        model.set_logger(new_logger)

        # Setup evaluation environment
        eval_env = make_vec_env(
            make_env(args.unity_env_path),
            n_envs=1,
            seed=args.seed + 1000
        )
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=log_path,
            log_path=log_path,
            eval_freq=max(args.eval_freq // args.num_cpus, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        )
        
        # Setup trajectory buffer
        traj_buffer = TrajectoryBuffer(
            traj_length=args.traj_length,
            buffer_size=args.total_timesteps,
            obs_dim=args.obs_dim,
            n_envs=args.num_cpus,
            device=device
        )
        
        traj_callback = CollectTrajectoryCallback(traj_buffer=traj_buffer)
        
        # Combine callbacks
        callback = CallbackList([eval_callback, traj_callback])
        
        # Train model
        print("Starting training...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback
        )
        
        # Save trajectory dataset
        print("Saving trajectory dataset...")
        traj_dataset = traj_buffer.build_dataset(model.policy)
        torch.save(
            traj_dataset,
            os.path.join('datasets', f'{args.exp_name}_traj_dataset.pt')
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'env' in locals():
            env.close()
        if 'eval_env' in locals():
            eval_env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GCRLLTL agent with Unity environment')
    
    # Environment arguments
    parser.add_argument('--unity_env_path', type=str, required=True,
                      help='Path to Unity environment executable')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to train on (cuda/cpu)')
    parser.add_argument('--num_cpus', type=int, default=4,
                      help='Number of parallel environments')
    
    # Training arguments
    parser.add_argument('--total_timesteps', type=int, default=1e7,
                      help='Total timesteps to train for')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.998,
                      help='Discount factor')
    parser.add_argument('--n_epochs', type=int, default=10,
                      help='Number of PPO epochs')
    parser.add_argument('--n_steps', type=int, default=50000,
                      help='Number of steps per environment per update')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='Batch size for PPO updates')
    parser.add_argument('--ent_coef', type=float, default=0.003,
                      help='Entropy coefficient')
    
    # Evaluation arguments
    parser.add_argument('--eval_freq', type=int, default=100000,
                      help='Evaluation frequency')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                      help='Number of evaluation episodes')
    
    # Buffer arguments
    parser.add_argument('--traj_length', type=int, default=1000,
                      help='Length of trajectories to store')
    parser.add_argument('--obs_dim', type=int, default=100,
                      help='Dimension of observations')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=123,
                      help='Random seed')
    parser.add_argument('--exp_name', type=str, default='unity_exp',
                      help='Experiment name')
    parser.add_argument('--timeout', type=int, default=1000,
                      help='Environment timeout')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create necessary directories
    os.makedirs('logs/ppo', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
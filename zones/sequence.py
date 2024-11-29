import argparse
import random
import os
from functools import partial
try:
    import psutil
except ImportError:
    print("Installing psutil...")
    os.system('pip install psutil')
    import psutil

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from envs.unity import UnityGCRLLTLWrapper

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)  # Reduced final dim
        
        base_obs_dim = observation_space.spaces['obs'].shape[0] - 24
        
        # Simplified architecture with proper initialization
        self.net = nn.Sequential(
            nn.Linear(base_obs_dim + 24, 256),  # Process all inputs together
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Initialize with smaller weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, observations):
        # Combine state and goal
        x = torch.cat([
            observations['obs'][:, :-24],  # State
            observations['obs'][:, -24:]   # Goal
        ], dim=1)
        return self.net(x)

def train_on_fixed_sequences(
    unity_env_path: str,
    timesteps_per_sequence: int = 2_000_000,
    num_envs: int = 4,
    device: str = 'cuda'
):
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor,
        features_extractor_kwargs=dict(),
        net_arch=[dict(pi=[128, 64], vf=[128, 64])],
        activation_fn=nn.ReLU
    )
    
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "clip_range_vf": 0.2,
        "ent_coef": 0.1,
        "max_grad_norm": 0.5,
        "policy_kwargs": policy_kwargs
    }

    goal_sequences = [
        ['red', 'green', 'yellow'],
        ['red', 'yellow', 'green'],
        ['green', 'red', 'yellow'],
        ['green', 'yellow', 'red'],
        ['yellow', 'red', 'green'],
        ['yellow', 'green', 'red']
    ]
    
    # Define curriculum stages
    stages = [
        {
            "timesteps": timesteps_per_sequence // 4,
            "ent_coef": 0.05,
            "learning_rate": 3e-4
        },
        {
            "timesteps": timesteps_per_sequence // 4,
            "ent_coef": 0.02,
            "learning_rate": 1e-4
        },
        {
            "timesteps": timesteps_per_sequence // 2,
            "ent_coef": 0.01,
            "learning_rate": 5e-5
        }
    ]
    
    # Base worker ID for each sequence
    base_worker_id = 5000  # Start from a safe port number
    stages = [
    {
        "timesteps": timesteps_per_sequence // 3,
        "learning_rate": 3e-4,
        "ent_coef": 0.01
    },
    {
        "timesteps": timesteps_per_sequence // 3,
        "learning_rate": 1e-4,
        "ent_coef": 0.005
    },
    {
        "timesteps": timesteps_per_sequence // 3,
        "learning_rate": 5e-5,
        "ent_coef": 0.001
    }
    ]   
    
    for sequence_idx, sequence in enumerate(goal_sequences):
        print(f"\nTraining on sequence {sequence_idx + 1}/6: {' -> '.join(sequence)}")
        
        # Calculate worker IDs for this sequence
        sequence_worker_ids = [base_worker_id + sequence_idx * num_envs + i for i in range(num_envs)]
        eval_worker_id = base_worker_id + (len(goal_sequences) * num_envs) + sequence_idx
        
        print(f"Using worker IDs: {sequence_worker_ids}")
        
        try:
            # Create environments with proper worker IDs
            env = SubprocVecEnv([
                make_env(unity_env_path, sequence, worker_id)
                for worker_id in sequence_worker_ids
            ], start_method='spawn')

            model = PPO("MultiInputPolicy", env, verbose=1, **ppo_params)
            for stage_idx, stage in enumerate(stages):
                model.learning_rate = stage["learning_rate"]
                model.ent_coef = stage["ent_coef"]
                print(f"\nStage {stage_idx + 1} - ent_coef: {stage['ent_coef']}, lr: {stage['learning_rate']}")
                
                # Eval environment with unique worker ID
                eval_env = make_env(unity_env_path, sequence, eval_worker_id)()
                
                callbacks = [
                    EvalCallback(
                        eval_env=eval_env,
                        best_model_save_path=f'models/sequence_{sequence_idx}/stage_{stage_idx}',
                        eval_freq=5000 // num_envs,
                        n_eval_episodes=10
                    ),
                    CheckpointCallback(
                        save_freq=20000 // num_envs,
                        save_path=f'models/sequence_{sequence_idx}/stage_{stage_idx}'
                    )
                ]
                
                try:
                    model.learn(
                        total_timesteps=stage["timesteps"],
                        callback=callbacks,
                        reset_num_timesteps=False
                    )
                except Exception as e:
                    print(f"Error in stage {stage_idx}: {str(e)}")
                    eval_env.close()
                    continue
                finally:
                    eval_env.close()

            # Final evaluation
            final_eval_env = make_env(unity_env_path, sequence, eval_worker_id + 1000)()
            try:
                mean_reward = evaluate_model(model, final_eval_env)
                print(f"Final evaluation reward: {mean_reward}")
            finally:
                final_eval_env.close()
                
            # Save final model
            model.save(f'models/sequence_{sequence_idx}_final')

        except Exception as e:
            print(f"Error training sequence {sequence}: {str(e)}")
        finally:
            env.close()
            
    print("\nTraining complete!")

def evaluate_model(model, env, n_eval_episodes=20):
    """Evaluate model performance"""
    episode_rewards = []
    for episode in range(n_eval_episodes):
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards)

def make_env(env_path, sequence, worker_id):
    """Create Unity environment with fixed sequence and proper cleanup"""
    def _init():
        try:
            env = UnityGCRLLTLWrapper(
                env_path=env_path,
                worker_id=worker_id,
                no_graphics=True,
                time_scale=20.0
            )
            success = env.set_fixed_goal_sequence(sequence)
            if not success:
                print(f"Warning: Failed to set sequence for worker {worker_id}")
            return env
        except Exception as e:
            print(f"Error initializing environment with worker {worker_id}: {e}")
            raise
    return _init

def cleanup_unity_processes():
    """Helper function to clean up Unity processes if needed"""
    for proc in psutil.process_iter():
        try:
            if "Unity" in proc.name():
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unity_env_path', type=str, required=True)
    parser.add_argument('--timesteps', type=int, default=2_000_000)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    try:
        # Cleanup any leftover Unity processes
        cleanup_unity_processes()
        
        # Run training
        dataset = train_on_fixed_sequences(
            unity_env_path=args.unity_env_path,
            timesteps_per_sequence=args.timesteps,
            num_envs=args.num_envs,
            device=args.device
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
    finally:
        # Final cleanup
        cleanup_unity_processes()
import argparse
import random
import os
from functools import partial

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

from envs.unity import UnityGCRLLTLWrapper


class CustomExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for the Unity environment"""
    def __init__(self, observation_space):
        # Calculate total observation dimension from Dict space
        total_obs_dim = observation_space.spaces['obs'].shape[0]
        super().__init__(observation_space, features_dim=256)
        
        # Split dimensions for state and goal vector
        self.state_dim = total_obs_dim - 24  # Base observations
        self.goal_dim = 24  # Goal vector dimension
        
        # State processing network
        self.state_net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Goal processing network
        self.goal_net = nn.Sequential(
            nn.Linear(self.goal_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Combined processing network
        self.combined_net = nn.Sequential(
            nn.Linear(256, 256),  # 128 (state) + 128 (goal)
            nn.ReLU(),
            nn.LayerNorm(256)
        )
    def forward(self, observations):
        # Extract observation dict
        obs = observations['obs']
        
        # Split observation into state and goal components
        state = obs[:, :self.state_dim]  # First part is state
        goal = obs[:, -self.goal_dim:]   # Last 24 dimensions are goal vector
        
        # Process state and goal separately
        state_features = self.state_net(state)
        goal_features = self.goal_net(goal)
        
        # Combine and process features
        combined = torch.cat([state_features, goal_features], dim=1)
        return self.combined_net(combined)

def make_env(env_path, sequence, worker_id):
    """Create a Unity environment with fixed sequence and proper initialization"""
    def _init():
        try:
            print(f"\nInitializing environment {worker_id}...")
            
            # Create environment
            env = UnityGCRLLTLWrapper(
                env_path=env_path,
                worker_id=worker_id,
                no_graphics=True
            )
            
            print(f"Setting sequence for env {worker_id}: {sequence}")
            # Set sequence before any reset
            success = env.set_fixed_goal_sequence(sequence)
            
            if not success:
                print(f"Failed to set sequence for env {worker_id}")
                raise RuntimeError("Failed to set goal sequence")
                
            # Now do the first reset with verification
            print("Performing initial reset...")
            obs = env.reset()[0]
            goal_encoding = obs['obs'][2:5]
            
            # Verify goal state
            if np.all(goal_encoding == 0):
                print(f"Warning: Invalid goal state after reset. Retrying...")
                # Try resetting again
                obs = env.reset()[0]
                goal_encoding = obs['obs'][2:5]
                
                if np.all(goal_encoding == 0):
                    raise RuntimeError(f"Failed to initialize valid goal state for env {worker_id}")
            
            print(f"Environment {worker_id} initialized successfully")
            print(f"Initial goal encoding: {goal_encoding}")
            
            return env
        except Exception as e:
            print(f"Error initializing environment {worker_id}: {e}")
            raise
    return _init

def quick_test_sequences(
    unity_env_path: str,
    timesteps: int = 100_000,
    num_envs: int = 4,
    device: str = 'cuda'
):
    """Quick test of training setup with minimal sequences"""
    
    # Test sequences - one simple, one complex
    test_sequences = [
        ['green', 'red', 'yellow']
    ]
    
    
    results = {}
    
    print("\nStarting quick test training...")
    print(f"Training each sequence for {timesteps} timesteps")
    
    for sequence_idx, sequence in enumerate(test_sequences):
        sequence_name = '->'.join(sequence)
        print(f"\n{'='*50}")
        print(f"Testing sequence {sequence_idx + 1}/4: {sequence_name}")
        print(f"{'='*50}")
        env = SubprocVecEnv([
            make_env(unity_env_path, sequence, sequence_idx+ 5000)
        ], start_method='spawn')
        
        policy_kwargs = dict(
            features_extractor_class=CustomExtractor,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.ReLU
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,  # Add entropy coefficient
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs/unity_training",
            device=device,
            verbose=1
        )
        
        # Setup evaluation callback
        eval_callback = EvalCallback(
            env,
            best_model_save_path=f'models/test/sequence_{sequence_idx}',
            log_path=f'logs/test/sequence_{sequence_idx}',
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True
        )
        
        try:
            # Train
            print(f"\nTraining sequence: {sequence_name}")
            model.learn(
                total_timesteps=timesteps,
                callback=eval_callback,
                progress_bar=True
            )
            
            # Evaluate
            print("\nEvaluating trained model...")
            eval_env = make_env(unity_env_path, sequence, 999)()
            successes = 0
            total_rewards = []
            movement_scores = []
            test_episodes = 10
            
            for episode in range(test_episodes):
                obs = eval_env.reset()[0]
                episode_reward = 0
                positions = []
                done = False
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    positions.append(info.get('position', [0, 0]))
                    
                    if info.get('sequence_complete', False):
                        successes += 1
                        break
                
                total_rewards.append(episode_reward)
                
                # Calculate movement score
                if len(positions) > 1:
                    total_movement = sum(np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1])) 
                                      for i in range(1, len(positions)))
                    movement_scores.append(total_movement)
            
            # Record results
            success_rate = successes / test_episodes
            avg_reward = np.mean(total_rewards)
            avg_movement = np.mean(movement_scores) if movement_scores else 0
            
            results[sequence_name] = {
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_movement': avg_movement
            }
            
            # Print sequence results
            print(f"\nResults for sequence {sequence_name}:")
            print(f"Success rate: {success_rate * 100:.1f}%")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average movement: {avg_movement:.2f}")
            
            # Save test model if promising
            if success_rate >= 0.3:
                model.save(f'models/test/promising_sequence_{sequence_idx}')
                print(f"\nPromising results - model saved!")
            
            eval_env.close()
            
        finally:
            env.close()
    
    # Print overall assessment
    print("\n" + "="*50)
    print("QUICK TEST RESULTS SUMMARY")
    print("="*50)
    
    overall_promising = False
    for sequence_name, result in results.items():
        print(f"\nSequence: {sequence_name}")
        print(f"Success Rate: {result['success_rate']*100:.1f}%")
        print(f"Avg Reward: {result['avg_reward']:.2f}")
        print(f"Movement Score: {result['avg_movement']:.2f}")
        
        if result['success_rate'] >= 0.3 or result['avg_movement'] > 1.0:
            overall_promising = True
    
    print("\nOVERALL ASSESSMENT:")
    if overall_promising:
        print("✅ Results are promising - proceed with full training")
        print("Recommended next steps:")
        print("1. Run medium training (500k steps): python train.py --mode medium")
        print("2. If still good, proceed to full training: python train.py --mode full")
    else:
        print("❌ Results need improvement before full training")
        print("Recommended adjustments:")
        print("1. Check reward scaling in Unity environment")
        print("2. Verify observation space")
        print("3. Try adjusting learning rate or network architecture")
        print("4. Ensure goals are properly set in environment")
    
    return results

def verify_environment(env_path):
    """Verify environment setup with detailed checks"""
    print("\nVerifying environment setup...")
    
    env = UnityGCRLLTLWrapper(env_path, worker_id=9999, no_graphics=True)
    
    # Test goal sequence
    test_sequence = ['red', 'green', 'yellow']
    print("\nSetting goal sequence...")
    success = env.set_fixed_goal_sequence(test_sequence)
    if not success:
        print("❌ Failed to set goal sequence")
        env.close()
        return False
        
    # Get initial observation
    print("\nGetting initial observation...")
    obs = env.reset()[0]
    
    # Verify observation structure
    print("\nObservation verification:")
    print(f"Shape: {obs['obs'].shape}")
    print(f"Position: {obs['obs'][:2]}")
    print(f"Goal encoding: {obs['obs'][2:5]}")
    
    # Check if goal is properly set
    goal_encoding = obs['obs'][2:5]
    expected_encoding = [1, 0, 0]  # First goal should be 'red'
    if not np.allclose(goal_encoding, expected_encoding):
        print("❌ Goal not properly encoded in observation")
        print(f"Expected: {expected_encoding}")
        print(f"Got: {goal_encoding}")
        env.close()
        return False
    
    # Test movements
    print("\nTesting movements...")
    directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    start_pos = obs['obs'][:2]
    
    for i, direction in enumerate(directions):
        action = i + 1  # Actions 1-4 are movements
        next_obs, reward, done, _, info = env.step(action)
        new_pos = next_obs['obs'][:2]
        
        movement = new_pos - start_pos
        print(f"\n{direction} movement:")
        print(f"  Position change: {movement}")
        print(f"  Reward: {reward}")
        
        # Very important - let's see if goal encoding changes
        print(f"  Goal encoding: {next_obs['obs'][2:5]}")
        
        start_pos = new_pos
    
    print("\nVerification complete!")
    env.close()
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unity_env_path', type=str, required=True,
                      help='Path to Unity environment executable')
    parser.add_argument('--timesteps', type=int, default=100,
                      help='Number of timesteps for quick test')
    parser.add_argument('--num_envs', type=int, default=4,
                      help='Number of parallel environments')
    parser.add_argument('--device', type=str, 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to train on (cuda/cpu)')
    parser.add_argument('--verify', action='store_true', 
                      help='Run environment verification')
    args = parser.parse_args()
    
    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create test directories
    os.makedirs('models/test', exist_ok=True)
    os.makedirs('logs/test', exist_ok=True)
    
    try:
        if args.verify:
            success = verify_environment(args.unity_env_path)
            if not success:
                print("\n❌ Environment verification failed!")
                print("Please fix the issues before training")
            else:
                print("\n✅ Environment verification passed!")
                print("You can proceed with training")
        else:
            results = quick_test_sequences(
                unity_env_path=args.unity_env_path,
                timesteps=args.timesteps,
                num_envs=args.num_envs,
                device=args.device
            )
    except KeyboardInterrupt:
        print("\nQuick test interrupted by user")
    except Exception as e:
        print(f"\nQuick test failed with error: {str(e)}")
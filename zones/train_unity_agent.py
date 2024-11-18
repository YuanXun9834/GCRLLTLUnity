import torch
import os
import random
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from torch import nn
from envs.unity import UnityGCRLLTLWrapper
from rl.traj_buffer import TrajectoryBuffer
from rl.callbacks import CollectTrajectoryCallback

class CustomExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for the Unity environment"""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        
        # Get the total observation dimension (including goal vector)
        base_obs_dim = observation_space.spaces['obs'].shape[0] - 24  # Subtract goal vector
        
        self.obs_net = nn.Sequential(
            nn.Linear(base_obs_dim, 128),  # 15 base dims
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.goal_net = nn.Sequential(
            nn.Linear(24, 64),  # Process goal vector
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.combined_net = nn.Sequential(
            nn.Linear(256, 256),  # 128 (obs_net) + 128 (goal_net)
            nn.ReLU(),
        )
    def forward(self, observations):
        obs = observations['obs']
        state = obs[:, :-24]  # Base observation (15 dims)
        goal = obs[:, -24:]   # Goal vector (24 dims)

        state_features = self.obs_net(state)
        goal_features = self.goal_net(goal)
        
        return self.combined_net(torch.cat([state_features, goal_features], dim=1))

def create_env_with_id(unity_env_path, worker_id, goal_sequence=None):
    """Create Unity environment with specific worker ID"""
    def _init():
        env = UnityGCRLLTLWrapper(
            env_path=unity_env_path,
            worker_id=worker_id,
            no_graphics=True,
            time_scale=20.0,
            max_steps=1000
        )
        if goal_sequence:
            env.fix_goal(goal_sequence[0])
        return env
    return _init

def train_goal_conditioned_agent(
    unity_env_path,
    total_timesteps,
    num_envs=4,
    device='cuda',
    seed=0,
    save_freq=10000
):
    """Train a goal-conditioned agent and collect trajectories"""
    
    print(f"Starting training with {num_envs} environments...")
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)
    
    # Create vectorized environment
    env_fns = [create_env_with_id(unity_env_path, i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns, start_method='spawn')
    
    # Create eval environment
    eval_env = SubprocVecEnv(
        [create_env_with_id(unity_env_path, num_envs + i) for i in range(2)],
        start_method='spawn'
    )

    # Initialize trajectory buffer
    traj_buffer = TrajectoryBuffer(
        traj_length=1000,
        buffer_size=total_timesteps,
        obs_dim=env.observation_space['obs'].shape[0],
        n_envs=num_envs,
        device=device
    )

    try:
        # Create PPO model with custom policy
        policy_kwargs = dict(
            features_extractor_class=CustomExtractor,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=nn.ReLU
        )

        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs/unity_training",
            device=device,
            verbose=1
        )

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path="models/checkpoints/",
            name_prefix="unity_model"
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="models/best_model",
            log_path="logs/eval",
            eval_freq=5000,
            n_eval_episodes=10,
            deterministic=True
        )

        traj_callback = CollectTrajectoryCallback(traj_buffer=traj_buffer)

        # Train model
        print("Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, traj_callback],
            progress_bar=True
        )

        # Save final model
        model_path = os.path.join("models", "final_model")
        model.save(model_path)
        print(f"Final model saved to {model_path}")

        # Build and save trajectory dataset
        print("Building dataset from collected trajectories...")
        trajectory_dataset = traj_buffer.build_dataset(model.policy)

        dataset_path = os.path.join("datasets", "unity_trajectory_dataset.pt")
        torch.save(trajectory_dataset, dataset_path)
        print(f"Trajectory dataset saved to {dataset_path}")

        return model, trajectory_dataset

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity_env_path", type=str, required=True,
                      help="Path to Unity environment executable")
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                      help="Total timesteps to train")
    parser.add_argument("--num_envs", type=int, default=4,
                      help="Number of parallel environments")
    parser.add_argument("--device", type=str,
                      default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed")
    parser.add_argument("--save_freq", type=int, default=10000,
                      help="Save frequency for checkpoints")

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/best_model", exist_ok=True)
    os.makedirs("logs/unity_training", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)
    os.makedirs("datasets", exist_ok=True)

    try:
        train_goal_conditioned_agent(
            unity_env_path=args.unity_env_path,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            save_freq=args.save_freq
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
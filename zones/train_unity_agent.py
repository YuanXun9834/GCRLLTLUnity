import torch
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from envs.unity import UnityGCRLLTLWrapper
from rl.traj_buffer import TrajectoryBuffer
from rl.callbacks import CollectTrajectoryCallback

def create_env_with_id(unity_env_path, worker_id):
    """Create Unity environment with specific worker ID"""
    def _init():
        return UnityGCRLLTLWrapper(
            env_path=unity_env_path,
            worker_id=worker_id,
            no_graphics=True,  # Set to True for training
            max_steps=100
        )
    return _init

def train_goal_conditioned_agent(unity_env_path, total_timesteps, num_envs, device):
    print(f"Starting training for {total_timesteps} timesteps...")
    
    env_fns = [create_env_with_id(unity_env_path, i) for i in range(num_envs)]
    env = SubprocVecEnv(env_fns, start_method='spawn')
    
    # Get the correct observation dimension from the environment
    obs_dim = env.observation_space['obs'].shape[0]
    print(f"Observation dimension: {obs_dim}")
    
    # Setup logging
    log_dir = "logs/training_progress"
    os.makedirs(log_dir, exist_ok=True)
    
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device=device
    )
    
    # Initialize trajectory buffer with correct observation dimension
    traj_buffer = TrajectoryBuffer(
        traj_length=1000,
        buffer_size=total_timesteps,
        obs_dim=obs_dim,  # Use the actual observation dimension
        n_envs=num_envs,
        device=device
    )
    
    try:
        print("Training model...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=CollectTrajectoryCallback(traj_buffer),
            progress_bar=True
        )
        
        print("Building trajectory dataset...")
        trajectory_dataset = traj_buffer.build_dataset(model.policy)
        
        # Save model and dataset
        os.makedirs("models", exist_ok=True)
        os.makedirs("datasets", exist_ok=True)
        
        model_path = os.path.join("models", "trained_model")
        dataset_path = os.path.join("datasets", "trajectory_dataset.pt")
        
        print(f"Saving model to {model_path}")
        model.save(model_path)
        
        print(f"Saving trajectory dataset to {dataset_path}")
        torch.save(trajectory_dataset, dataset_path)
        
        print("Training completed successfully!")
        print(f"Total episodes completed: {model.num_timesteps}")
        print(f"Total trajectories collected: {len(trajectory_dataset)}")
        
        return model, trajectory_dataset
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        env.close()
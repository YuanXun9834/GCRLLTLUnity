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
    
    # Get the correct observation dimension
    obs_dim = env.observation_space['obs'].shape[0]
    print(f"Observation dimension: {obs_dim}")
    
    # Calculate appropriate buffer size
    buffer_size = max(total_timesteps, 2048 * num_envs)
    print(f"Buffer size: {buffer_size}")
    
    # Initialize trajectory buffer
    traj_buffer = TrajectoryBuffer(
        traj_length=1000,
        buffer_size=buffer_size,
        obs_dim=obs_dim,
        n_envs=num_envs,
        device=device
    )
    
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log="logs",
            device=device
        )
        
        print("Starting training...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=CollectTrajectoryCallback(traj_buffer),
            progress_bar=True
        )
        
        # Save trajectory dataset
        print("Building dataset from collected trajectories...")
        trajectory_dataset = traj_buffer.build_dataset(model.policy)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Get absolute path of script
        
        # Create directories
        datasets_dir = os.path.join(base_dir, "datasets")
        models_dir = os.path.join(base_dir, "models")
        
        os.makedirs(datasets_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Save with absolute paths
        model_path = os.path.join(models_dir, "trained_model")
        dataset_path = os.path.join(datasets_dir, "trajectory_dataset.pt")
        
        print(f"\nDirectory Information:")
        print(f"Base directory: {base_dir}")
        print(f"Datasets directory: {datasets_dir}")
        print(f"Dataset will be saved to: {dataset_path}")
        
        # When saving the dataset
        if len(trajectory_dataset.states) > 0:
            print(f"\nSaving dataset:")
            print(f"Number of states: {len(trajectory_dataset.states)}")
            print(f"Number of goal values: {len(trajectory_dataset.goal_values)}")
            print(f"Saving to: {dataset_path}")
            torch.save(trajectory_dataset, dataset_path)
            
            # Verify save
            if os.path.exists(dataset_path):
                print(f"Dataset saved successfully! File size: {os.path.getsize(dataset_path)} bytes")
            else:
                print("Warning: Dataset file not found after saving!")
        else:
            print("\nWarning: No trajectories collected! Dataset will not be saved.")
        
        return model, trajectory_dataset
            
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        env.close()
import torch
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
    # Create list of env functions with unique worker IDs
    env_fns = [create_env_with_id(unity_env_path, i) for i in range(num_envs)]
    
    # Create vectorized environment with seed handling disabled
    env = SubprocVecEnv(env_fns, start_method='spawn')
    
    # Initialize PPO model with appropriate parameters
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=device,
        n_steps=512,  # Adjust based on your needs
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.01,
    )
    
    # Initialize trajectory buffer
    traj_buffer = TrajectoryBuffer(
        traj_length=1000,
        buffer_size=total_timesteps,
        obs_dim=100,  # Adjust based on your observation space
        n_envs=num_envs,
        device=device
    )
    
    # Setup callback
    callback = CollectTrajectoryCallback(traj_buffer=traj_buffer)
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )
    
    # Build dataset from collected trajectories
    trajectory_dataset = traj_buffer.build_dataset(model.policy)
    
    return model, trajectory_dataset
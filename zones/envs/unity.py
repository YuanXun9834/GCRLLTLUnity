import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, Any
import atexit
import signal
import sys

# ML-Agents imports
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

class UnityGCRLLTLWrapper(gym.Env):
    def __init__(
        self,
        env_path: str,
        worker_id: int = 0,
        no_graphics: bool = True,
        time_scale: float = 20.0,
        seed: int = 0,
        max_steps: int = 100
    ):
        super().__init__()
        
        self.env_path = env_path
        self.worker_id = worker_id
        self.unity_env = None
        self.channel = EngineConfigurationChannel()
        
        try:
            print(f"Initializing Unity environment {self.worker_id}")
            # Create Unity environment
            self.unity_env = UnityEnvironment(
                file_name=env_path,
                worker_id=worker_id,
                no_graphics=no_graphics,
                seed=seed,
                side_channels=[self.channel],
                timeout_wait=60
            )
            
            # Set time scale and resolution only
            self.channel.set_configuration_parameters(
                time_scale=time_scale,
                width=640,
                height=480,
                quality_level=0
            )
            
            # Initialize goal vectors first
            self.goals_representation = self._initialize_goal_vectors()
            
            # Initialize environment
            print(f"Resetting Unity environment {self.worker_id}")
            self.unity_env.reset()
            
            # Get behavior specs
            self.behavior_names = list(self.unity_env.behavior_specs.keys())
            if not self.behavior_names:
                raise ValueError("No behaviors found in Unity environment")
            
            print(f"Available behaviors: {self.behavior_names}")
            self.behavior_name = self.behavior_names[0]
            self.spec = self.unity_env.behavior_specs[self.behavior_name]
            
            print(f"Using behavior: {self.behavior_name}")
            print(f"Behavior spec: {self.spec}")
            
            # Setup spaces
            self._setup_spaces()
            
            # Initialize state
            self.current_goal = None
            self.steps = 0
            self.goal_achieved = False
            self.last_obs = None
            self.max_steps = max_steps
            
            print(f"Unity environment {self.worker_id} initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Unity environment: {str(e)}")
            if self.unity_env is not None:
                self.unity_env.close()
            raise

    def _initialize_goal_vectors(self):
        """Initialize goal vectors for each zone type"""
        print("Initializing goal vectors...")
        zone_vectors = {}
        
        # Define base vectors for each goal type
        base_vectors = {
            'green': np.array([1, 0, 0]),  # For GreenPlus
            'red': np.array([0, 1, 0]),    # For RedEx
            'yellow': np.array([0, 0, 1])  # For YellowStar
        }
        
        # Create 24-dimensional vectors (8 sets of 3D vectors)
        for zone, base in base_vectors.items():
            # Repeat the base vector 8 times
            zone_vectors[zone] = np.tile(base, 8)
            # Normalize the vector
            zone_vectors[zone] = zone_vectors[zone] / np.linalg.norm(zone_vectors[zone])
            print(f"Created vector for {zone} zone with shape {zone_vectors[zone].shape}")
            
        return zone_vectors

    def _setup_spaces(self):
        """Setup observation and action spaces"""
        print("Setting up observation and action spaces...")
        
        # Get observation specs from Unity
        obs_shapes = 0
        print(f"Observation specs: {self.spec.observation_specs}")
        
        for obs_spec in self.spec.observation_specs:
            if len(obs_spec.shape) == 1:  # Only use vector observations
                obs_shapes += obs_spec.shape[0]
        
        print(f"Total observation dimension: {obs_shapes}")
        
        # Add space for goal vector
        goal_dim = 24  # 8 sets of 3D vectors
        
        # Set up observation space
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_shapes + goal_dim,),
                dtype=np.float32
            ),
            'success': spaces.Box(low=0, high=1, shape=(1,)),
            'steps': spaces.Box(low=0, high=np.inf, shape=(1,))
        })
        
        print(f"Observation space: {self.observation_space}")
        
        # Set up action space based on Unity spec
        if self.spec.action_spec.continuous_size > 0:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.spec.action_spec.continuous_size,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(
                self.spec.action_spec.discrete_branches[0]
            )
            
        print(f"Action space: {self.action_space}")

    def _process_obs(self, steps):
        """Process Unity observation steps into numpy array"""
        obs_list = []
        
        # Detailed debug info
        print("\nProcessing observations:")
        print(f"Steps type: {type(steps)}")
        print(f"Steps attributes: {dir(steps)}")
        print(f"Number of agents: {len(steps)}")
        
        # Check if we have any observations
        if len(steps.obs) == 0:
            print("Warning: Empty observation list from Unity")
            print(f"Full steps object: {steps}")
            return np.zeros(38)
            
        for i, obs in enumerate(steps.obs):
            print(f"\nObservation {i}:")
            print(f"Shape: {obs.shape}")
            print(f"Name: {self.spec.observation_specs[i].name}")
            print(f"Values: {obs[0][:5]}")  # Print first 5 values
            
            if len(obs.shape) == 1:  # Vector observations
                if np.any(np.isnan(obs)):
                    print(f"Warning: NaN values in observation {i}")
                    continue
                if np.any(np.isinf(obs)):
                    print(f"Warning: Inf values in observation {i}")
                    continue
                print(f"Adding vector observation {i}")
                obs_list.append(obs[0])
            elif len(obs.shape) == 3:  # Visual observations
                if np.any(np.isnan(obs)):
                    print(f"Warning: NaN values in visual observation {i}")
                    continue
                print(f"Adding visual observation {i}")
                flat_visual = obs[0].reshape(-1)
                obs_list.append(flat_visual)

        if not obs_list:
            print("Warning: No valid observations after processing")
            return np.zeros(38)

        try:
            concatenated = np.concatenate(obs_list)
            print(f"Successfully concatenated observations, shape: {concatenated.shape}")
            return concatenated
        except Exception as e:
            print(f"Error concatenating observations: {e}")
            return np.zeros(38)

    def _get_obs(self, obs):
        """Convert Unity observation to GCRLLTL format"""
        if self.current_goal is None:
            goal_vec = np.zeros(24)
        else:
            goal_vec = self.goals_representation[self.current_goal]
            
        # Make sure obs is the right shape
        if isinstance(obs, np.ndarray):
            processed_obs = obs
        else:
            print(f"Warning: Invalid observation type: {type(obs)}")
            processed_obs = np.zeros(38)
        
        return {
            'obs': np.concatenate([processed_obs, goal_vec]).astype(np.float32),
            'success': np.array([float(self.goal_achieved)], dtype=np.float32),
            'steps': np.array([self.steps], dtype=np.float32)
        }

    def _check_goal_achievement(self, obs):
        """Check if current goal is achieved"""
        # Implement based on your Unity environment's goal criteria
        # For now, return False
        return False

    def step(self, action):
        """Execute step in environment"""
        if self.unity_env is None:
            raise RuntimeError("Unity environment is not initialized")
                
        try:
            # Convert action to Unity format
            if isinstance(self.action_space, spaces.Box):
                unity_action = ActionTuple(continuous=np.array([action]))
            else:
                unity_action = ActionTuple(discrete=np.array([[action]]))
            
            # Set action in Unity
            self.unity_env.set_actions(self.behavior_name, unity_action)
            self.unity_env.step()
            
            # Get result
            decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
            
            # Check if episode ended
            terminated = len(terminal_steps) > 0
            truncated = False  # Set to True if episode is truncated due to time limit
            
            # Get observation and reward
            if terminated:
                obs = self._process_obs(terminal_steps)
                reward = terminal_steps.reward[0]
            else:
                obs = self._process_obs(decision_steps)
                reward = decision_steps.reward[0]
                
            self.steps += 1
            self.last_obs = obs
            
            # Check goal achievement
            self.goal_achieved = self._check_goal_achievement(obs)
            
            info = {
                'goal_achieved': self.goal_achieved,
                'steps': self.steps
            }
            
            truncated = self.steps >= self.max_steps
            if truncated:
                terminated = True
            
            return self._get_obs(obs), reward, terminated, truncated, info
                
        except Exception as e:
            print(f"Error during step: {str(e)}")
            raise

    def reset(self, seed=None, options=None):
        """Reset environment with optional seed and options"""
        if self.unity_env is None:
            raise RuntimeError("Unity environment is not initialized")
                
        try:
            # Set seed if provided (Unity handles seeding internally)
            if seed is not None:
                self.unity_env.seed = seed
                
            self.unity_env.reset()
            decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
                
            self.steps = 0
            self.goal_achieved = False
            
            # Print debug info
            print(f"Reset - Decision steps shape: {decision_steps.obs[0].shape}")
                
            # Get initial observation
            obs = self._process_obs(decision_steps)
            self.last_obs = obs
                
            formatted_obs = self._get_obs(obs)
            print(f"Reset - Formatted observation shapes: {[(k, v.shape) for k, v in formatted_obs.items()]}")
                
            return formatted_obs, {}
                
        except Exception as e:
            print(f"Error during reset: {str(e)}")
            raise

    def close(self):
        """Close Unity environment"""
        if self.unity_env is not None:
            try:
                print(f"Closing Unity environment {self.worker_id}")
                self.unity_env.close()
                self.unity_env = None
            except Exception as e:
                print(f"Error closing Unity environment: {str(e)}")

    def fix_goal(self, goal: str):
        """Set current goal"""
        if goal not in self.goals_representation:
            raise ValueError(f"Unknown goal: {goal}. Available goals: {list(self.goals_representation.keys())}")
        self.current_goal = goal
        self.goal_achieved = False
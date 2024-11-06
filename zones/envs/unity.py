import gym
from gym import spaces
import numpy as np
from typing import Dict, Optional, Any, Tuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from stable_baselines3 import PPO

class UnityGCRLLTLWrapper(gym.Env):
    """Unity environment wrapper for GCRLLTL"""
    def __init__(
        self,
        env_path: str,
        worker_id: int = 0,
        no_graphics: bool = True,
        time_scale: float = 20.0,
        seed: int = 0,
        use_primitives: bool = False  # Added but not used in this version
    ):
        super().__init__()
        
        # Create Unity environment
        self.channel = EngineConfigurationChannel()
        self.unity_env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            no_graphics=no_graphics,
            seed=seed,
            side_channels=[self.channel]
        )
        self.channel.set_configuration_parameters(time_scale=time_scale)
        
        # Connect to Unity environment
        self.unity_env.reset()
        self.behavior_name = list(self.unity_env.behavior_specs)[0]
        self.spec = self.unity_env.behavior_specs[self.behavior_name]
        
        # Set up observation and action spaces
        self._setup_spaces()
        
        # Initialize state variables
        self.current_goal = None
        self.steps = 0
        self.goal_achieved = False
        self.last_obs = None
        self.goals_representation = self._initialize_goal_vectors()
        
    def _initialize_goal_vectors(self) -> Dict[str, np.ndarray]:
        """Initialize goal vectors for each zone type"""
        zone_vectors = {}
        base_vectors = {
            'green': np.array([1, 0, 0]),
            'red': np.array([0, 1, 0]),
            'yellow': np.array([0, 0, 1])
        }
        
        for zone, base in base_vectors.items():
            zone_vectors[zone] = np.tile(base, 8)
            zone_vectors[zone] = zone_vectors[zone] / np.linalg.norm(zone_vectors[zone])
            
        return zone_vectors
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Get observation specs from Unity
        obs_shapes = 0
        for obs_spec in self.spec.observation_specs:
            if len(obs_spec.shape) == 1:  # Only use vector observations
                obs_shapes += obs_spec.shape[0]
        
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
        
        # Set up action space based on Unity spec
        if self.spec.action_spec.continuous_size > 0:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.spec.action_spec.continuous_size,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(self.spec.action_spec.discrete_size)
    
    def reset(self):
        """Reset environment"""
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        
        self.steps = 0
        self.goal_achieved = False
        
        # Get initial observation
        obs = self._process_obs(decision_steps)
        self.last_obs = obs
        
        return self._get_obs(obs)
    
    def step(self, action):
        """Execute step in environment"""
        # Convert action to Unity format if needed
        if isinstance(self.action_space, spaces.Box):
            unity_action = action
        else:
            unity_action = np.array([action], dtype=np.int32)
            
        # Set action in Unity
        self.unity_env.set_actions(self.behavior_name, unity_action)
        self.unity_env.step()
        
        # Get result
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        
        # Check if episode ended
        done = len(terminal_steps.agent_id) > 0
        
        # Get observation and reward
        if done:
            obs = self._process_obs(terminal_steps)
            reward = terminal_steps.reward[0]
        else:
            obs = self._process_obs(decision_steps)
            reward = decision_steps.reward[0]
            
        self.steps += 1
        self.last_obs = obs
        
        # Check goal achievement (implement based on your Unity environment)
        self.goal_achieved = self._check_goal_achievement(obs)
        
        info = {
            'goal_achieved': self.goal_achieved
        }
        
        return self._get_obs(obs), reward, done, info
    
    def _process_obs(self, steps):
        """Process Unity observation steps into numpy array"""
        obs = []
        for o in steps.obs:
            if len(o.shape) == 1:  # Only use vector observations
                obs.append(o[0])
        return np.concatenate(obs)
    
    def _get_obs(self, obs):
        """Convert Unity observation to GCRLLTL format"""
        if self.current_goal is None:
            goal_vec = np.zeros(24)  # 8 sets of 3D vectors
        else:
            goal_vec = self.goals_representation[self.current_goal]
            
        return {
            'obs': np.concatenate([obs, goal_vec]).astype(np.float32),
            'success': np.array([float(self.goal_achieved)], dtype=np.float32),
            'steps': np.array([self.steps], dtype=np.float32)
        }
    
    def _check_goal_achievement(self, obs):
        """Implement goal achievement check based on your Unity environment"""
        # This should be implemented based on your specific Unity environment
        # For now, return False
        return False
    
    def fix_goal(self, goal: str):
        """Set current goal"""
        assert goal in self.goals_representation, f"Unknown goal: {goal}"
        self.current_goal = goal
        self.goal_achieved = False
    
    def close(self):
        """Close Unity environment"""
        if self.unity_env:
            self.unity_env.close()

    def render(self, mode='human'):
        """Implement if needed"""
        pass
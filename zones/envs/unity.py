import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import logging
import os
from datetime import datetime


os.makedirs('logs/debug', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/debug/unity_env_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class UnityGCRLLTLWrapper(gym.Env):
    def __init__(
        self,
        env_path: str,
        worker_id: int = 0,
        no_graphics: bool = True,
        time_scale: float = 20.0,
        seed: int = 0,
        max_steps: int = 1000
    ):
        super().__init__()
        self.logger = logging.getLogger(f"UnityEnv-{worker_id}")
        self.logger.info(f"Initializing Unity environment with worker ID {worker_id}")
        self.env_path = env_path
        self.worker_id = worker_id
        self.unity_env = None
        self.channel = EngineConfigurationChannel()
        self.max_steps = max_steps
        
        # Initialize Unity environment
        print(f"Initializing Unity environment {worker_id}")
        self.unity_env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            no_graphics=no_graphics,
            seed=seed,
            side_channels=[self.channel],
            timeout_wait=60
        )
        
        self.channel.set_configuration_parameters(
            time_scale=time_scale,
            width=640,
            height=480,
            quality_level=0
        )
        
        # Initialize environment
        self.unity_env.reset()
        self.behavior_name = list(self.unity_env.behavior_specs.keys())[0]
        self.spec = self.unity_env.behavior_specs[self.behavior_name]
        
        # Initialize spaces
        self._setup_spaces()
        
        # State tracking
        self.current_goal = None
        self.steps = 0
        self.has_powerup = False
        self.achieved_goals = set()
        
        # Goal representations
        self.goals_representation = {
            'green': np.tile([1, 0, 0], 8),
            'red': np.tile([0, 1, 0], 8),
            'yellow': np.tile([0, 0, 1], 8)
        }
        
        # Normalize goal vectors
        for goal in self.goals_representation:
            self.goals_representation[goal] = self.goals_representation[goal] / np.linalg.norm(self.goals_representation[goal])

    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Base observation size: agent position (2) + goal one-hot (3) + powerup (1) + achieved goals (3) + goal positions (6)
        base_obs_size = 15  # 2 + 3 + 1 + 3 + 6
        goal_dim = 24
        
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(base_obs_size + goal_dim,),
                dtype=np.float32
            ),
            'success': spaces.Box(low=0, high=1, shape=(1,)),
            'steps': spaces.Box(low=0, high=np.inf, shape=(1,))
        })
        
        self.action_space = spaces.Discrete(5)

    def _process_obs(self, steps):
        """Process Unity observation into numpy array"""
        if len(steps) == 0:
            return np.zeros(15)
            
        # Get only the vector observation (index 1)
        vector_obs = steps.obs[1][0]  # Shape: (15,)
        
        # Extract the components we need from the vector observation
        agent_pos = vector_obs[0:2]  # First two values are position
        goal_onehot = np.zeros(3)
        if self.current_goal == 'green':
            goal_onehot[0] = 1
        elif self.current_goal == 'red':
            goal_onehot[1] = 1
        elif self.current_goal == 'yellow':
            goal_onehot[2] = 1
            
        # Powerup status (1)
        powerup = np.array([float(self.has_powerup)])
        
        # Achieved goals (3)
        achieved = np.zeros(3)
        if 'green' in self.achieved_goals:
            achieved[0] = 1
        if 'red' in self.achieved_goals:
            achieved[1] = 1
        if 'yellow' in self.achieved_goals:
            achieved[2] = 1
                
        # Goal positions from observation (6 values: 3 goals x 2 coordinates)
        goal_positions = vector_obs[2:8]
        
        # Concatenate all components
        final_obs = np.concatenate([
            agent_pos.flatten(),      # 2
            goal_onehot.flatten(),    # 3
            powerup.flatten(),        # 1
            achieved.flatten(),       # 3
            goal_positions.flatten()  # 6
        ])                           # Total: 15
        
        return final_obs.astype(np.float32)

    def _get_obs(self, obs):
        """Convert processed observation to GCRL format"""
        goal_vec = self.goals_representation[self.current_goal] if self.current_goal else np.zeros(24)
        
        return {
            'obs': np.concatenate([obs, goal_vec]).astype(np.float32),
            'success': np.array([float(self._check_goal_achievement(obs))], dtype=np.float32),
            'steps': np.array([self.steps], dtype=np.float32)
        }

    def _check_goal_achievement(self, obs):
        """Check if current goal is achieved based on proximity"""
        if self.current_goal is None:
            return False
            
        agent_pos = obs[:2]
        goal_positions = {
            'red': obs[8:10],
            'yellow': obs[10:12],
            'green': obs[12:14]
        }
        
        goal_pos = goal_positions[self.current_goal]
        distance = np.linalg.norm(agent_pos - goal_pos)
        
        return distance < 0.4

    def step(self, action):
        """Execute action in environment"""
        unity_action = ActionTuple(discrete=np.array([[action]]))
        
        self.unity_env.set_actions(self.behavior_name, unity_action)
        self.unity_env.step()
        
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        
        terminated = len(terminal_steps) > 0
        truncated = self.steps >= self.max_steps
        
        if terminated:
            obs = self._process_obs(terminal_steps)
            reward = terminal_steps.reward[0]
        else:
            obs = self._process_obs(decision_steps)
            reward = decision_steps.reward[0]
        
        self.steps += 1
        
        # Check goal achievement and update state
        goal_achieved = self._check_goal_achievement(obs)
        if goal_achieved:
            if self.current_goal == 'green':
                self.has_powerup = True
                reward += 2.0
            reward += 1.0
            self.achieved_goals.add(self.current_goal)
        
        reward -= 0.01  # Small step penalty
        
        if goal_achieved and self.has_powerup and self.current_goal != 'green':
            reward += 2.0
        
        info = {
            'goal_achieved': goal_achieved,
            'has_powerup': self.has_powerup,
            'achieved_goals': self.achieved_goals
        }
        
        return self._get_obs(obs), reward, terminated or truncated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            self.unity_env.seed = seed
            
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        
        self.steps = 0
        self.has_powerup = False
        self.achieved_goals.clear()
        
        obs = self._process_obs(decision_steps)
        
        # Set initial goal if none is set
        if self.current_goal is None:
            self.current_goal = 'green'  # Default to green as first goal
            
        return self._get_obs(obs), {}

    def close(self):
        """Clean up environment"""
        if self.unity_env is not None:
            self.unity_env.close()
            self.unity_env = None

    def fix_goal(self, goal: str):
        """Set current goal"""
        if goal not in self.goals_representation:
            raise ValueError(f"Unknown goal: {goal}. Available goals: {list(self.goals_representation.keys())}")
        self.current_goal = goal
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import logging
import uuid
import os
import random
from datetime import datetime
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage
)

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
class GoalSequenceChannel(SideChannel):
    def __init__(self, worker_id):
        # Convert worker_id into UUID bytes in ML-Agents format
        hex_id = format(worker_id + 5000, '032x')  # Pad to 32 hex chars
        channel_id = uuid.UUID(hex_id)
        super().__init__(channel_id)
        self.worker_id = worker_id
        print(f"Created channel for worker {worker_id} with ID: {channel_id}")
        
    def on_message_received(self, msg: IncomingMessage) -> None:
        """Handle any incoming messages from Unity"""
        pass

    def send_sequence(self, sequence):
        """Send goal sequence to Unity"""
        try:
            # Create message
            msg = OutgoingMessage()
            msg.write_int32(len(sequence))
            
            # Convert goals to indices
            goal_mapping = {
                'red': 0,    # RedEx
                'green': 1,  # GreenPlus
                'yellow': 2  # YellowStar
            }
            
            # Write each goal index
            for goal in sequence:
                goal_idx = goal_mapping[goal]
                msg.write_int32(goal_idx)
            
            print(f"Sending sequence to Unity: {sequence}")
            print(f"Converted indices: {[goal_mapping[g] for g in sequence]}")
            
            # Use queue_message_to_send instead of send_message
            self.queue_message_to_send(msg)
            return True
            
        except Exception as e:
            print(f"Error sending goal sequence: {e}")
            return False


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
        
        # State tracking variables
        self.last_position = None
        self.last_distance_to_goal = None
        self.last_action = None
        self.consecutive_same_actions = 0
        self.visited_positions = set()
        self.current_goal = None
        self.steps = 0
        self.has_powerup = False
        # Add goal tracking state
        self.achieved_goals = set()
        # Reward shaping parameters
        self.distance_reward_scale = 2.0
        self.progress_reward = 0.5
        self.exploration_bonus = 0.1
        self.min_distance_to_goal = float('inf')


        self.sequence_channel = GoalSequenceChannel(worker_id)
        self.channel = EngineConfigurationChannel()
        self.fixed_goal_sequence = None

        # Store goal sequence and state
        self._goal_sequence = None
        self._current_goal = None
        
        self.goals_representation = {
            'red': np.tile([1, 0, 0], 8),     # RedEx (0)
            'green': np.tile([0, 1, 0], 8),   # GreenPlus (1)
            'yellow': np.tile([0, 0, 1], 8)   # YellowStar (2)
        }
        # Initialize Unity environment
        print(f"Initializing Unity environment {worker_id}")
        self.unity_env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            no_graphics=no_graphics,
            seed=seed,
            side_channels=[self.channel, self.sequence_channel],
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
        
        # Normalize goal vectors
        for goal in self.goals_representation:
            self.goals_representation[goal] = self.goals_representation[goal] / np.linalg.norm(self.goals_representation[goal])

    def _setup_spaces(self):
        """Setup observation and action spaces"""
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
        
        # Changed from 5 to 4 actions
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right

    def _process_obs(self, steps):
        """Process Unity observation with corrected goal encoding"""
        if len(steps) == 0:
            return np.zeros(15)
                
        # Get vector observation
        vector_obs = steps.obs[1][0]  # Shape: (15,)
            # Expected positions (matching Unity's GridArea.cs)
        agent_pos = vector_obs[:2]
        red_pos = vector_obs[8:10]     # Update indices to match order
        green_pos = vector_obs[10:12]  # Update indices to match order
        yellow_pos = vector_obs[12:14] # Yellow goal position
        expected_positions = {
            'agent': np.array([2.5, 2.5]),  # Center
            'red': np.array([1.0, 1.0]),    # Bottom left
            'green': np.array([4.0, 1.0]),  # Bottom right
            'yellow': np.array([2.5, 4.0])  # Top middle
        }
        
        # Optional: Add position verification logging
        if self.steps == 0:  # Only log on reset
            print("\nPosition Verification:")
            print(f"Agent: expected={expected_positions['agent']}, actual={agent_pos}")
            print(f"Red: expected={expected_positions['red']}, actual={red_pos}")
            print(f"Green: expected={expected_positions['green']}, actual={green_pos}")
            print(f"Yellow: expected={expected_positions['yellow']}, actual={yellow_pos}")
            # Log the current goal from Unity's observation
        goal_one_hot = vector_obs[2:5]
        goal_type = "unknown"
        if goal_one_hot[0] == 1:    # First position for RedEx
            goal_type = "red"
        elif goal_one_hot[1] == 1:  # Second position for GreenPlus
            goal_type = "green"
        elif goal_one_hot[2] == 1:  # Third position for YellowStar
            goal_type = "yellow"
        
        print(f"Unity observation:")
        print(f"- Goal one-hot encoding: {goal_one_hot}")
        print(f"- Interpreted goal: {goal_type}")
        print(f"- Expected goal: {self.current_goal}")
        
        return vector_obs.astype(np.float32)

    def _get_obs(self, obs):
        """Convert processed observation to final format"""
        if self.current_goal is None:
            goal_vec = np.zeros(24)
        else:
            try:
                goal_vec = self.goals_representation[self.current_goal]
            except KeyError:
                print(f"Warning: Unknown goal {self.current_goal}")
                goal_vec = np.zeros(24)
        
        # Add goal representation vector to observation
        combined_obs = np.concatenate([obs, goal_vec]).astype(np.float32)
        
        # Check if current goal is achieved
        success = False
        if self.current_goal is not None:
            # Get goal positions from observation
            goal_positions = {
                'red': obs[8:10],
                'green': obs[10:12],
                'yellow': obs[12:14]
            }
            
            agent_pos = obs[:2]
            if self.current_goal in goal_positions:
                goal_pos = goal_positions[self.current_goal]
                distance = np.linalg.norm(agent_pos - goal_pos)
                success = distance < 0.3  # Goal reach threshold
                
                if success and self.current_goal not in self.achieved_goals:
                    self.achieved_goals.add(self.current_goal)
        
        return {
            'obs': combined_obs,
            'success': np.array([float(success)], dtype=np.float32),
            'steps': np.array([self.steps], dtype=np.float32)
        }


    def _calculate_reward(self, obs, action, unity_reward):
        """Calculate shaped rewards on top of Unity's base rewards"""
        reward = unity_reward  # Start with Unity's immediate rewards
        
        if self.current_goal is None:
            return reward
            
        # Get positions
        agent_pos = obs[:2]
        goal_positions = {
            'red': obs[8:10],
            'green': obs[10:12],
            'yellow': obs[12:14]
        }
        
        if self.current_goal not in goal_positions:
            return reward
            
        current_goal_pos = goal_positions[self.current_goal]
        current_distance = np.linalg.norm(agent_pos - current_goal_pos)
        
        # Distance-based shaping
        if self.last_distance_to_goal is not None:
            distance_improvement = self.last_distance_to_goal - current_distance
            reward += distance_improvement * self.distance_reward_scale
            
            # Progress milestone reward
            if current_distance < self.min_distance_to_goal:
                reward += self.progress_reward
                self.min_distance_to_goal = current_distance
        
        # Exploration bonus
        pos = tuple(np.round(agent_pos, decimals=1))
        if pos not in self.visited_positions:
            reward += self.exploration_bonus
            self.visited_positions.add(pos)
            
        # Update tracking
        self.last_distance_to_goal = current_distance
        
        return reward

    def step(self, action):
        """Execute step in environment"""
        print("\n=== WRAPPER STEP START ===")  # Distinctive marker
        if self.unity_env is None:
            raise RuntimeError("Unity environment not initialized")
                
        self.steps += 1
        
        # Check step limit
        # if self.steps >= self.max_steps:
        #     print("!!! steps > max_steps???")
        #     print(f"steps = {self.steps}")
        #     print(f"max_steps = {self.max_steps}")
        #     return self._get_obs(self._process_obs([])), 0.0, True, False, {
        #         'timeout': True,
        #         'steps': self.steps
        #     }
            # Log action translation
        # Add id to help track if multiple envs
        print(f"Worker ID {self.worker_id}:")
        print(f"Step {self.steps}")
        
        # Verbose action logging
        print(f"Action Debug:")
        print(f"1. Received action type: {type(action)}")
        print(f"2. Received action value: {action}")
        print(f"3. Converting to Unity format...")
        unity_action = ActionTuple(discrete=np.array([[int(action)]]))
        print(f"4. Created Unity action: {unity_action.discrete}")
        print(f"5. Unity action shape: {unity_action.discrete.shape}")
        print(f"6. Unity action dtype: {unity_action.discrete.dtype}")
        
        # Track where the action goes
        try:
            print("7. About to send action to Unity...")
            self.unity_env.set_actions(self.behavior_name, unity_action)
            print("8. Successfully sent action to Unity")
            self.unity_env.step()
            print("9. Unity step complete")
        except Exception as e:
            print(f"Error in Unity interaction: {e}")
        
        # Get result from Unity
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        done = len(terminal_steps) > 0
        
        # Process observation and get current Unity state
        obs = self._process_obs(terminal_steps if done else decision_steps)
        unity_reward = terminal_steps.reward[0] if done else decision_steps.reward[0]
        print("\nDEBUG: Goal State Before Update:")
        print(f"Current goal in Python: {self.current_goal}")
        # Check if Unity has updated the goal
        goal_one_hot = obs[2:5]
        print(f"DEBUG step - Before goal update:")
        print(f"Current goal: {self.current_goal}")
        print(f"One-hot encoding: {goal_one_hot}")
        new_goal = None
        if goal_one_hot[0] == 1:        # Changed: Check third position for RedEx
            new_goal = 'red'
        elif goal_one_hot[1] == 1:      # Middle position for GreenPlus
            new_goal = 'green'
        elif goal_one_hot[2] == 1:      # First position for YellowStar
            new_goal = 'yellow'
        print(f"Decoded new goal from Unity: {new_goal}")    
        # Update current goal if changed
        if new_goal != self.current_goal:
            print(f"Goal state changing:")
            print(f"- Old goal: {self.current_goal}")
            print(f"- New goal: {new_goal}")
            print(f"- Current achieved goals: {self.achieved_goals}")
            self.current_goal = new_goal
        # Calculate reward
        reward = self._calculate_reward(obs, action, unity_reward)
        
        # Get final observation with updated goal
        final_obs = self._get_obs(obs)
        
        info = {
            'current_goal': self.current_goal,
            'achieved_goals': self.achieved_goals,
            'steps': self.steps,
            'goal_changed': new_goal != self.current_goal
        }
        print("=== WRAPPER STEP END ===\n")  # End marker
        return final_obs, reward, done, False, info

    def set_fixed_goal_sequence(self, sequence):
        """Set sequence via side channel"""
        print(f"\nSetting goal sequence: {sequence}")
        self._goal_sequence = sequence  # Store sequence
        success = self.sequence_channel.send_sequence(sequence)
        
        if not success:
            print("Failed to queue sequence message")
            return False
        
        # Let Unity process the message
        self.unity_env.step()
        return True
        
    def reset(self, *, seed=None, options=None):
        """Reset with sequence verification"""
        if self.unity_env is None:
            raise RuntimeError("Unity environment not initialized")
            
        # Resend sequence if we have one
        if self._goal_sequence is not None:
            success = self.set_fixed_goal_sequence(self._goal_sequence)
            if not success:
                print("Failed to restore goal sequence")
            
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        
        # Reset state
        self.steps = 0
        self.achieved_goals.clear()
        
        # Process observation
        obs = self._process_obs(decision_steps)
        goal_one_hot = obs[2:5]
        
        if np.all(goal_one_hot == 0):
            print("Warning: Invalid goal state - attempting recovery")
            self.unity_env.step()  # Let Unity update
            decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
            obs = self._process_obs(decision_steps)
            
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

    
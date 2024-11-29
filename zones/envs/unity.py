from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import gymnasium as gym
import time
from gymnasium import spaces
import numpy as np
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage
)
import uuid

class GoalSequenceChannel(SideChannel):
    def __init__(self):
        # Use same UUID as Unity side
        channel_id = uuid.UUID('621f0a70-4f87-11ee-be56-0242ac120002')
        super().__init__(channel_id)
    
    def on_message_received(self, msg: IncomingMessage) -> None:
        """Required implementation of abstract method"""
        pass
    
    def send_message(self, msg: OutgoingMessage) -> None:
        """Send message to Unity"""
        super().queue_message_to_send(msg)
class UnityGCRLLTLWrapper(gym.Env):
    def __init__(self, env_path, worker_id=0, no_graphics=False, time_scale=1.0):
        self.env_path = env_path
        
        # Setup Unity environment
        self.channel = EngineConfigurationChannel()
        self.sequence_channel = GoalSequenceChannel()
        self.unity_env = UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            no_graphics=no_graphics,
            side_channels=[self.channel, self.sequence_channel],
            timeout_wait=60
        )
        
        # Configure environment settings
        self.channel.set_configuration_parameters(
            time_scale=time_scale,
            width=640,
            height=480,
            quality_level=0
        )
        
        # Initialize connection
        self.unity_env.reset()
        self.behavior_name = list(self.unity_env.behavior_specs.keys())[0]
        
        # Current goal tracking
        self._current_goal = None
        self._goal_sequence = None
        
        # Setup spaces
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Setup basic observation and action spaces"""
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(4)  # UP, DOWN, LEFT, RIGHT

    def _process_obs(self, steps):
        if len(steps) == 0:
            return np.zeros(15)
                
        # Get vector observation
        vector_obs = steps.obs[1][0]  # Shape: (15,)
        
        # Unity uses (x,z), but we need to map it to our (x,y) system
        unity_x, unity_z = vector_obs[:2]
        agent_pos = np.array([unity_x, unity_z])  # Keep x, but use z as our y
        
        # Similarly map goal positions
        goal_positions = {
            'red': vector_obs[8:10],    # Maps Unity's (x,z) for red
            'green': vector_obs[10:12],  # Maps Unity's (x,z) for green
            'yellow': vector_obs[12:14]  # Maps Unity's (x,z) for yellow
        }
        
        return vector_obs
    
    def step(self, action):
        """Take step in environment with proper coordinate mapping"""
        # Map our actions to Unity's coordinate system
        unity_action = self._map_action_to_unity(action)
        unity_action = ActionTuple(discrete=np.array([[int(unity_action)]])) 

        # Execute action
        self.unity_env.set_actions(self.behavior_name, unity_action)
        self.unity_env.step()
        
        # Get results
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        done = len(terminal_steps) > 0
        
        # Get observation and verify actual movement
        if done:
            obs = self._process_obs(terminal_steps)
            reward = terminal_steps.reward[0]
        else:
            obs = self._process_obs(decision_steps)
            reward = decision_steps.reward[0]
        
        return {'obs': obs}, reward, done, False, {}

    def _map_action_to_unity(self, action):
        """Map our action space to Unity's proper coordinate system"""
        # Unity coordinates vs our coordinates:
        # Unity (x,z):     Our system:
        # +z is forward     +y is up
        # -z is backward    -y is down
        # +x is right       +x is right 
        # -x is left        -x is left

        unity_action_map = {
            0: 3,  # UP should map to Unity's FORWARD (+z)
            1: 2,  # DOWN should map to Unity's BACKWARD (-z)
            2: 0,  # LEFT should map to Unity's LEFT (-x)
            3: 1   # RIGHT should map to Unity's RIGHT (+x)
        }
        return unity_action_map.get(action, action)
        
    def reset(self):
        """Reset environment"""
        self.unity_env.reset()
        decision_steps, _ = self.unity_env.get_steps(self.behavior_name)
        
        obs = {'obs': self._process_obs(decision_steps)}
        return obs, {}

    def set_fixed_goal_sequence(self, sequence):
        """Set goal sequence through side channel to Unity"""
        # Convert sequence to Unity's goal indices
        goal_to_unity = {
            'red': 0,    # RedEx
            'green': 1,  # GreenPlus  
            'yellow': 2  # YellowStar
        }
        
        try:
            # Create message
            msg = OutgoingMessage()
            msg.write_int32(len(sequence))  # Write sequence length
            
            # Convert and write each goal
            unity_sequence = [goal_to_unity[goal] for goal in sequence]
            for goal_idx in unity_sequence:
                msg.write_int32(goal_idx)
                
            # Send through channel
            self.sequence_channel.send_message(msg)
            print(f"Sent sequence to Unity: {sequence}")
            
            # Store locally
            self._goal_sequence = sequence
            self._current_goal = sequence[0]
            return True
            
        except Exception as e:
            print(f"Failed to send sequence: {e}")
            return False

    def close(self):
        """Cleanup"""
        if self.unity_env:
            self.unity_env.close()
            self.unity_env = None
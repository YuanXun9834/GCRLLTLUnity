from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class CollectTrajectoryCallback(BaseCallback):
    def __init__(self, traj_buffer, verbose: int = 0):
        super().__init__(verbose)
        self.traj_buffer = traj_buffer
        self.episode_states = []
        self.episode_rewards = []
        self.success_episodes = []

    def _on_step(self) -> bool:
        """
        This method is called after every step in the environment.
        Access training info through the parent class's locals dictionary.
        """
        try:
            # Get observations and rewards
            obs = self.locals['new_obs']  # The new observation after the step
            reward = self.locals['rewards']  # The reward received
            
            if isinstance(obs, dict):
                # Handle dictionary observations
                self.episode_states.append(obs['obs'].copy())
            else:
                self.episode_states.append(obs.copy())
            
            self.episode_rewards.append(reward)
            
            # Print some debugging info periodically
            if self.n_calls % 1000 == 0:
                print(f"Collected {len(self.episode_states)} states in current episode")
                print(f"Last reward: {reward}")
                
        except KeyError as e:
            print(f"KeyError in callback: {e}")
            print(f"Available keys in locals: {self.locals.keys()}")
        except Exception as e:
            print(f"Error in callback: {e}")
            
        return True

    def _on_rollout_end(self) -> None:
        """
        This method is called after a rollout is completed.
        A rollout is a series of steps collected for updating the policy.
        """
        try:
            # Calculate episode success based on rewards
            episode_total_reward = sum(self.episode_rewards)
            
            # Only store successful trajectories
            if episode_total_reward > 5.0:  # Threshold for "good" episodes
                self.success_episodes.append({
                    'states': np.array(self.episode_states),
                    'rewards': np.array(self.episode_rewards),
                })
                
                print(f"Stored successful episode with reward: {episode_total_reward}")
            
            # Add to buffer if we have enough successful episodes
            if len(self.success_episodes) >= 5:
                print(f"Adding {len(self.success_episodes)} episodes to buffer")
                self.traj_buffer.add_rollouts(self.success_episodes)
                self.success_episodes = []

            # Clear episode storage
            self.episode_states = []
            self.episode_rewards = []
            
        except Exception as e:
            print(f"Error in _on_rollout_end: {e}")

    def _on_training_end(self) -> None:
        """
        This method is called at the end of training.
        Save any remaining successful episodes.
        """
        if self.success_episodes:
            print(f"Training ended. Adding final {len(self.success_episodes)} episodes to buffer")
            self.traj_buffer.add_rollouts(self.success_episodes)
            self.success_episodes = []
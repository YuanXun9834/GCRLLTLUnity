from stable_baselines3.common.callbacks import BaseCallback


class CollectTrajectoryCallback(BaseCallback):
    def __init__(self, traj_buffer, verbose: int = 0):
        super().__init__(verbose)
        self.traj_buffer = traj_buffer
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        print(f"\nCollecting rollout {self.rollout_count}:")
        # Use the buffer size instead of len
        print(f"Rollout buffer observations shape: {self.model.rollout_buffer.observations['obs'].shape}")
        print(f"Rollout buffer size: {self.model.rollout_buffer.buffer_size}")
        
        try:
            self.traj_buffer.add_rollouts(self.model.rollout_buffer)
            print(f"Successfully added rollout {self.rollout_count}")
        except Exception as e:
            print(f"Error adding rollout {self.rollout_count}: {e}")
        
        self.rollout_count += 1

    def _on_step(self) -> bool:
        # We can add per-step debugging here
        if self.n_calls % 1000 == 0:  # Log every 1000 steps
            print(f"Step {self.n_calls}")
        return True

import os

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization


class CollectTrajectoryCallback(BaseCallback):

    def __init__(self, traj_buffer, verbose: int = 0):
        super().__init__(verbose)
        self.traj_buffer = traj_buffer
        self.goals = []

    def _on_rollout_start(self) -> None:
        print(self.training_env)
        print(self.training_env.unwrapped)
        print(self.model.rollout_buffer)
        print('WTF', self.training_env.env_method('current_env_index'))
        print(self.training_env.env_method.current_env_index())
        exit()
        pass
        # self.goals.append(self.training_env.current_goal())
        # print(self.model)
        # print(self.training_env)
        # print(dir(self.training_env))
        # print(self.training_env._get_indices(None))
        # print(self.training_env.num_envs)
        # print(self.training_env.processes)
        # ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', 
        #  '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', 
        #  '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', 
        #  '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', 
        #  '__subclasshook__', '__weakref__', '_abc_cache', '_abc_negative_cache', '_abc_negative_cache_version', 
        #  '_abc_registry', '_get_indices', '_get_target_remotes', 'action_space', 'close', 'closed', 'env_is_wrapped', 
        #  'env_method', 'get_attr', 'get_images', 'getattr_depth_check', 'metadata', 'num_envs', 'observation_space', 
        #  'processes', 'remotes', 'render', 'reset', 'seed', 'set_attr', 'step', 'step_async', 'step_wait', 'unwrapped', 'waiting', 'work_remotes']
    
    def _on_rollout_end(self) -> None:
        model = self.model
        env = self.training_env
        # 1st rollout
        if env.reset_continual and env.success and env.current_start() == 'ANYWHERE':
            env.rollout_buffer.add_rollout(model.rollout_buffer)
            self.goals.append(env.current_goal())
        elif env.reset_continual and env.start != 'ANYWHERE':
            assert env.current_start() == self.goals[0]
            if env.success:
                self.goals.append(env.current_goals)
                env.rollout_buffer.add_rollout(model.rollout_buffer)
                env.rollout_buffer.push_rollout()
            else:
                env.rollout_buffer.revoke_rollout()
                self.goals = []

    def _on_step(self) -> bool:
        return True

class SimpleEvalCallback(EvalCallback):

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                # NOTE: save GCPPO model directly is not available
                # if self.best_model_save_path is not None:
                #     self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True
    

class PolicyCheckpointCallback(BaseCallback):
    """
    Callback for saving a ActorCritic policy in model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = 'gc_ppo', verbose: int = 0):
        super(PolicyCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            policy_path = os.path.join(self.save_path, f"{self.name_prefix}_policy_{self.num_timesteps}_steps")
            self.model.save_policy(path=policy_path)
            if self.verbose > 1:
                print(f"Saving ActorCriticPolicy checkpoint to {self.save_path}")
        return True
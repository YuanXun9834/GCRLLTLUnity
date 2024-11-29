import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.unity import UnityGCRLLTLWrapper
import logging
from collections import defaultdict

# Set logging level to INFO only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('verification.log'),
        logging.StreamHandler()
    ]
)

# Disable matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def verify_training(model_path, env_path, num_episodes=20):
    """Verify trained model performance with enhanced diagnostics"""
    print("\nStarting training verification...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with debug flags
    env = UnityGCRLLTLWrapper(
        env_path=env_path,
        worker_id=9999,
        no_graphics=False
    )
    
    # Track detailed statistics
    episode_stats = {
        'complete_sequences': 0,
        'partial_sequences': defaultdict(int),
        'goal_order_stats': defaultdict(int),
        'all_sequences': [],
        'rewards': [],
        'steps_per_goal': defaultdict(list),
        'action_distributions': defaultdict(int),  # Will store int actions
        'goal_achievement_times': [],
        'trajectories': []
    }

    for episode in range(num_episodes):
        print(f"\n{'='*20} Episode {episode + 1} {'='*20}")
        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        achieved_goals = set()
        current_sequence = []
        episode_trajectory = []
        
        done = False
        while not done:
            # Record pre-action state
            pre_action_pos = obs['obs'][:2]
            
            # Get model action
            action, _ = model.predict(obs, deterministic=True)
            # Convert numpy array to integer
            action_int = action.item() if isinstance(action, np.ndarray) else int(action)
            
            print(f"\nStep {steps}:")
            print(f"Current position: {pre_action_pos}")
            print(f"Selected action: {action_int}")
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action_int)
            done = terminated or truncated
            
            # Record post-action state
            post_action_pos = obs['obs'][:2]
            movement_delta = np.linalg.norm(post_action_pos - pre_action_pos)
            
            # Update statistics (using integer action)
            total_reward += reward
            steps += 1
            episode_stats['action_distributions'][action_int] += 1
            
            # Store step information
            episode_trajectory.append({
                'step': steps,
                'pre_pos': pre_action_pos.copy(),  # Make copies of numpy arrays
                'post_pos': post_action_pos.copy(),
                'action': action_int,  # Store as integer
                'reward': float(reward),  # Convert to Python float
                'movement': float(movement_delta),
                'current_goal': info.get('current_goal'),
                'achieved_goals': set(info.get('achieved_goals', set()))  # Make copy of set
            })
            
            # Print step information
            print(f"Movement: {movement_delta:.4f}")
            print(f"Reward: {reward:.4f}")
            print(f"Current goal: {info.get('current_goal')}")
            print(f"Achieved goals: {info.get('achieved_goals', set())}")
            
            # Check goal achievements
            current_achieved = set(info.get('achieved_goals', set()))
            if current_achieved != achieved_goals:
                new_goals = current_achieved - achieved_goals
                for goal in new_goals:
                    current_sequence.append(goal)
                    episode_stats['steps_per_goal'][goal].append(steps)
                achieved_goals = current_achieved
            
            if steps >= 1000:  # Timeout
                print(f"Episode {episode + 1}: Timeout after {steps} steps")
                break
        
        # Record episode statistics
        episode_stats['rewards'].append(total_reward)
        episode_stats['all_sequences'].append(current_sequence)
        episode_stats['trajectories'].append(episode_trajectory)
        
        if len(achieved_goals) == 3:  # Complete sequence
            episode_stats['complete_sequences'] += 1
            print(f"Completed sequence in {steps} steps: {' -> '.join(current_sequence)}")
        else:
            episode_stats['partial_sequences'][len(achieved_goals)] += 1
            print(f"Partial completion: achieved {len(achieved_goals)} goals: {' -> '.join(current_sequence)}")
        
        # Record sequence order
        sequence_str = '->'.join(current_sequence) if current_sequence else ''
        episode_stats['goal_order_stats'][sequence_str] += 1
    
    # Print comprehensive summary
    print("\nDetailed Results Summary:")
    print(f"Complete Sequences: {episode_stats['complete_sequences']}/{num_episodes} "
          f"({episode_stats['complete_sequences']/num_episodes*100:.1f}%)")
    
    print("\nPartial Completions:")
    for num_goals, count in sorted(episode_stats['partial_sequences'].items()):
        print(f"{num_goals} goals achieved: {count} times")
    
    print("\nSequence Orders Attempted:")
    for sequence, count in episode_stats['goal_order_stats'].items():
        print(f"{sequence}: {count} times")
    
    print("\nAction Distribution:")
    total_actions = sum(episode_stats['action_distributions'].values())
    for action, count in sorted(episode_stats['action_distributions'].items()):
        action_name = {
            0: 'NO_ACTION',
            1: 'UP',
            2: 'DOWN',
            3: 'LEFT',
            4: 'RIGHT'
        }.get(action, f'UNKNOWN_{action}')
        print(f"{action_name}: {count} times ({count/total_actions*100:.1f}%)")
    
    print("\nAverage Steps Per Goal:")
    for goal, steps_list in episode_stats['steps_per_goal'].items():
        if steps_list:
            avg_steps = np.mean(steps_list)
            print(f"{goal}: {avg_steps:.1f} steps")
    
    print(f"\nAverage reward per episode: {np.mean(episode_stats['rewards']):.2f}")
    
    # Plot trajectories
    plot_trajectories(episode_stats['trajectories'])
    
    env.close()
    return episode_stats


def plot_trajectories(trajectories):
    """Plot movement patterns and reward distribution"""
    plt.figure(figsize=(15, 5))
    
    # Plot movement patterns
    plt.subplot(131)
    for episode_traj in trajectories:
        positions = np.array([step['post_pos'] for step in episode_traj])
        plt.plot(positions[:, 0], positions[:, 1], alpha=0.5)
    plt.title('Agent Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    # Plot reward distribution
    plt.subplot(132)
    rewards = [step['reward'] for traj in trajectories for step in traj]
    plt.hist(rewards, bins=50)
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Count')
    
    # Plot movement heatmap
    plt.subplot(133)
    all_positions = np.array([step['post_pos'] for traj in trajectories for step in traj])
    plt.hist2d(all_positions[:, 0], all_positions[:, 1], bins=20)
    plt.title('Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png')
    plt.close()


def verify_dataset(dataset_path):
    """Verify trajectory dataset"""
    print("\nVerifying trajectory dataset...")
    
    # Load dataset
    dataset = torch.load(dataset_path)
    
    print(f"\nDataset Statistics:")
    print(f"Number of trajectories: {len(dataset)}")
    
    # Verify state dimensions
    state_dim = dataset.states[0].shape[0] if len(dataset.states) > 0 else 0
    print(f"State dimension: {state_dim}")
    print(f"Expected dimension: {39}")  # 15 (base) + 24 (goal vector)
    
    # Analyze goal values
    goal_values = np.array(dataset.goal_values)
    print(f"\nGoal Values Statistics:")
    print(f"Min value: {goal_values.min():.3f}")
    print(f"Max value: {goal_values.max():.3f}")
    print(f"Mean value: {goal_values.mean():.3f}")
    print(f"Std value: {goal_values.std():.3f}")
    
    # Plot goal value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(goal_values, bins=50)
    plt.title('Distribution of Goal Values')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.savefig('goal_values_distribution.png')
    plt.close()
    
    # Check for NaN or infinity values
    has_nan = np.any(np.isnan(goal_values))
    has_inf = np.any(np.isinf(goal_values))
    print(f"\nData Quality:")
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Infinity: {has_inf}")
    
    # Analyze state distributions
    states = np.array(dataset.states)
    print(f"\nState Statistics:")
    print(f"State value range: [{states.min():.3f}, {states.max():.3f}]")
    print(f"State mean: {states.mean():.3f}")
    print(f"State std: {states.std():.3f}")
    
    return {
        'num_trajectories': len(dataset),
        'state_dim': state_dim,
        'goal_values_stats': {
            'min': goal_values.min(),
            'max': goal_values.max(),
            'mean': goal_values.mean(),
            'std': goal_values.std()
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to trajectory dataset')
    parser.add_argument('--unity_env_path', type=str, required=True,
                      help='Path to Unity environment')
    parser.add_argument('--num_episodes', type=int, default=10,
                      help='Number of episodes to test per goal')
    
    args = parser.parse_args()
    
    # Verify training
    training_results = verify_training(
        model_path=args.model_path,
        env_path=args.unity_env_path,
        num_episodes=args.num_episodes
    )
    
    # Verify dataset
    dataset_stats = verify_dataset(args.dataset_path)
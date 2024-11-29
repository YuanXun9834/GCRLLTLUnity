import torch
import numpy as np
from stable_baselines3 import PPO
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from envs.unity import UnityGCRLLTLWrapper

def verify_sequence_models(
    unity_env_path: str,
    num_eval_episodes: int = 50,
    sequences = [
        ['red', 'green', 'yellow'],
        ['red', 'yellow', 'green'],
        ['green', 'red', 'yellow'],
        ['green', 'yellow', 'red'],
        ['yellow', 'red', 'green'],
        ['yellow', 'green', 'red']
    ]
):
    """Verify performance of each sequence-specific model"""
    
    results = {}
    
    for seq_idx, sequence in enumerate(sequences):
        print(f"\n{'='*50}")
        print(f"Evaluating sequence {seq_idx + 1}/6: {' -> '.join(sequence)}")
        print(f"{'='*50}")
        
        model_path = f'models/sequence_{seq_idx}_final'
        try:
            # Load model
            model = PPO.load(model_path)
            
            # Create environment with fixed sequence
            env = UnityGCRLLTLWrapper(
                env_path=unity_env_path,
                worker_id=seq_idx + 100,  # Different from training worker_ids
                no_graphics=True
            )
            env.set_fixed_goal_sequence(sequence)
            
            # Evaluate
            sequence_results = {
                'complete_sequences': 0,
                'partial_completions': defaultdict(int),
                'avg_steps_per_goal': defaultdict(list),
                'rewards': [],
                'goal_achievement_times': defaultdict(list)
            }
            
            for episode in range(num_eval_episodes):
                obs = env.reset()[0]
                done = False
                episode_reward = 0
                steps = 0
                last_goal_step = 0
                achieved_goals = set()
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    episode_reward += reward
                    steps += 1
                    
                    # Track goal achievements
                    current_goals = set(info.get('achieved_goals', []))
                    new_goals = current_goals - achieved_goals
                    if new_goals:
                        for goal in new_goals:
                            goal_steps = steps - last_goal_step
                            sequence_results['goal_achievement_times'][goal].append(goal_steps)
                            sequence_results['avg_steps_per_goal'][goal].append(goal_steps)
                            last_goal_step = steps
                        achieved_goals = current_goals
                
                # Record episode results
                sequence_results['rewards'].append(episode_reward)
                num_goals = len(achieved_goals)
                if num_goals == 3:
                    sequence_results['complete_sequences'] += 1
                else:
                    sequence_results['partial_completions'][num_goals] += 1
            
            # Calculate statistics
            results[tuple(sequence)] = {
                'success_rate': sequence_results['complete_sequences'] / num_eval_episodes,
                'average_reward': np.mean(sequence_results['rewards']),
                'partial_completions': dict(sequence_results['partial_completions']),
                'avg_steps_per_goal': {
                    goal: np.mean(steps) for goal, steps in sequence_results['avg_steps_per_goal'].items()
                },
                'goal_timing': {
                    goal: {
                        'mean': np.mean(times),
                        'std': np.std(times)
                    } for goal, times in sequence_results['goal_achievement_times'].items()
                }
            }
            
            # Print sequence results
            print(f"\nResults for sequence {' -> '.join(sequence)}:")
            print(f"Success rate: {results[tuple(sequence)]['success_rate']*100:.1f}%")
            print(f"Average reward: {results[tuple(sequence)]['average_reward']:.2f}")
            print("\nPartial completions:")
            for goals, count in results[tuple(sequence)]['partial_completions'].items():
                print(f"{goals} goals achieved: {count} times")
            print("\nAverage steps per goal:")
            for goal, avg_steps in results[tuple(sequence)]['avg_steps_per_goal'].items():
                print(f"{goal}: {avg_steps:.1f} steps")
            
            env.close()
            
        except Exception as e:
            print(f"Error evaluating sequence {sequence}: {str(e)}")
            continue
    
    return results

def analyze_trajectory_dataset(dataset_path: str):
    """Analyze distribution and quality of collected trajectories"""
    print(f"\nAnalyzing trajectory dataset: {dataset_path}")
    
    try:
        dataset = torch.load(dataset_path)
        
        # Basic statistics
        num_trajectories = len(dataset.states)
        print(f"\nDataset Statistics:")
        print(f"Total trajectories: {num_trajectories}")
        
        # Analyze state distributions
        states = np.array(dataset.states)
        goal_vectors = states[:, -24:]  # Last 24 dimensions are goal vector
        
        # Count goal types
        goal_counts = defaultdict(int)
        for goal_vec in goal_vectors:
            # Identify goal type from one-hot encoding
            if np.allclose(goal_vec[:8], 1):
                goal_counts['green'] += 1
            elif np.allclose(goal_vec[8:16], 1):
                goal_counts['red'] += 1
            elif np.allclose(goal_vec[16:], 1):
                goal_counts['yellow'] += 1
        
        print("\nGoal Distribution:")
        total_goals = sum(goal_counts.values())
        for goal, count in goal_counts.items():
            percentage = (count / total_goals) * 100
            print(f"{goal}: {count} ({percentage:.1f}%)")
        
        # Analyze value distribution
        values = np.array(dataset.goal_values)
        print("\nValue Statistics:")
        print(f"Mean value: {np.mean(values):.3f}")
        print(f"Std value: {np.std(values):.3f}")
        print(f"Min value: {np.min(values):.3f}")
        print(f"Max value: {np.max(values):.3f}")
        
        # Plot distributions
        plt.figure(figsize=(15, 5))
        
        # Goal distribution
        plt.subplot(131)
        goals = list(goal_counts.keys())
        counts = [goal_counts[g] for g in goals]
        plt.bar(goals, counts)
        plt.title('Goal Distribution')
        plt.ylabel('Count')
        
        # Value distribution
        plt.subplot(132)
        plt.hist(values, bins=50)
        plt.title('Value Distribution')
        plt.xlabel('Value')
        plt.ylabel('Count')
        
        # Success rate by goal
        plt.subplot(133)
        success_threshold = 0.8  # Consider trajectories with value > 0.8 as successful
        success_rates = {
            goal: np.mean(values[goal_vectors == g] > success_threshold)
            for goal, g in zip(goals, range(3))
        }
        plt.bar(success_rates.keys(), success_rates.values())
        plt.title('Success Rate by Goal')
        plt.ylabel('Success Rate')
        
        plt.tight_layout()
        plt.savefig('trajectory_analysis.png')
        plt.close()
        
        # Check for balance
        balance_threshold = 0.2  # Maximum allowed deviation from perfect balance
        is_balanced = all(
            abs(count/total_goals - 1/3) < balance_threshold
            for count in goal_counts.values()
        )
        
        if not is_balanced:
            print("\nWARNING: Dataset is not well-balanced across goals!")
            print("Consider collecting more data for underrepresented goals.")
        else:
            print("\nDataset is well-balanced across goals.")
        
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--unity_env_path', type=str, required=True,
                      help='Path to Unity environment executable')
    parser.add_argument('--dataset_path', type=str, required=True,
                      help='Path to trajectory dataset')
    parser.add_argument('--num_episodes', type=int, default=50,
                      help='Number of evaluation episodes per sequence')
    
    args = parser.parse_args()
    
    # Verify each sequence model
    model_results = verify_sequence_models(
        unity_env_path=args.unity_env_path,
        num_eval_episodes=args.num_episodes
    )
    
    # Analyze success rates across sequences
    print("\nOverall Model Performance Summary:")
    success_rates = [results['success_rate'] for results in model_results.values()]
    print(f"Average success rate across sequences: {np.mean(success_rates)*100:.1f}%")
    print(f"Best sequence success rate: {np.max(success_rates)*100:.1f}%")
    print(f"Worst sequence success rate: {np.min(success_rates)*100:.1f}%")
    
    # Create performance plot
    plt.figure(figsize=(10, 6))
    sequences = [' -> '.join(seq) for seq in model_results.keys()]
    rates = [results['success_rate'] * 100 for results in model_results.values()]
    plt.bar(sequences, rates)
    plt.xticks(rotation=45)
    plt.ylabel('Success Rate (%)')
    plt.title('Success Rate by Goal Sequence')
    plt.tight_layout()
    plt.savefig('sequence_performance.png')
    plt.close()
    
    # Analyze trajectory dataset
    analyze_trajectory_dataset(args.dataset_path)
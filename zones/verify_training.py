import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.unity import UnityGCRLLTLWrapper

def verify_training(model_path, env_path, num_episodes=10):
    """Verify trained model performance"""
    print("\nVerifying trained model...")
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = UnityGCRLLTLWrapper(
        env_path=env_path,
        worker_id=9999,  # High worker ID to avoid conflicts
        no_graphics=False  # Set to True if you don't need visualization
    )
    
    # Test different goals
    goals = ['green', 'red', 'yellow']
    results = {goal: {'successes': 0, 'times': [], 'rewards': []} for goal in goals}
    
    for goal in goals:
        print(f"\nTesting goal: {goal}")
        
        for episode in range(num_episodes):
            env.fix_goal(goal)
            obs, _ = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if info.get('goal_achieved', False):
                    results[goal]['successes'] += 1
                    results[goal]['times'].append(steps)
                    results[goal]['rewards'].append(total_reward)
                    print(f"Episode {episode + 1}: Goal achieved in {steps} steps!")
                    break
                
                if steps >= 1000:  # Timeout
                    print(f"Episode {episode + 1}: Timeout")
                    break
    
    # Print results
    print("\nResults Summary:")
    for goal in goals:
        success_rate = results[goal]['successes'] / num_episodes * 100
        avg_time = np.mean(results[goal]['times']) if results[goal]['times'] else float('inf')
        avg_reward = np.mean(results[goal]['rewards']) if results[goal]['rewards'] else 0
        
        print(f"\n{goal.upper()} Goal:")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Steps to Goal: {avg_time:.1f}")
        print(f"Average Reward: {avg_reward:.2f}")
    
    env.close()
    return results

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
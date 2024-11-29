import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from envs.unity import UnityGCRLLTLWrapper
from collections import defaultdict
from tqdm import tqdm

class SequenceVerifier:
    def __init__(self, model_path, env_path, device='cuda'):
        self.model = PPO.load(model_path, device=device)
        self.env_path = env_path
        self.device = device
        
    def verify_sequence(self, sequence, n_episodes=20, max_steps=1000, verbose=True):
        """Verify a specific sequence with detailed analysis"""
        results = {
            'complete_success': 0,
            'partial_completions': defaultdict(int),
            'goal_stats': defaultdict(lambda: {'attempts': 0, 'successes': 0, 'avg_time': []}),
            'sequences': [],
            'total_rewards': [],
            'episode_lengths': [],
            'goal_order_stats': defaultdict(int),
            'failure_points': defaultdict(int)
        }
        
        env = UnityGCRLLTLWrapper(
            self.env_path,
            worker_id=999,
            no_graphics=True
        )
        env.set_fixed_goal_sequence(sequence)
        
        print(f"\nVerifying sequence: {' -> '.join(sequence)}")
        print(f"Running {n_episodes} episodes...")
        
        for episode in tqdm(range(n_episodes)):
            episode_data = self._run_episode(env, sequence, max_steps)
            self._update_results(results, episode_data, sequence)
            
        self._print_analysis(results, sequence)
        env.close()
        return results
    
    def _run_episode(self, env, sequence, max_steps):
        """Run a single episode and collect detailed data"""
        obs = env.reset()[0]
        done = False
        steps = 0
        total_reward = 0
        current_goal_idx = 0
        episode_data = {
            'achieved_goals': [],
            'goal_times': {},
            'positions': [],
            'actions': [],
            'rewards': [],
            'total_steps': 0
        }
        
        while not done and steps < max_steps:
            # Record position
            episode_data['positions'].append(obs['obs'][:2])
            
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            episode_data['actions'].append(action)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward
            
            episode_data['rewards'].append(reward)
            
            # Track goal achievements
            if info.get('goal_achieved', False):
                achieved_goal = info.get('current_goal')
                episode_data['achieved_goals'].append(achieved_goal)
                episode_data['goal_times'][achieved_goal] = steps
                current_goal_idx += 1
        
        episode_data['total_steps'] = steps
        episode_data['total_reward'] = total_reward
        return episode_data
    
    def _update_results(self, results, episode_data, sequence):
        """Update results with episode data"""
        # Track complete successes
        achieved = episode_data['achieved_goals']
        if len(achieved) == len(sequence):
            results['complete_success'] += 1
        
        # Track partial completions
        results['partial_completions'][len(achieved)] += 1
        
        # Update goal stats
        for idx, goal in enumerate(sequence):
            results['goal_stats'][goal]['attempts'] += 1
            if idx < len(achieved) and achieved[idx] == goal:
                results['goal_stats'][goal]['successes'] += 1
                results['goal_stats'][goal]['avg_time'].append(
                    episode_data['goal_times'][goal]
                )
        
        # Track sequences
        results['sequences'].append(achieved)
        results['total_rewards'].append(episode_data['total_reward'])
        results['episode_lengths'].append(episode_data['total_steps'])
        
        # Track goal order
        if achieved:
            results['goal_order_stats'][tuple(achieved)] += 1
        
        # Track failure points
        if len(achieved) < len(sequence):
            failure_point = achieved[-1] if achieved else 'start'
            results['failure_points'][failure_point] += 1
    
    def _print_analysis(self, results, sequence):
        """Print detailed analysis of results"""
        n_episodes = sum(results['partial_completions'].values())
        
        print("\n" + "="*50)
        print("DETAILED SEQUENCE ANALYSIS")
        print("="*50)
        
        print(f"\nTarget Sequence: {' -> '.join(sequence)}")
        
        print("\nOverall Performance:")
        print(f"Complete Success Rate: {results['complete_success']/n_episodes*100:.1f}%")
        print(f"Average Reward: {np.mean(results['total_rewards']):.2f}")
        print(f"Average Episode Length: {np.mean(results['episode_lengths']):.1f} steps")
        
        print("\nPartial Completions:")
        for goals_achieved, count in sorted(results['partial_completions'].items()):
            print(f"{goals_achieved} goals: {count} episodes ({count/n_episodes*100:.1f}%)")
            
        print("\nIndividual Goal Performance:")
        for goal, stats in results['goal_stats'].items():
            success_rate = stats['successes']/stats['attempts']*100
            avg_time = np.mean(stats['avg_time']) if stats['avg_time'] else float('inf')
            print(f"\n{goal}:")
            print(f"- Success Rate: {success_rate:.1f}%")
            print(f"- Average Time to Reach: {avg_time:.1f} steps")
            
        print("\nMost Common Sequences Achieved:")
        for sequence, count in sorted(
            results['goal_order_stats'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]:
            print(f"{' -> '.join(sequence)}: {count} times ({count/n_episodes*100:.1f}%)")
            
        print("\nFailure Analysis:")
        for point, count in sorted(
            results['failure_points'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            print(f"Failed at {point}: {count} times ({count/n_episodes*100:.1f}%)")
            
        self._plot_analysis(results)
    
    def _plot_analysis(self, results):
        """Create visualization of results"""
        try:
            import matplotlib.pyplot as plt
            
            # Plot reward distribution
            plt.figure(figsize=(12, 4))
            
            plt.subplot(131)
            plt.hist(results['total_rewards'], bins=20)
            plt.title('Reward Distribution')
            plt.xlabel('Total Reward')
            plt.ylabel('Count')
            
            plt.subplot(132)
            goals_achieved = list(results['partial_completions'].keys())
            counts = list(results['partial_completions'].values())
            plt.bar(goals_achieved, counts)
            plt.title('Goals Achieved Distribution')
            plt.xlabel('Number of Goals')
            plt.ylabel('Episodes')
            
            plt.subplot(133)
            plt.hist(results['episode_lengths'], bins=20)
            plt.title('Episode Length Distribution')
            plt.xlabel('Steps')
            plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('sequence_analysis.png')
            print("\nAnalysis plots saved to sequence_analysis.png")
            
        except ImportError:
            print("\nMatplotlib not available - skipping plots")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--env_path', type=str, required=True)
    parser.add_argument('--sequences', type=str, nargs='+', default=[],
                      help='Sequences to verify, e.g. "red,green,yellow" "yellow,red,green"')
    parser.add_argument('--n_episodes', type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    verifier = SequenceVerifier(args.model_path, args.env_path, args.device)
    
    # Parse sequences
    sequences = []
    if args.sequences:
        for seq in args.sequences:
            sequences.append(seq.split(','))
    else:
        # Default sequences
        sequences = [
            ['red', 'green', 'yellow'],
            ['red', 'yellow', 'green'],
            ['green', 'red', 'yellow'],
            ['green', 'yellow', 'red'],
            ['yellow', 'red', 'green'],
            ['yellow', 'green', 'red']
        ]
    
    # Verify each sequence
    all_results = {}
    for sequence in sequences:
        results = verifier.verify_sequence(
            sequence,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps
        )
        all_results[tuple(sequence)] = results
    
    # Print comparative analysis
    print("\n" + "="*50)
    print("COMPARATIVE ANALYSIS")
    print("="*50)
    
    for sequence, results in all_results.items():
        success_rate = results['complete_success']/args.n_episodes*100
        avg_reward = np.mean(results['total_rewards'])
        print(f"\nSequence: {' -> '.join(sequence)}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    main()
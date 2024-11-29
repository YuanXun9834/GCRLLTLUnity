import argparse
import os
import torch
from stable_baselines3 import PPO
from envs.unity import UnityGCRLLTLWrapper

def execute_ltl_sequence(
    unity_env_path: str,
    goal_sequence: list,
    models_dir: str = 'models',
    device: str = 'cuda',
    render: bool = True
):
    """Execute an LTL-generated goal sequence using sequence-specific models"""
    
    # Define all possible sequences for lookup
    all_sequences = [
        ['red', 'green', 'yellow'],
        ['red', 'yellow', 'green'],
        ['green', 'red', 'yellow'],
        ['green', 'yellow', 'red'],
        ['yellow', 'red', 'green'],
        ['yellow', 'green', 'red']
    ]
    
    # Find which sequence matches our optimal path
    sequence_idx = None
    for idx, seq in enumerate(all_sequences):
        if seq == goal_sequence:
            sequence_idx = idx
            break
            
    if sequence_idx is None:
        raise ValueError(f"Generated sequence {goal_sequence} doesn't match any trained sequence!")
        
    print(f"\nExecuting sequence {' -> '.join(goal_sequence)} using model {sequence_idx}")
    
    try:
        # Load the appropriate model
        model_path = os.path.join(models_dir, f'sequence_{sequence_idx}_final')
        model = PPO.load(model_path, device=device)
        print(f"Loaded model from {model_path}")
        
        # Create environment with this fixed sequence
        env = UnityGCRLLTLWrapper(
            env_path=unity_env_path,
            worker_id=100,  # Different from training worker_ids
            no_graphics=not render,  # Enable rendering if requested
            time_scale=1.0 if render else 20.0  # Slower if rendering
        )
        
        # Set the sequence
        env.set_fixed_goal_sequence(goal_sequence)
        
        # Execute the sequence
        obs = env.reset()[0]
        done = False
        total_reward = 0
        steps = 0
        achieved_goals = set()
        
        print("\nStarting execution:")
        while not done:
            # Get action from the sequence-specific model
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Track newly achieved goals
            current_goals = set(info.get('achieved_goals', []))
            new_goals = current_goals - achieved_goals
            if new_goals:
                for goal in new_goals:
                    print(f"Achieved goal: {goal} at step {steps}")
            achieved_goals = current_goals
            
            # Print progress periodically
            if steps % 100 == 0:
                print(f"Step {steps}: {len(achieved_goals)}/{len(goal_sequence)} goals achieved")
            
            # Check for timeout or completion
            if steps >= env.max_steps:
                print("\nExecution timed out!")
                break
                
        print("\nExecution completed!")
        print(f"Goals achieved: {len(achieved_goals)}/{len(goal_sequence)}")
        print(f"Total steps: {steps}")
        print(f"Total reward: {total_reward:.2f}")
        
        return {
            'success': len(achieved_goals) == len(goal_sequence),
            'goals_achieved': achieved_goals,
            'steps': steps,
            'reward': total_reward
        }
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        raise
    finally:
        env.close()

def execute_ltl_task(
    ltl_formula: str,
    unity_env_path: str,
    gcvf_model,
    render: bool = True
):
    """Complete pipeline from LTL to execution"""
    
    # Convert LTL to Büchi automaton
    automaton = ltl_to_buchi(ltl_formula)
    
    # Get value map from GCVF
    value_map = get_value_map(gcvf_model)
    
    # Find optimal path
    goals, avoid_zones = find_optimal_path(automaton, value_map)
    
    print(f"\nOptimal sequence found: {' -> '.join(goals)}")
    if avoid_zones:
        print(f"Zones to avoid: {avoid_zones}")
    
    # Execute the sequence
    result = execute_ltl_sequence(
        unity_env_path=unity_env_path,
        goal_sequence=goals,
        render=render
    )
    
    return result

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ltl', type=str, required=True,
                      help='LTL formula to execute')
    parser.add_argument('--unity_env_path', type=str, required=True,
                      help='Path to Unity environment')
    parser.add_argument('--gcvf_path', type=str, required=True,
                      help='Path to trained GCVF model')
    parser.add_argument('--render', action='store_true',
                      help='Enable Unity rendering')
    
    args = parser.parse_args()
    
    # Load GCVF model
    gcvf_model = torch.load(args.gcvf_path)
    
    # Execute task
    result = execute_ltl_task(
        ltl_formula=args.ltl,
        unity_env_path=args.unity_env_path,
        gcvf_model=gcvf_model,
        render=args.render
    )
    
    # Print results
    print("\nTask execution complete!")
    print(f"Success: {result['success']}")
    print(f"Goals achieved: {len(result['goals_achieved'])}")
    print(f"Total steps: {result['steps']}")
    print(f"Total reward: {result['reward']:.2f}")
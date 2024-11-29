import argparse
import os
import torch
import numpy as np
from stable_baselines3 import PPO
from envs.unity import UnityGCRLLTLWrapper
from algo.ltl import gltl2ba, get_ltl_args
from algo.scc import path_finding

def get_value_map(gcvf_model, initial_state):
    """Calculate value map using GCVF model"""
    value_map = {}
    zones = ['red', 'green', 'yellow']
    
    # Single zone values
    for zone in zones:
        with torch.no_grad():
            value_map[zone] = gcvf_model.predict(torch.cat([initial_state, zone]).unsqueeze(0))
    
    # Zone pair values
    for src_zone in zones:
        for dst_zone in zones:
            if src_zone != dst_zone:
                key = f"{src_zone}{dst_zone}"
                with torch.no_grad():
                    state_with_goals = torch.cat([
                        initial_state,
                        src_zone,
                        dst_zone
                    ]).unsqueeze(0)
                    value_map[key] = gcvf_model.predict(state_with_goals)
    
    return value_map

def find_optimal_sequence(ltl_formula, gcvf_model, initial_state):
    """Convert LTL to optimal goal sequence using GLTL2BA and GCVF"""
    try:
        # Convert LTL to Büchi automaton using GLTL2BA
        ltl_args = get_ltl_args(formula=ltl_formula)
        buchi_graph = gltl2ba(ltl_args)
        
        # Get value map from GCVF
        value_map = get_value_map(gcvf_model, initial_state)
        
        # Find optimal path through automaton
        goals, avoid_zones = path_finding(ltl_formula, value_map)
        
        return goals, avoid_zones
        
    except Exception as e:
        print(f"Error finding optimal sequence: {str(e)}")
        raise

def select_sequence_model(goals, models_dir='models'):
    """Select appropriate trained model based on goal sequence"""
    # Define all possible sequences in order of training
    all_sequences = [
        ['red', 'green', 'yellow'],
        ['red', 'yellow', 'green'],
        ['green', 'red', 'yellow'],
        ['green', 'yellow', 'red'],
        ['yellow', 'red', 'green'],
        ['yellow', 'green', 'red']
    ]
    
    # Find matching sequence
    sequence_idx = None
    for idx, seq in enumerate(all_sequences):
        if seq == goals:
            sequence_idx = idx
            break
            
    if sequence_idx is None:
        raise ValueError(f"No trained model found for sequence {' -> '.join(goals)}")
        
    model_path = os.path.join(models_dir, f'sequence_{sequence_idx}_final')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    return model_path

def execute_ltl_sequence(
    unity_env_path: str,
    model_path: str,
    goal_sequence: list,
    avoid_zones: list = None,
    device: str = 'cuda',
    render: bool = True
):
    """Execute a goal sequence using the appropriate trained model"""
    try:
        # Load model
        model = PPO.load(model_path, device=device)
        
        # Create environment
        env = UnityGCRLLTLWrapper(
            env_path=unity_env_path,
            worker_id=100,
            no_graphics=not render,
            time_scale=1.0 if render else 20.0
        )
        
        # Set sequence
        success = env.set_fixed_goal_sequence(goal_sequence)
        if not success:
            raise RuntimeError("Failed to set goal sequence")
        
        # Execute sequence
        obs = env.reset()[0]
        done = False
        total_reward = 0
        steps = 0
        achieved_goals = set()
        
        print(f"\nExecuting sequence: {' -> '.join(goal_sequence)}")
        if avoid_zones:
            print(f"Avoiding zones: {', '.join(avoid_zones)}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Track achievements
            current_goals = set(info.get('achieved_goals', []))
            new_goals = current_goals - achieved_goals
            if new_goals:
                for goal in new_goals:
                    print(f"Achieved goal: {goal} at step {steps}")
            achieved_goals = current_goals
            
            if steps % 100 == 0:
                print(f"Step {steps}: {len(achieved_goals)}/{len(goal_sequence)} goals achieved")
            
            if steps >= env.max_steps:
                print("\nExecution timed out!")
                break
        
        return {
            'success': len(achieved_goals) == len(goal_sequence),
            'goals_achieved': achieved_goals,
            'steps': steps,
            'reward': total_reward
        }
        
    finally:
        env.close()

def main(args):
    # Load GCVF model
    gcvf_model = torch.load(args.gcvf_path)
    
    # Get initial state (can be modified based on your needs)
    initial_state = torch.zeros(100)  # Adjust size as needed
    
    print(f"Processing LTL formula: {args.ltl}")
    
    # Find optimal sequence
    goals, avoid_zones = find_optimal_sequence(args.ltl, gcvf_model, initial_state)
    print(f"\nOptimal sequence found:")
    print(f"Goals: {' -> '.join(goals)}")
    if avoid_zones:
        print(f"Zones to avoid: {', '.join(avoid_zones)}")
    
    # Select appropriate model
    try:
        model_path = select_sequence_model(goals)
        print(f"Selected model: {model_path}")
        
        # Execute sequence
        result = execute_ltl_sequence(
            unity_env_path=args.unity_env_path,
            model_path=model_path,
            goal_sequence=goals,
            avoid_zones=avoid_zones,
            render=args.render
        )
        
        # Print results
        print("\nExecution Results:")
        print(f"Success: {result['success']}")
        print(f"Goals achieved: {len(result['goals_achieved'])}/{len(goals)}")
        print(f"Total steps: {result['steps']}")
        print(f"Total reward: {result['reward']:.2f}")
        
    except Exception as e:
        print(f"Error executing sequence: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ltl', type=str, required=True,
                      help='LTL formula to execute')
    parser.add_argument('--unity_env_path', type=str, required=True,
                      help='Path to Unity environment')
    parser.add_argument('--gcvf_path', type=str, required=True,
                      help='Path to trained GCVF model')
    parser.add_argument('--render', action='store_true',
                      help='Enable Unity rendering')
    parser.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nExecution failed: {str(e)}")
import argparse
import logging
import torch
from train_unity_agent import train_goal_conditioned_agent
from train_gcvf import main as train_value_function
from algo.ltl import gltl2ba, get_ltl_formula
from algo.scc import path_finding
from algo.reaching import reaching
from envs.unity import UnityGCRLLTLWrapper

def main(args):
    # Step 1: Train goal-conditioned agent and collect trajectory data
    if args.train_agent:
        if args.unity_env_path is None:
            raise ValueError("--unity_env_path is required for training the agent")
        print("\n=== Starting Training ===")
        model, trajectory_dataset = train_goal_conditioned_agent(
            unity_env_path=args.unity_env_path,
            total_timesteps=args.total_timesteps,
            num_envs=args.num_envs,
            device=args.device
        )
        
        print("\n=== Training Complete ===")
        print("Model saved to: models/trained_model")
        print("Dataset saved to: datasets/trajectory_dataset.pt")
    # Step 2: Train Goal-Conditioned Value Function
    if args.train_gcvf:
        if args.dataset_path is None:
            raise ValueError("--dataset_path is required for training GCVF")
        gcvf = train_value_function(
            dataset_path=args.dataset_path,
            device=args.device
        )

    # Step 3: Execute LTL task
    if args.execute_ltl:
        if any(arg is None for arg in [args.unity_env_path, args.model_path, args.gcvf_path, args.ltl_formula]):
            raise ValueError("--unity_env_path, --model_path, --gcvf_path, and --ltl_formula are required for LTL execution")
        # Convert LTL formula using ltl2ba
        ltl_args = get_ltl_formula(formula=args.ltl_formula)
        buchi_graph = gltl2ba(ltl_args)
        
        # Get value map for path finding
        value_map = get_value_map(model, gcvf, observation, zone_vector, args.device)
        
        # Find path that satisfies the LTL formula
        GOALS, AVOID_ZONES = path_finding(args.ltl_formula, value_map)
        
        # Execute the path in the environment
        env = UnityGCRLLTLWrapper(args.unity_env_path)
        task_info = reaching(env, model, GOALS, AVOID_ZONES, 
                           value_threshold=args.value_threshold, 
                           device=args.device)
        
        print(f"Task completion: {task_info['complete']}")
        print(f"Dangerous states encountered: {task_info['dangerous']}")
        if 'zone_history' in task_info:
            print(f"Zone history: {task_info['zone_history']}")

def get_value_map(model, gcvf, ob, zone_vector, device):
    """Calculate value map for path finding"""
    core_ob = ob[:-24]  # Remove zone vector from observation
    zones = ['green', 'red', 'yellow']
    value_map = {}
    
    # Add single zone values
    for zone in zones:
        value_map[zone] = model.predict_value(
            torch.cat([core_ob, zone_vector[zone]]).to(device)
        )
    
    # Add zone pair values
    for zone1 in zones:
        for zone2 in zones:
            if zone1 != zone2:
                map_ob = torch.cat([
                    core_ob, 
                    zone_vector[zone1],
                    zone_vector[zone2]
                ]).to(device)
                value_map[f"{zone1}{zone2}"] = gcvf.predict(map_ob)
    
    return value_map

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--unity_env_path", type=str,
                      help="Path to Unity environment executable")
    parser.add_argument("--train_agent", action="store_true")
    parser.add_argument("--train_gcvf", action="store_true")
    parser.add_argument("--execute_ltl", action="store_true")
    parser.add_argument("--ltl_formula", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gcvf_path", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total_timesteps", type=int, default=1e6)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--value_threshold", type=float, default=0.85)
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        logging.exception("Error during execution")
        raise
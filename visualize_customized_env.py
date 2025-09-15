#!/usr/bin/env python3
"""
Simple visualization script for CustomizedEnv without requiring trained models
This script provides random actions to demonstrate the environment
"""

import numpy as np
import time
import yaml
from robotarium_gym.scenarios.CustomizedEnv.customized_warehouse import CustomizedWarehouse
from robotarium_gym.utilities.misc import objectview

def run_visualization():
    """Run the customized environment with random actions for visualization"""
    
    # Load configuration
    with open('robotarium_gym/scenarios/CustomizedEnv/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    args = objectview(config)
    
    # Create environment
    env = CustomizedWarehouse(args)
    
    print("🚀 Starting CustomizedEnv Visualization")
    print(f"📊 Environment: {args.scenario}")
    print(f"🤖 Number of robots: {args.n_agents}")
    print(f"📏 Map boundaries: [{args.LEFT}, {args.RIGHT}] x [{args.UP}, {args.DOWN}]")
    print(f"🎯 Goal zones: {args.goal_width} units wide")
    print(f"📦 Obstacles: 5 rectangular obstacles")
    print("=" * 50)
    
    # Reset environment
    obs = env.reset()
    print("✅ Environment reset successfully")
    
    episode = 0
    step = 0
    
    try:
        while episode < args.episodes:
            episode += 1
            print(f"\n🎬 Episode {episode}/{args.episodes}")
            
            # Reset for new episode
            if episode > 1:
                obs = env.reset()
                step = 0
            
            done = False
            while not done and step < args.max_episode_steps:
                step += 1
                
                # Generate random actions for all agents
                actions = np.random.randint(0, 5, size=args.n_agents)
                
                # Take step
                obs, rewards, dones, info = env.step(actions)
                
                # Check if episode is done
                done = any(dones)
                
                # Print progress every 20 steps
                if step % 20 == 0:
                    print(f"  Step {step}: Rewards = {[f'{r:.2f}' for r in rewards]}")
                    if 'message' in info:
                        print(f"  Message: {info['message']}")
            
            print(f"  ✅ Episode {episode} completed in {step} steps")
            print(f"  📊 Final rewards: {[f'{r:.2f}' for r in rewards]}")
            if 'dist_travelled' in info:
                print(f"  📏 Distance travelled: {[f'{d:.2f}' for d in info['dist_travelled']]}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        print("\n🏁 Visualization completed")
        print("💡 Check the 'gifs' folder for saved GIFs if enabled")

if __name__ == "__main__":
    run_visualization()

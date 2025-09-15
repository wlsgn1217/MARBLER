#!/usr/bin/env python3
"""
Run CustomizedEnv with full visualization enabled
"""

import numpy as np
import yaml
from robotarium_gym.scenarios.CustomizedEnv.customized_warehouse import CustomizedWarehouse
from robotarium_gym.utilities.misc import objectview

def run_visualization():
    """Run the customized environment with full visualization"""
    
    # Load configuration
    with open('robotarium_gym/scenarios/CustomizedEnv/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable visualization
    config['show_figure_frequency'] = 1
    config['save_gif'] = True
    config['real_time'] = True
    config['episodes'] = 3  # Run fewer episodes for visualization
    config['max_episode_steps'] = 100  # Shorter episodes for better viewing
    
    args = objectview(config)
    
    # Create environment
    env = CustomizedWarehouse(args)
    
    print("🚀 Starting CustomizedEnv with Full Visualization")
    print(f"📊 Environment: {args.scenario}")
    print(f"🤖 Number of robots: {args.n_agents}")
    print(f"📏 Map boundaries: [{args.LEFT}, {args.RIGHT}] x [{args.UP}, {args.DOWN}]")
    print(f"🎯 Goal zones: {args.goal_width} units wide")
    print(f"📦 Obstacles: 5 rectangular obstacles")
    print(f"🎬 Episodes: {args.episodes}")
    print(f"⏱️  Real-time: {args.real_time}")
    print("=" * 60)
    print("🎥 Visualization will open in a new window...")
    print("📱 Watch the robots navigate around the obstacles!")
    print("=" * 60)
    
    # Reset environment
    obs = env.reset()
    print("✅ Environment reset successfully")
    print(f"📍 Initial robot positions:")
    for i in range(args.n_agents):
        pos = env.agent_poses[:2, i]
        print(f"  Robot {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
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
                print(f"📍 Robot positions after reset:")
                for i in range(args.n_agents):
                    pos = env.agent_poses[:2, i]
                    print(f"  Robot {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
            
            done = False
            while not done and step < args.max_episode_steps:
                step += 1
                
                # Generate random actions for all agents
                actions = np.random.randint(0, 5, size=args.n_agents)
                action_names = ['left', 'right', 'up', 'down', 'no_action']
                
                # Take step
                obs, rewards, dones, info = env.step(actions)
                
                # Check if episode is done
                done = any(dones)
                
                # Print progress every 20 steps
                if step % 20 == 0:
                    print(f"  Step {step}: Actions = {[action_names[a] for a in actions]}")
                    print(f"    Rewards = {[f'{r:.2f}' for r in rewards]}")
                    print(f"    Positions:")
                    for i in range(args.n_agents):
                        pos = env.agent_poses[:2, i]
                        loaded = "LOADED" if env.agents[i].loaded else "empty"
                        print(f"      Robot {i}: ({pos[0]:.3f}, {pos[1]:.3f}) [{loaded}]")
                    if 'message' in info:
                        print(f"    Message: {info['message']}")
            
            print(f"  ✅ Episode {episode} completed in {step} steps")
            print(f"  📊 Final rewards: {[f'{r:.2f}' for r in rewards]}")
            if 'dist_travelled' in info:
                print(f"  📏 Distance travelled: {[f'{d:.2f}' for d in info['dist_travelled']]}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Visualization stopped by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🏁 Visualization completed")
        print("💡 Check the 'gifs' folder for saved GIFs!")

if __name__ == "__main__":
    run_visualization()

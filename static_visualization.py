#!/usr/bin/env python3
"""
Generate static visualization of the CustomizedEnv with obstacles and robot positions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import yaml
from robotarium_gym.scenarios.CustomizedEnv.customized_warehouse import CustomizedWarehouse
from robotarium_gym.utilities.misc import objectview

def create_static_visualization():
    """Create a static plot showing the environment layout"""
    
    # Load configuration
    with open('robotarium_gym/scenarios/CustomizedEnv/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Disable visualization to avoid GUI issues
    config['show_figure_frequency'] = -1
    config['save_gif'] = False
    config['real_time'] = False
    config['episodes'] = 1
    config['max_episode_steps'] = 50
    
    args = objectview(config)
    
    # Create environment
    env = CustomizedWarehouse(args)
    
    print("🚀 Creating Static Visualization of CustomizedEnv")
    print(f"📊 Environment: {args.scenario}")
    print(f"🤖 Number of robots: {args.n_agents}")
    print(f"📏 Map boundaries: [{args.LEFT}, {args.RIGHT}] x [{args.UP}, {args.DOWN}]")
    print("=" * 60)
    
    # Reset environment
    obs = env.reset()
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Set up the plot
    ax.set_xlim([args.LEFT, args.RIGHT])
    ax.set_ylim([args.UP, args.DOWN])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('CustomizedEnv: Warehouse with Rectangular Obstacles', fontsize=16, fontweight='bold')
    ax.set_xlabel('X Position (meters)', fontsize=12)
    ax.set_ylabel('Y Position (meters)', fontsize=12)
    
    # Load goal zones from config
    goal_zones_config = getattr(args, 'goal_zones', [])
    if not goal_zones_config:
        # Default fallback if not defined in config
        w = args.goal_width
        goal_zones_config = [
            [-1.5, 0.2, w, 0.6, "load", "red"],     # Load Zone 1
            [-1.5, -0.8, w, 0.6, "load", "red"],   # Load Zone 2
            [0.2, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 1
            [0.5, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 2
            [0.8, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 3
            [1.1, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 4
        ]
    
    # Convert to visualization format
    goal_zones = []
    load_count = 1
    unload_count = 1
    for zone in goal_zones_config:
        if len(zone) >= 6:  # [x, y, width, height, type, color]
            x, y, w, h, zone_type, color = zone[:6]
            if zone_type == "load":
                label = f'Load Zone {load_count}'
                load_count += 1
            elif zone_type == "unload":
                label = f'Unload {unload_count}'
                unload_count += 1
            else:
                label = f'Zone {load_count + unload_count}'
            
            goal_zones.append({
                'rect': [x, y, w, h], 
                'color': color, 
                'alpha': 0.8, 
                'label': label
            })
    
    for zone in goal_zones:
        rect = patches.Rectangle(zone['rect'][:2], zone['rect'][2], zone['rect'][3], 
                               color=zone['color'], alpha=zone['alpha'], zorder=1)
        ax.add_patch(rect)
        # Add label
        center_x = zone['rect'][0] + zone['rect'][2]/2
        center_y = zone['rect'][1] + zone['rect'][3]/2
        ax.text(center_x, center_y, zone['label'], ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
    
    # Load obstacle positions from config
    obstacle_definitions = getattr(args, 'obstacle_positions', [])
    if not obstacle_definitions:
        # Default fallback if not defined in config
        obstacle_definitions = [
            [0.2, -0.8, 0.15, 1.6],    # Left column
            [0.5, -0.8, 0.15, 1.6],    # Second column
            [0.8, -0.8, 0.15, 1.6],    # Third column
            [1.1, -0.8, 0.15, 1.6],    # Right column
        ]
    
    colors = ['darkgray'] * len(obstacle_definitions)
    obstacle_labels = [f'Column {i+1}' for i in range(len(obstacle_definitions))]
    
    for i, (x, y, w, h) in enumerate(obstacle_definitions):
        rect = patches.Rectangle([x, y], w, h, color=colors[i], alpha=0.8, zorder=2)
        ax.add_patch(rect)
        # Add label
        ax.text(x + w/2, y + h/2, f'Obstacle {i+1}', ha='center', va='center', 
               fontsize=8, fontweight='bold', color='white')
    
    # Draw initial robot positions
    robot_colors = ['blue', 'orange', 'purple', 'brown']
    robot_markers = []
    
    for i in range(args.n_agents):
        pos = env.agent_poses[:2, i]
        marker = ax.scatter(pos[0], pos[1], s=200, marker='o', 
                           facecolors='none', edgecolors=robot_colors[i], 
                           linewidth=3, label=f'Robot {i}')
        robot_markers.append(marker)
        # Add robot label
        ax.text(pos[0], pos[1] + 0.1, f'R{i}', ha='center', va='bottom', 
               fontsize=10, fontweight='bold', color=robot_colors[i])
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Add environment info
    info_text = f"""Environment Configuration:
• Map Size: {args.RIGHT - args.LEFT:.1f}m × {args.DOWN - args.UP:.1f}m
• Robots: {args.n_agents}
• Obstacles: {len(obstacle_definitions)}
• Goal Zones: 4 (Loading/Unloading)
• Step Distance: {args.step_dist}m
• Collision Detection: Enabled
• Safe Spawning: Enabled"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('customized_env_layout.png', dpi=300, bbox_inches='tight')
    print("✅ Static visualization saved as 'customized_env_layout.png'")
    
    # Now run a short simulation and plot robot trajectories
    print("\n🎬 Running short simulation to show robot movement...")
    
    # Store robot positions for trajectory plotting
    trajectories = [[] for _ in range(args.n_agents)]
    
    for step in range(20):  # Short simulation
        # Generate random actions
        actions = np.random.randint(0, 5, size=args.n_agents)
        
        # Take step
        obs, rewards, dones, info = env.step(actions)
        
        # Store positions
        for i in range(args.n_agents):
            pos = env.agent_poses[:2, i]
            trajectories[i].append(pos.copy())
        
        if any(dones):
            break
    
    # Plot trajectories
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    ax2.set_xlim([args.LEFT, args.RIGHT])
    ax2.set_ylim([args.UP, args.DOWN])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Robot Trajectories in CustomizedEnv', fontsize=16, fontweight='bold')
    ax2.set_xlabel('X Position (meters)', fontsize=12)
    ax2.set_ylabel('Y Position (meters)', fontsize=12)
    
    # Draw goal zones (same as before)
    for zone in goal_zones:
        rect = patches.Rectangle(zone['rect'][:2], zone['rect'][2], zone['rect'][3], 
                               color=zone['color'], alpha=0.8, zorder=1)
        ax2.add_patch(rect)
    
    # Draw obstacles (same as before)
    for i, (x, y, w, h) in enumerate(obstacle_definitions):
        rect = patches.Rectangle([x, y], w, h, color=colors[i], alpha=0.8, zorder=2)
        ax2.add_patch(rect)
    
    # Plot trajectories
    for i, traj in enumerate(trajectories):
        if len(traj) > 1:
            traj_array = np.array(traj)
            ax2.plot(traj_array[:, 0], traj_array[:, 1], 
                    color=robot_colors[i], linewidth=2, alpha=0.7, 
                    label=f'Robot {i} Path')
            # Mark start and end points
            ax2.scatter(traj[0][0], traj[0][1], s=100, marker='o', 
                       color=robot_colors[i], edgecolors='black', linewidth=2, 
                       label=f'Robot {i} Start')
            ax2.scatter(traj[-1][0], traj[-1][1], s=100, marker='s', 
                       color=robot_colors[i], edgecolors='black', linewidth=2, 
                       label=f'Robot {i} End')
    
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig('customized_env_trajectories.png', dpi=300, bbox_inches='tight')
    print("✅ Robot trajectories saved as 'customized_env_trajectories.png'")
    
    print("\n🎉 Visualization complete!")
    print("📁 Generated files:")
    print("  • customized_env_layout.png - Environment layout with obstacles")
    print("  • customized_env_trajectories.png - Robot movement paths")
    
    return fig, fig2

if __name__ == "__main__":
    create_static_visualization()

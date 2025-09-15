from rps.utilities.misc import *
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from robotarium_gym.scenarios.base import BaseVisualization


class Visualize(BaseVisualization):
    """Customized visualization with rectangular obstacles and goal zones"""
    
    def __init__(self, args):
        self.args = args
        self.agent_marker_size_m = 0.15
        self.line_width = 3
        self.CM = plt.cm.get_cmap('hsv', 10)  # Agent/goal color scheme
        self.show_figure = True

    def initialize_markers(self, robotarium, agents):
        """Initialize all visual elements including obstacles and goal zones"""
        agent_marker_size = determine_marker_size(robotarium, self.agent_marker_size_m)

        # Initialize goal zones (loading/unloading areas)
        self.goals = []
        w = self.args.goal_width
        
        # Load goal zones from config
        goal_zones = getattr(self.args, 'goal_zones', [])
        if not goal_zones:
            # Default fallback if not defined in config
            goal_zones = [
                [-1.5, 0.2, w, 0.6, "load", "red"],     # Load Zone 1
                [-1.5, -0.8, w, 0.6, "load", "red"],   # Load Zone 2
                [0.2, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 1
                [0.5, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 2
                [0.8, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 3
                [1.1, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 4
            ]
        
        # Draw goal zones from config
        for zone in goal_zones:
            if len(zone) >= 6:  # [x, y, width, height, type, color]
                x, y, w, h, zone_type, color = zone[:6]
                self.goals.append(robotarium.axes.add_patch(
                    patches.Rectangle([x, y], w, h, 
                                    color=color, zorder=-1, alpha=0.8)
                ))

        # Add rectangular obstacles (matching the collision detection)
        self.obstacles = []
        
        # Load obstacle positions from config
        obstacle_definitions = getattr(self.args, 'obstacle_positions', [])
        if not obstacle_definitions:
            # Default fallback if not defined in config
            obstacle_definitions = [
                [0.2, -0.8, 0.15, 1.6],    # Left column
                [0.5, -0.8, 0.15, 1.6],    # Second column
                [0.8, -0.8, 0.15, 1.6],    # Third column
                [1.1, -0.8, 0.15, 1.6],    # Right column
            ]
        
        colors = ['darkgray'] * len(obstacle_definitions)
        
        for i, (x, y, w, h) in enumerate(obstacle_definitions):
            obstacle = robotarium.axes.add_patch(
                patches.Rectangle([x, y], w, h, 
                                color=colors[i], zorder=0, alpha=0.8)
            )
            self.obstacles.append(obstacle)

        # Initialize robot markers
        self.robot_markers = []
        for ii in range(agents.num_robots):
            marker = robotarium.axes.scatter(
                agents.agent_poses[0, ii], agents.agent_poses[1, ii],
                s=agent_marker_size, marker='o', facecolors='none',
                edgecolors=(self.CM(3) if ii % 2 == 0 else self.CM(0)), 
                linewidth=self.line_width
            )
            self.robot_markers.append(marker)

        # Add labels for goal zones
        load_count = 1
        unload_count = 1
        for zone in goal_zones:
            if len(zone) >= 6:  # [x, y, width, height, type, color]
                x, y, w, h, zone_type, color = zone[:6]
                if zone_type == "load":
                    robotarium.axes.text(x + w/2, y + h/2, f'Load\nZone {load_count}', 
                                       ha='center', va='center', fontsize=8,
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    load_count += 1
                elif zone_type == "unload":
                    robotarium.axes.text(x + w/2, y + h/2, f'Unload\n{unload_count}', 
                                       ha='center', va='center', fontsize=6,
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                    unload_count += 1

        # Set axis limits and labels
        robotarium.axes.set_xlim([-1.5, 1.5])
        robotarium.axes.set_ylim([-1, 1])
        robotarium.axes.set_aspect('equal')
        robotarium.axes.grid(True, alpha=0.3)
        robotarium.axes.set_title('Customized Warehouse Environment with Obstacles', fontsize=12)

    def update_markers(self, robotarium, agents):
        """Update robot positions and other dynamic elements"""
        for i in range(agents.agent_poses.shape[1]):
            # Update robot positions
            self.robot_markers[i].set_offsets(agents.agent_poses[:2, i].T)
            
            # Update marker sizes if figure window size changed
            self.robot_markers[i].set_sizes([determine_marker_size(robotarium, self.agent_marker_size_m)])
            
            # Change robot color if loaded
            if agents.agents[i].loaded:
                self.robot_markers[i].set_edgecolors('gold')
                self.robot_markers[i].set_linewidth(4)
            else:
                original_color = self.CM(3) if i % 2 == 0 else self.CM(0)
                self.robot_markers[i].set_edgecolors(original_color)
                self.robot_markers[i].set_linewidth(self.line_width)

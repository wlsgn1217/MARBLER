from gym import spaces
import numpy as np
import copy
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_gym.utilities.misc import *
from robotarium_gym.scenarios.CustomizedEnv.visualize import Visualize
from robotarium_gym.utilities.roboEnv import roboEnv


class Agent:
    """Customized agents for visualization-focused environment"""
    def __init__(self, index, action_id_to_word, goal='Red'):
        self.index = index
        self.goal = goal
        self.loaded = False
        self.action_id2w = action_id_to_word

    def generate_goal(self, goal_pose, action, args, obstacles):    
        """Updates the goal_pose based on the agent's actions with obstacle avoidance"""
        original_pose = copy.deepcopy(goal_pose)
        
        if self.action_id2w[action] == 'left':
            goal_pose[0] = max(goal_pose[0] - args.step_dist, args.LEFT)
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                  args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'right':
            goal_pose[0] = min(goal_pose[0] + args.step_dist, args.RIGHT)
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                  args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        elif self.action_id2w[action] == 'up':
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                  args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = max(goal_pose[1] - args.step_dist, args.UP)
        elif self.action_id2w[action] == 'down':
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                  args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = min(goal_pose[1] + args.step_dist, args.DOWN)
        else:  # no_action
            goal_pose[0] = args.LEFT if goal_pose[0] < args.LEFT else \
                  args.RIGHT if goal_pose[0] > args.RIGHT else goal_pose[0]
            goal_pose[1] = args.UP if goal_pose[1] < args.UP else \
                  args.DOWN if goal_pose[1] > args.DOWN else goal_pose[1]
        
        # Check if the new goal position collides with obstacles
        if self._check_obstacle_collision(goal_pose, obstacles):
            # If collision, try to find a safe position nearby
            safe_goal = self._find_safe_position(original_pose, goal_pose, obstacles, args)
            if safe_goal is not None:
                goal_pose = safe_goal
            else:
                # If no safe position found, stay in original position
                goal_pose = original_pose
        
        return goal_pose

    def _check_obstacle_collision(self, position, obstacles):
        """Check if a position collides with any obstacle"""
        x, y = position[0], position[1]
        robot_radius = 0.1  # Robot collision radius
        
        for obstacle in obstacles:
            obs_x, obs_y, obs_w, obs_h = obstacle
            # Check if robot (with radius) intersects with obstacle
            if (obs_x - robot_radius <= x <= obs_x + obs_w + robot_radius and
                obs_y - robot_radius <= y <= obs_y + obs_h + robot_radius):
                return True
        return False

    def _find_safe_position(self, original_pose, goal_pose, obstacles, args):
        """Find a safe position near the goal that doesn't collide with obstacles"""
        # Try positions around the goal in a small radius
        for radius in [0.05, 0.1, 0.15, 0.2]:
            for angle in np.linspace(0, 2*np.pi, 8):
                test_x = goal_pose[0] + radius * np.cos(angle)
                test_y = goal_pose[1] + radius * np.sin(angle)
                
                # Check boundaries
                if (args.LEFT <= test_x <= args.RIGHT and 
                    args.UP <= test_y <= args.DOWN):
                    
                    test_pos = np.array([test_x, test_y, 0])
                    if not self._check_obstacle_collision(test_pos, obstacles):
                        return test_pos
        
        return None


class CustomizedWarehouse(BaseEnv):
    """Customized warehouse environment focused on visualization with obstacles"""
    
    def __init__(self, args):
        self.args = args
        self.num_robots = self.args.n_agents
        self.agent_poses = None

        # Agent observation: [pos_x, pos_y, loaded] where loaded is a bool
        self.agent_obs_dim = 3 
        
        # Action mapping
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'no_action'}

        # Load obstacle positions from config
        self.obstacles = getattr(args, 'obstacle_positions', [])
        if not self.obstacles:
            # Default fallback if not defined in config
            self.obstacles = [
                [0.2, -0.8, 0.15, 1.6],    # Left column
                [0.5, -0.8, 0.15, 1.6],    # Second column
                [0.8, -0.8, 0.15, 1.6],    # Third column
                [1.1, -0.8, 0.15, 1.6],    # Right column
            ]

        if self.args.seed != -1:
             np.random.seed(self.args.seed)
        
        # Initialize agents
        self.agents = [Agent(i, self.action_id2w) for i in range(self.num_robots)]
        
        # Assign goals to agents (alternating pattern)
        for i, a in enumerate(self.agents):
            if i % 2 == 0:  # Even numbered agents get green zone
                a.goal = 'Green'
            else:  # Odd numbered agents get red zone
                a.goal = 'Red'

        # Initialize action and observation spaces
        actions = []
        observations = []
        for a in self.agents:
            actions.append(spaces.Discrete(5))
            # Each agent's observation includes neighbors
            obs_dim = self.agent_obs_dim * (self.args.num_neighbors + 1)
            observations.append(spaces.Box(low=-2.0, high=2.0, shape=(obs_dim,), dtype=np.float32))
        
        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))
        
        # Initialize visualizer and environment
        self.visualizer = Visualize(self.args)
        self.env = roboEnv(self, args)  

    def _is_position_safe(self, position, obstacles):
        """Check if a position is safe (not inside obstacles)"""
        x, y = position[0], position[1]
        robot_radius = 0.1  # Robot collision radius
        
        for obstacle in obstacles:
            obs_x, obs_y, obs_w, obs_h = obstacle
            # Check if robot (with radius) intersects with obstacle
            if (obs_x - robot_radius <= x <= obs_x + obs_w + robot_radius and
                obs_y - robot_radius <= y <= obs_y + obs_h + robot_radius):
                return False
        return True

    def _generate_safe_spawn_positions(self, num_robots, max_attempts=1000):
        """Generate safe spawn positions that don't collide with obstacles"""
        positions = []
        attempts = 0
        
        while len(positions) < num_robots and attempts < max_attempts:
            attempts += 1
            
            # Generate random position within boundaries
            x = np.random.uniform(self.args.LEFT + 0.2, self.args.RIGHT - 0.2)
            y = np.random.uniform(self.args.UP + 0.2, self.args.DOWN - 0.2)
            position = np.array([x, y, 0])
            
            # Check if position is safe (not in obstacles)
            if self._is_position_safe(position, self.obstacles):
                # Check if position is far enough from existing positions
                too_close = False
                for existing_pos in positions:
                    distance = np.linalg.norm(position[:2] - existing_pos[:2])
                    if distance < self.args.start_dist:
                        too_close = True
                        break
                
                if not too_close:
                    positions.append(position)
        
        if len(positions) < num_robots:
            print(f"Warning: Only generated {len(positions)} safe positions out of {num_robots} requested")
            # Fill remaining positions with boundary positions
            while len(positions) < num_robots:
                # Place robots at safe boundary positions
                if len(positions) % 4 == 0:
                    pos = np.array([self.args.LEFT + 0.3, self.args.UP + 0.3, 0])
                elif len(positions) % 4 == 1:
                    pos = np.array([self.args.RIGHT - 0.3, self.args.UP + 0.3, 0])
                elif len(positions) % 4 == 2:
                    pos = np.array([self.args.LEFT + 0.3, self.args.DOWN - 0.3, 0])
                else:
                    pos = np.array([self.args.RIGHT - 0.3, self.args.DOWN - 0.3, 0])
                positions.append(pos)
        
        return np.array(positions).T

    def reset(self):
        """Reset the environment to initial state"""
        self.episode_steps = 0
        for a in self.agents:
            a.loaded = False
        
        # Generate safe spawn positions
        self.agent_poses = self._generate_safe_spawn_positions(self.num_robots)
        
        # Adjust poses to match Robotarium coordinate system
        self.agent_poses[0] += (1.5 + self.args.LEFT) / 2
        self.agent_poses[0] -= (1.5 - self.args.RIGHT) / 2
        self.agent_poses[1] -= (1 + self.args.UP) / 2
        self.agent_poses[1] += (1 - self.args.DOWN) / 2
        
        self.env.reset()
        return [[0] * (self.agent_obs_dim * (self.args.num_neighbors + 1))] * self.num_robots
    
    def step(self, actions_):
        """Take a step in the environment"""
        self.episode_steps += 1
        info = {}

        # Execute actions in Robotarium
        message, dist, frames = self.env.step(actions_) 

        obs = self.get_observations()
        
        if message == '':
            rewards = self.get_rewards()       
            terminated = self.episode_steps > self.args.max_episode_steps
        else:
            print("Ending due to", message)
            info['message'] = message
            rewards = [-5] * self.num_robots
            terminated = True
        
        info['dist_travelled'] = dist
        if self.args.save_gif:
            info['frames'] = frames
            
        return obs, rewards, [terminated] * self.num_robots, info
    
    def get_observations(self):
        """Get observations for all agents"""
        observations = []
        for a in self.agents:
            observations.append([*self.agent_poses[:, a.index][:2], a.loaded])

        # Add neighbor observations
        full_observations = []
        for i, agent in enumerate(self.agents):
            full_observations.append(observations[agent.index])
            
            if self.args.num_neighbors >= self.num_robots - 1:
                nbr_indices = [i for i in range(self.num_robots) if i != agent.index]
            else:
                nbr_indices = get_nearest_neighbors(self.agent_poses, agent.index, self.args.num_neighbors)
            
            for nbr_index in nbr_indices:
                full_observations[i] = np.concatenate((full_observations[i], observations[nbr_index]))
                
        return full_observations

    def get_rewards(self):
        """Calculate rewards based on goal achievement"""
        rewards = []
        
        # Load goal zones from config
        goal_zones = getattr(self.args, 'goal_zones', [])
        if not goal_zones:
            # Default fallback if not defined in config
            goal_zones = [
                [-1.5, 0.2, self.args.goal_width, 0.6, "load", "red"],     # Load Zone 1
                [-1.5, -0.8, self.args.goal_width, 0.6, "load", "red"],   # Load Zone 2
                [0.2, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 1
                [0.5, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 2
                [0.8, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 3
                [1.1, -0.9, 0.15, 0.2, "unload", "green"],   # Unload 4
            ]
        
        for a in self.agents:
            pos = self.agent_poses[:, a.index][:2]
            
            if a.loaded:             
                # Check if in unloading zone
                in_unload_zone = False
                for zone in goal_zones:
                    if len(zone) >= 5 and zone[4] == "unload":  # Check if it's an unload zone
                        if (zone[0] <= pos[0] <= zone[0] + zone[2] and 
                            zone[1] <= pos[1] <= zone[1] + zone[3]):
                            rewards.append(self.args.unload_reward)
                            a.loaded = False
                            in_unload_zone = True
                            break
                
                if not in_unload_zone:
                    rewards.append(0)
            else:
                # Check if in loading zone
                in_load_zone = False
                for zone in goal_zones:
                    if len(zone) >= 5 and zone[4] == "load":  # Check if it's a load zone
                        if (zone[0] <= pos[0] <= zone[0] + zone[2] and 
                            zone[1] <= pos[1] <= zone[1] + zone[3]):
                            rewards.append(self.args.load_reward)
                            a.loaded = True
                            in_load_zone = True
                            break
                
                if not in_load_zone:
                    rewards.append(0)
        return rewards

    def _generate_step_goal_positions(self, actions):
        """Generate goal positions for each agent based on actions with obstacle avoidance"""
        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:, i] = agent.generate_goal(goal[:, i], actions[i], self.args, self.obstacles)
        return goal

    def get_observation_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space
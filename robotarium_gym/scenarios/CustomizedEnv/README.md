# CustomizedEnv - Visualization-Focused Warehouse Environment

This is a customized warehouse environment designed for visualization and demonstration purposes, featuring rectangular obstacles and non-grid-based robot movement.

## Features

### 🏗️ **Environment Layout**
- **Map Size**: Configurable boundaries (default: 2.8m x 1.8m)
- **Obstacles**: 5 rectangular obstacles of different sizes and positions
- **Goal Zones**: 4 colored zones for loading/unloading tasks
- **Robots**: 4 robots with continuous movement (not grid-based)

### 🤖 **Robot Movement**
- **Actions**: 5 discrete actions (left, right, up, down, no_action)
- **Movement**: Continuous goal-based navigation
- **Collision Avoidance**: Built-in barrier certificates
- **Speed**: Configurable step distance (default: 0.15m per action)

### 🎯 **Goal System**
- **Loading Zones**: Right side (Red/Green zones)
- **Unloading Zones**: Left side (Red/Green zones)
- **Rewards**: Configurable rewards for goal achievement
- **Visual Feedback**: Robots change color when loaded

### 📦 **Obstacles**
The environment includes 5 rectangular obstacles:
1. **Central Obstacle**: Large black rectangle (1.2m x 0.6m)
2. **Left Obstacle**: Medium gray rectangle (0.4m x 0.4m)
3. **Right Obstacle**: Small gray rectangle (0.3m x 0.5m)
4. **Top Obstacle**: Small gray rectangle (0.4m x 0.3m)
5. **Bottom Obstacle**: Medium gray rectangle (0.5m x 0.2m)

## Usage

### Quick Start
```bash
# Activate the environment
conda activate marbler

# Run with random actions (no training required)
python3 visualize_customized_env.py

# Or run with the main system (requires model files)
python3 -m robotarium_gym.main --scenario CustomizedEnv
```

### Configuration

Edit `config.yaml` to customize:

```yaml
# Environment size
LEFT: -1.4
RIGHT: 1.4
UP: -0.9
DOWN: 0.9

# Robot settings
n_agents: 4
step_dist: 0.15
start_dist: 0.4

# Visualization
show_figure_frequency: 1
save_gif: True
real_time: False

# Goal zones
goal_width: 0.4
load_reward: 1
unload_reward: 3
```

## Customization

### Adding More Obstacles
Edit `visualize.py` in the `initialize_markers` method:

```python
# Add a new obstacle
obstacle = robotarium.axes.add_patch(
    patches.Rectangle([x, y], width, height, 
                     color='color', zorder=0, alpha=0.8)
)
self.obstacles.append(obstacle)
```

### Changing Map Size
Update the boundaries in `config.yaml`:
```yaml
LEFT: -2.0    # Left boundary
RIGHT: 2.0    # Right boundary
UP: -1.0      # Top boundary
DOWN: 1.0     # Bottom boundary
```

### Modifying Robot Behavior
Edit `customized_warehouse.py`:
- Change action mapping in `action_id2w`
- Modify goal generation in `generate_goal` method
- Adjust rewards in `get_rewards` method

## File Structure

```
CustomizedEnv/
├── customized_warehouse.py  # Main environment class
├── visualize.py             # Visualization with obstacles
├── config.yaml             # Configuration file
├── models/                 # Model files (copied from Warehouse)
│   ├── vdn.json
│   └── vdn.th
└── README.md              # This file
```

## Visualization Features

- **Real-time Robot Tracking**: Live position updates
- **Obstacle Rendering**: Static rectangular obstacles
- **Goal Zone Highlighting**: Color-coded loading/unloading areas
- **Load Status**: Robots change appearance when loaded
- **GIF Generation**: Automatic episode recording
- **Grid Overlay**: Optional grid for reference

## Troubleshooting

### Common Issues

1. **Model Dimension Mismatch**: Use the visualization script instead of main.py
2. **Import Errors**: Ensure all files are in the correct locations
3. **Visualization Not Showing**: Check `show_figure_frequency` in config
4. **Robots Not Moving**: Verify `update_frequency` and `step_dist` settings

### Performance Tips

- Set `real_time: False` for faster visualization
- Increase `update_frequency` for smoother movement
- Decrease `gif_frequency` to reduce file size
- Use `show_figure_frequency: -1` to disable visualization for training

## Examples

### Basic Visualization
```python
from robotarium_gym.scenarios.CustomizedEnv.customized_warehouse import CustomizedWarehouse
from robotarium_gym.utilities.misc import objectview
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
args = objectview(config)

# Create and run environment
env = CustomizedWarehouse(args)
obs = env.reset()

# Run with random actions
for step in range(100):
    actions = np.random.randint(0, 5, size=args.n_agents)
    obs, rewards, dones, info = env.step(actions)
    if any(dones):
        obs = env.reset()
```

This environment is perfect for:
- 🎓 **Educational demonstrations**
- 🔬 **Research prototyping**
- 🎮 **Interactive visualization**
- 📊 **Algorithm testing**
- 🏗️ **Environment design**

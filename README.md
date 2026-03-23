# Multi-Agent Deadlock Fix

## Overview

This project demonstrates and compares multi-agent navigation behavior in constrained maps,
with a focus on deadlock formation and deadlock resolution.

Core pipeline:

1. A global path is generated with sampling-based planning (RRT variants).
2. The path is smoothed and followed using B-Splines.
3. Artificial Potential Fields and Randomized Pushback Velocities are used to guarantee collision-free deadlock resolutions
4. 3 maps are included to demonstrate deadlock resolutions.
5. Simulations are animated and saved for visual analysis.

## Usage

Preferred Python environment: `3.13`

Install dependencies (`uv` is recommended):

`uv pip sync requirements.txt`

Run scenarios:

`python narrow_corridor.py`

`python 4agents.py`

`python maze.py`

Optional baseline deadlock demo:

`python baseline_deadlock_demo.py`

## Project Structure

```text
multiAgentDeadlockFix/
├── narrow_corridor.py
├── 4agents.py
├── maze.py
├── baseline_deadlock_demo.py
├── rrt_bridge.py
├── tube_bspline_short.py
├── plot_maps.py
├── requirements.txt
├── README.md
└── output/
	├── deadlock/
	└── maps/
```

### File Guide

- `narrow_corridor.py`: Two-agent corridor scenario using hybrid APF + global path planning.
- `4agents.py`: Four-agent intersection stress test for multi-agent coordination and deadlock handling.
- `maze.py`: Chicane/maze-style scenario for narrow passage interactions.
- `baseline_deadlock_demo.py`: Baseline APF behavior with symmetry preserved to intentionally show deadlock.
- `rrt_bridge.py`: RRT* + bridge sampling global planner for narrow-passage path generation.
- `tube_bspline_short.py`: Local short-horizon B-spline tube planner with obstacle-aware constraints.
- `plot_maps.py`: Utility script to render and save static map layouts.
- `output/`: Generated simulation outputs (videos/plots).
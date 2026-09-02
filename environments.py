import numpy as np

class DynamicAgent:
    def __init__(self, id, start, goal, color):
        self.id = id
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.color = color
        self.global_path = None

        # Private trial schedule: fresh push direction each period, redrawn until free
        self.trial_period = np.random.uniform(1.5, 3.0)
        self.trial_angles = np.random.uniform(0, 2*np.pi, 1000)
        self.trial_offset = np.random.uniform(0, self.trial_period)

        # Stretch oscillator for the spline control point
        self.random_stretch_freq = np.random.uniform(0.5, 1.5)
        self.random_stretch_phase = np.random.uniform(0, 2*np.pi)

def get_scenario(scenario_name, gap=1.6):
    """Returns obstacles and agent configurations for a given map."""
    # Corridor
    if scenario_name == "corridor":
        obstacles = [
            (4.8, 5.0 + gap/2, 0.4, 7.0 - gap/2), (4.8, -2.0, 0.4, 7.0 - gap/2),
            (-2.0, 11.0, 14.0, 0.5), (-2.0, -1.5, 14.0, 0.5),
            (-1.5, -1.5, 0.5, 13.0), (11.0, -1.5, 0.5, 13.0)
        ]
        agents = [
            DynamicAgent(1, [0.0, 5.0], [10.0, 5.0], 'blue'),
            DynamicAgent(2, [10.0, 5.0], [0.0, 5.0], 'red')
        ]
    # Maze
    elif scenario_name == "maze":
        obstacles = [
            (-2.0, 7.0, 14.0, 1.0), (-2.0, 2.0, 14.0, 1.0),
            (3.0, 3.0 + gap, 1.0, 4.0 - gap), (6.0, 3.0, 1.5, 4.0 - gap),
        ]
        agents = [
            DynamicAgent(1, [0.0, 5.0], [10.0, 5.0], 'blue'),
            DynamicAgent(2, [10.0, 5.0], [0.0, 5.0], 'red')
        ]
    # Warehouse: two aisles of width gap, joined only at the ends
    elif scenario_name == "warehouse":
        h = 1.8
        y0, y1, y2 = 1.0, 1.0 + h + gap, 1.0 + 2*(h + gap)
        obstacles = [
            (1.0, y0, 8.0, h), (1.0, y1, 8.0, h), (1.0, y2, 8.0, h),
            (-2.0, 11.0, 14.0, 0.5), (-2.0, -1.5, 14.0, 0.5),
            (-1.5, -1.5, 0.5, 13.0), (11.0, -1.5, 0.5, 13.0)
        ]
        a_y, b_y = y0 + h + gap/2, y1 + h + gap/2
        agents = [
            DynamicAgent(1, [0.0, a_y], [10.0, b_y], 'blue'),
            DynamicAgent(2, [10.0, a_y], [0.0, b_y], 'red'),
            DynamicAgent(3, [0.0, b_y], [10.0, a_y], 'green'),
            DynamicAgent(4, [10.0, b_y], [0.0, a_y], 'purple')
        ]
    # Intersection
    else:
        obstacles = [
            (-2.0, -2.0, 6.0, 6.0), (-2.0, 6.0, 6.0, 6.0),   
            (6.0, -2.0, 6.0, 6.0), (6.0, 6.0, 6.0, 6.0),    
        ]
        agents = [
            DynamicAgent(1, [0.0, 5.0], [10.0, 5.0], 'blue'),
            DynamicAgent(2, [10.0, 5.0], [0.0, 5.0], 'red'),
            DynamicAgent(3, [5.0, 0.0], [5.0, 10.0], 'green'),
            DynamicAgent(4, [5.0, 10.0], [5.0, 0.0], 'purple')
        ]
    return agents, obstacles
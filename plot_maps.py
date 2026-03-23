import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_map(obstacles, agents, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw physical obstacles
    for (ox, oy, w, h) in obstacles:
        ax.add_patch(patches.Rectangle((ox, oy), w, h, color='black', alpha=0.8))
        
    # Draw agents (Start circles and Goal stars)
    for agent in agents:
        start = agent['start']
        goal = agent['goal']
        color = agent['color']
        
        # Start position (Large Circle)
        ax.scatter(start[0], start[1], marker='o', s=200, color=color, edgecolors='white', linewidths=1.5, zorder=4)
        ax.text(start[0], start[1] + 0.5, 'Start', color=color, fontsize=12, fontweight='bold', ha='center', va='bottom')
        
        # Goal position (Large Star)
        ax.scatter(goal[0], goal[1], marker='*', s=400, color='yellow', edgecolors=color, linewidths=2, zorder=5)
        ax.text(goal[0], goal[1] - 0.5, 'Goal', color=color, fontsize=12, fontweight='bold', ha='center', va='top')

    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlim(-2.0, 12.0)
    ax.set_ylim(-2.0, 12.0)
    ax.set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot automatically
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    # ---------------------------
    # Map 1: 1.5m Narrow Corridor
    # ---------------------------
    obs1 = [
        (4.8, 5.75, 0.4, 6.25),   
        (4.8, -2.0, 0.4, 6.25),   
        (-2.0, 11.0, 14.0, 0.5),  
        (-2.0, -1.5, 14.0, 0.5),  
        (-1.5, -1.5, 0.5, 13.0),  
        (11.0, -1.5, 0.5, 13.0)   
    ]
    agents1 = [
        {'start': [0.0, 5.0], 'goal': [10.0, 5.0], 'color': 'blue'},
        {'start': [10.0, 5.0], 'goal': [0.0, 5.0], 'color': 'red'}
    ]
    plot_map(obs1, agents1, "Initial Setup: 1.5m Narrow Corridor", "map1_corridor.png")

    # ---------------------------
    # Map 2: 4-Way Intersection
    # ---------------------------
    obs2 = [
        (-2.0, -2.0, 6.0, 6.0),  
        (-2.0, 6.0, 6.0, 6.0),   
        (6.0, -2.0, 6.0, 6.0),   
        (6.0, 6.0, 6.0, 6.0)     
    ]
    agents2 = [
        {'start': [0.0, 5.0], 'goal': [10.0, 5.0], 'color': 'blue'},
        {'start': [10.0, 5.0], 'goal': [0.0, 5.0], 'color': 'red'},
        {'start': [5.0, 0.0], 'goal': [5.0, 10.0], 'color': 'green'},
        {'start': [5.0, 10.0], 'goal': [5.0, 0.0], 'color': 'purple'}
    ]
    plot_map(obs2, agents2, "Initial Setup: 4-Way Intersection", "map2_intersection.png")

    # ---------------------------
    # Map 3: The Narrow Chicane
    # ---------------------------
    obs3 = [
        (-2.0, 7.0, 14.0, 1.0),   
        (-2.0, 2.0, 14.0, 1.0),   
        (3.0, 4.5, 1.5, 2.5),     
        (6.0, 3.0, 1.5, 2.5)      
    ]
    agents3 = [
        {'start': [0.0, 5.0], 'goal': [10.0, 5.0], 'color': 'blue'},
        {'start': [10.0, 5.0], 'goal': [0.0, 5.0], 'color': 'red'}
    ]
    plot_map(obs3, agents3, "Initial Setup: The Narrow Chicane", "map3_chicane.png")
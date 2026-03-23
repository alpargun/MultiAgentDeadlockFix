import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# You can keep these imported if you want to integrate this back into your main framework later
# from rrt_bridge import RRTStarBridge
# from tube_bspline_short import TubeBSplineShort

np.random.seed(42)

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
ROBOT_RADIUS = 0.35    
GOAL_TOLERANCE = 0.05  

# ==========================================
# STANDARD APF (Chaos Disabled)
# ==========================================
class StandardAPF:
    """
    A pure Artificial Potential Field with the Stochastic State Machine completely disabled.
    This guarantees a deadlock in symmetric environments to serve as a baseline demo.
    """
    def __init__(self, k_att=2.0, k_rep_agent=5.0, k_rep_wall=8.0, r_base=ROBOT_RADIUS, kappa=0.5):
        self.k_att = k_att               
        self.k_rep_agent = k_rep_agent   
        self.k_rep_wall = k_rep_wall     
        self.r_base = r_base             
        self.kappa = kappa               
        self.d_safe_wall = ROBOT_RADIUS + 0.05 

    def get_closest_point_on_rect(self, pos, rect):
        rx, ry, rw, rh = rect
        cx = np.clip(pos[0], rx, rx + rw)
        cy = np.clip(pos[1], ry, ry + rh)
        return np.array([cx, cy])

    def get_desired_velocity(self, agent, p_target, other_agents, obstacles):
        p_i = agent.pos
        dist_to_final_goal = np.linalg.norm(agent.goal - p_i)

        if dist_to_final_goal <= GOAL_TOLERANCE:
            return np.zeros(2)

        v_max = 1.5
        arrival_radius = 1.0
        target_speed = v_max * (dist_to_final_goal / arrival_radius) if dist_to_final_goal < arrival_radius else v_max
        
        dir_to_waypoint = p_target - p_i
        dist_to_waypoint = np.linalg.norm(dir_to_waypoint)
        dir_target_norm = dir_to_waypoint / dist_to_waypoint if dist_to_waypoint > 0 else np.zeros(2)
        v_att = dir_target_norm * target_speed

        # ---------------------------------------------------------
        # PHASE 2: HYPER-STIFF CUBIC REPULSION
        # ---------------------------------------------------------
        # --- THE PERFECT DEADLOCK FIX ---
        # We remove the velocity-dependent (pulsating) forcefield. 
        # By setting a constant, rigid safety radius (Robot Radius + 0.5m buffer), 
        # the forces will scale perfectly with distance without oscillating, 
        # resulting in a flawless, frozen mathematical equilibrium.
        
        static_d_safe = self.r_base + 0.5 

        v_rep_agent = np.zeros(2)
        for other in other_agents:
            dist_ij = np.linalg.norm(p_i - other.pos)
            if 0.01 < dist_ij < static_d_safe:  # Use the static boundary here!
                rep_mag = self.k_rep_agent * ((1.0/dist_ij - 1.0/static_d_safe)**3) * (1.0/(dist_ij**2))
                rep_mag = min(rep_mag, 20.0) 
                v_rep_agent += rep_mag * ((p_i - other.pos) / dist_ij)

        v_rep_wall = np.zeros(2)
        for obs in obstacles:
            closest_pt = self.get_closest_point_on_rect(p_i, obs)
            dist_wall = np.linalg.norm(p_i - closest_pt)
            if 0.01 < dist_wall < self.d_safe_wall:
                rep_mag = self.k_rep_wall * ((1.0/dist_wall - 1.0/self.d_safe_wall)**3) * (1.0/(dist_wall**2))
                rep_mag = min(rep_mag, 15.0) 
                v_rep_wall += rep_mag * ((p_i - closest_pt) / dist_wall)

        # ---------------------------------------------------------
        # CHAOS MODE DISABLED
        # ---------------------------------------------------------
        agent.mode = 1 
        v_final = v_att + v_rep_agent + v_rep_wall
        
        # --- THE ANTI-JITTER FIX: NUMERICAL DAMPING ---
        # Because we take discrete steps, the robot jumps slightly *past* # the exact mathematical zero-point, causing it to bounce. 
        # If the net force is pushing us backwards, we apply massive friction 
        # to simulate continuous physics settling gracefully into the trap.
        forward_progress = np.dot(v_final, dir_target_norm)
        if forward_progress < 0:
            v_final *= 0.05 # 95% velocity reduction when pushed backwards
            
        speed = np.linalg.norm(v_final)
        if speed > target_speed:
            v_final = (v_final / speed) * target_speed

        return v_final

# ==========================================
# AGENT SETUP
# ==========================================
class DynamicAgent:
    def __init__(self, id, start, goal, color):
        self.id = id
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.color = color
        self.global_path = None
        self.history = [self.pos.copy()]
        self.mode = 1           
        self.mode_history = [self.mode] 

def get_current_target(agent):
    dists = np.linalg.norm(agent.global_path - agent.pos, axis=1)
    closest_idx = np.argmin(dists)
    target_idx = min(closest_idx + 3, len(agent.global_path) - 1)
    return agent.global_path[target_idx]

# ==========================================
# SIMULATION EXECUTION
# ==========================================
def run_simulation():
    obstacles = [
        (-2.0, -2.0, 6.0, 6.0),  
        (-2.0, 6.0, 6.0, 6.0),   
        (6.0, -2.0, 6.0, 6.0),   
        (6.0, 6.0, 6.0, 6.0),    
    ]
    
    # PERFECT SYMMETRY: No offsets used here to guarantee mathematical deadlock
    agents = [
        DynamicAgent(1, [0.0, 5.0], [10.0, 5.0], 'blue'),
        DynamicAgent(2, [10.0, 5.0], [0.0, 5.0], 'red'),
        DynamicAgent(3, [5.0, 0.0], [5.0, 10.0], 'green'),
        DynamicAgent(4, [5.0, 10.0], [5.0, 0.0], 'purple')
    ]
    
    print("Computing Perfect Straight-Line Paths for Baseline Demo...")
    # RRT bypassed. Mathematical straight lines ensure no symmetry-breaking wiggles.
    for a in agents:
        xs = np.linspace(a.pos[0], a.goal[0], 100)
        ys = np.linspace(a.pos[1], a.goal[1], 100)
        a.global_path = np.column_stack((xs, ys))

    print("Running Background Simulation...")
    apf = StandardAPF(r_base=ROBOT_RADIUS) 
    
    # LOW TIME STEP: Prevents teleporting through collision boundaries
    dt = 0.1 
    
    for step in range(800): 
        all_parked = all(np.linalg.norm(a.pos - a.goal) <= GOAL_TOLERANCE for a in agents)
        if all_parked:
            print(f"Simulation completed successfully! Both agents parked at step {step}.")
            break
            
        for i, agent in enumerate(agents):
            dist_to_final_goal = np.linalg.norm(agent.pos - agent.goal)
            
            if dist_to_final_goal <= GOAL_TOLERANCE:
                agent.history.append(agent.pos.copy())
                agent.mode_history.append(1)
                continue 
                
            other_agents = [a for a in agents if a.id != agent.id]
            target_wp = get_current_target(agent)
            
            v_des = apf.get_desired_velocity(agent, target_wp, other_agents, obstacles)
            
            if np.linalg.norm(v_des) == 0:
                agent.history.append(agent.pos.copy())
                agent.mode_history.append(1)
                continue
                
            # BYPASS B-SPLINE: Force robots to rigidly follow standard APF logic
            agent.pos = agent.pos + (v_des * dt) 
            
            agent.history.append(agent.pos.copy())
            agent.mode_history.append(agent.mode)

    print("Rendering Animation...")
    fig, ax = plt.subplots(figsize=(8, 8)) 
    
    for (ox, oy, w, h) in obstacles:
        ax.add_patch(patches.Rectangle((ox, oy), w, h, color='black', alpha=0.8))
        
    for a in agents:
        ax.plot(a.global_path[:,0], a.global_path[:,1], color=a.color, linestyle=':', alpha=0.3)
        ax.scatter(a.history[0][0], a.history[0][1], marker='o', s=100, color=a.color, edgecolors='black', zorder=4)
        ax.scatter(a.goal[0], a.goal[1], marker='*', s=300, color='yellow', edgecolors=a.color, linewidths=2, zorder=5)

    ax.set_title("Standard APF Failure: Perfect Deadlock", fontsize=14)
    ax.set_xlim(-1.0, 11.0)
    ax.set_ylim(-1.0, 11.0)
    ax.set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.6)

    mode_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', fontsize=12, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), zorder=10)
    
    lines = []
    heads = []
    for a in agents:
        line, = ax.plot([], [], color=a.color, lw=2.5, zorder=3)
        head, = ax.plot([], [], 'o', color=a.color, markersize=10, markeredgecolor='white', zorder=5)
        lines.append(line)
        heads.append(head)

    def update(frame):
        for i, a in enumerate(agents):
            hist = np.array(a.history[:frame+1])
            lines[i].set_data(hist[:, 0], hist[:, 1])
            heads[i].set_data([hist[-1, 0]], [hist[-1, 1]])
        mode_text.set_text(f"Step: {frame} | ALL AGENTS DEADLOCKED")
        return lines + heads + [mode_text]

    max_frames = max(len(a.history) for a in agents)
    ani = FuncAnimation(fig, update, frames=max_frames, blit=False, interval=50, repeat=False)
    ani.save('baseline_deadlock_4agents.mp4', writer='ffmpeg', fps=30)
    print("Saved baseline_deadlock_4agents.mp4")
    
    # ---------------------------------------------------------
    # ERROR CONVERGENCE PLOT
    # ---------------------------------------------------------
    print("Rendering Error Plot...")
    fig_error, ax_error = plt.subplots(figsize=(10, 5))
    
    for a in agents:
        distances = [np.linalg.norm(pos - a.goal) for pos in a.history]
        ax_error.plot(range(len(distances)), distances, label=f'Agent {a.id}', color=a.color, lw=2)
        
    ax_error.set_title("Distance to Goal: Standard APF Deadlock", fontsize=14, fontweight='bold')
    ax_error.set_xlabel("Simulation Step", fontsize=12)
    ax_error.set_ylabel("Distance to Goal (m)", fontsize=12)
    ax_error.grid(True, linestyle='--', alpha=0.6)
    ax_error.legend()
    
    plt.tight_layout()
    plt.savefig('baseline_error_plot_4agents.png', dpi=300)
    print("Saved baseline_error_plot_4agents.png")

    # Show the final error plot
    plt.show()

if __name__ == "__main__":
    run_simulation()
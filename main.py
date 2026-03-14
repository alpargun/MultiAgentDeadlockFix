import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

from rrt_bridge import RRTStarBridge
from tube_bspline_short import TubeBSplineShort

# ==========================================
# 2. RRT PATH SMOOTHER
# ==========================================
def smooth_global_path(path, num_points=100):
    clean_path = [path[0]]
    for pt in path[1:]:
        if np.linalg.norm(pt - clean_path[-1]) > 0.05:
            clean_path.append(pt)
    clean_path = np.array(clean_path)

    if len(clean_path) <= 3:
        dists = np.cumsum(np.linalg.norm(np.diff(clean_path, axis=0), axis=1))
        dists = np.insert(dists, 0, 0)
        if dists[-1] == 0: 
            return clean_path
        fx = si.interp1d(dists, clean_path[:,0], kind='linear')
        fy = si.interp1d(dists, clean_path[:,1], kind='linear')
        sample_dists = np.linspace(0, dists[-1], num_points)
        return np.column_stack((fx(sample_dists), fy(sample_dists)))

    tck, u = si.splprep([clean_path[:,0], clean_path[:,1]], s=2.0, k=3)
    u_new = np.linspace(0, 1.0, num_points)
    x_smooth, y_smooth = si.splev(u_new, tck)
    return np.column_stack((x_smooth, y_smooth))

# ==========================================
# 3. 2-MODE HYBRID APF
# ==========================================
class TwoModeAPF:
    def __init__(self, k_att=2.0, k_rep_agent=5.0, k_rep_wall=8.0, r_base=0.8, kappa=0.5, deadlock_tol=0.2):
        self.k_att = k_att
        self.k_rep_agent = k_rep_agent
        self.k_rep_wall = k_rep_wall
        self.r_base = r_base
        self.kappa = kappa
        self.deadlock_tol = deadlock_tol
        self.d_safe_wall = 0.6 

    def get_closest_point_on_rect(self, pos, rect):
        rx, ry, rw, rh = rect
        cx = np.clip(pos[0], rx, rx + rw)
        cy = np.clip(pos[1], ry, ry + rh)
        return np.array([cx, cy])

    def get_desired_velocity(self, agent, p_target, other_agents, obstacles):
        p_i = agent.pos
        dir_target = p_target - p_i
        dist_to_goal = np.linalg.norm(dir_target)

        # 1. Snap to 0 velocity if within minimal margin
        if dist_to_goal < 0.05:
            return np.zeros(2)

        # 2. Linear damping: slow down as we approach the target
        v_max = 1.5
        arrival_radius = 1.0
        target_speed = v_max * (dist_to_goal / arrival_radius) if dist_to_goal < arrival_radius else v_max
        
        dir_target_norm = dir_target / dist_to_goal if dist_to_goal > 0 else np.zeros(2)
        v_att = dir_target_norm * target_speed

        # 3. HYPER-STIFF CUBIC REPULSION
        current_speed = 0.0
        if len(agent.history) > 1:
            current_speed = np.linalg.norm(agent.history[-1] - agent.history[-2]) / 0.5
        dynamic_d_safe = self.r_base + (self.kappa * current_speed)

        v_rep = np.zeros(2)
        
        for other in other_agents:
            dist_ij = np.linalg.norm(p_i - other.pos)
            if 0.01 < dist_ij < dynamic_d_safe:
                rep_mag = self.k_rep_agent * ((1.0/dist_ij - 1.0/dynamic_d_safe)**3) * (1.0/(dist_ij**2))
                v_rep += rep_mag * ((p_i - other.pos) / dist_ij)

        for obs in obstacles:
            closest_pt = self.get_closest_point_on_rect(p_i, obs)
            dist_wall = np.linalg.norm(p_i - closest_pt)
            if 0.01 < dist_wall < self.d_safe_wall:
                rep_mag = self.k_rep_wall * ((1.0/dist_wall - 1.0/self.d_safe_wall)**3) * (1.0/(dist_wall**2))
                v_rep += rep_mag * ((p_i - closest_pt) / dist_wall)

        # Calculate theoretical progress using UN-DAMPED attraction.
        v_att_undamped = dir_target_norm * v_max
        theoretical_forward_progress = np.dot(v_att_undamped + v_rep, dir_target_norm)

        # ---------------------------------------------------------
        # PURE GEOMETRIC STATE MACHINE
        # ---------------------------------------------------------
        if agent.mode == 1:
            if theoretical_forward_progress < self.deadlock_tol:
                agent.mode = 2
                agent.rand_angle = np.random.uniform(0, 2 * np.pi) 
                
        elif agent.mode == 2:
            # --- CHANGED: BULLETPROOF HYSTERESIS ---
            # Max possible progress is 1.5. We demand > 1.4 to exit Chaos.
            # The agent MUST bounce until it is completely outside the wall's 
            # 0.6m repulsive forcefield. No more ping-ponging!
            if theoretical_forward_progress > 1.4:
                agent.mode = 1

        # ---------------------------------------------------------
        # VELOCITY CALCULATION
        # ---------------------------------------------------------
        v_final = np.zeros(2)

        if agent.mode == 1:
            v_final = v_att + v_rep
        else:
            agent.rand_angle += np.random.uniform(-0.5, 0.5)
            rand_mag = np.random.uniform(2.0, 4.0) 
            v_rand = np.array([np.cos(agent.rand_angle), np.sin(agent.rand_angle)]) * rand_mag
            v_final = v_rand + v_rep
            
        speed = np.linalg.norm(v_final)
        
        # Cap velocity to our damped target_speed to ensure smooth parking
        if speed > target_speed:
            v_final = (v_final / speed) * target_speed

        return v_final

# ==========================================
# 4. SIMULATION EXECUTION & ANIMATION
# ==========================================
class DynamicAgent:
    def __init__(self, id, start, goal, color):
        self.id = id
        self.pos = np.array(start, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.color = color
        self.global_path = None
        self.path_idx = 1 
        self.history = [self.pos.copy()]
        
        self.mode = 1           
        self.rand_angle = 0.0 
        self.mode_history = [self.mode] 

def get_current_target(agent):
    dists = np.linalg.norm(agent.global_path - agent.pos, axis=1)
    closest_idx = np.argmin(dists)
    
    # --- CHANGED: TIGHTER LOOKAHEAD ---
    # Ensures the agent tightly tracks the safe RRT* centerline through the corridor and doesn't try to cut the corner.
    target_idx = min(closest_idx + 3, len(agent.global_path) - 1)
    return agent.global_path[target_idx]

def run_simulation():
    # THE STRESS TEST: 1.5m Corridor
    obstacles = [
        (4.8, 5.75, 0.4, 6.25),   
        (4.8, -2.0, 0.4, 6.25),   
        (-2.0, 11.0, 14.0, 0.5),  
        (-2.0, -1.5, 14.0, 0.5),  
        (-1.5, -1.5, 0.5, 13.0),  
        (11.0, -1.5, 0.5, 13.0)   
    ]
    
    a1 = DynamicAgent(1, [0.0, 5.0], [10.0, 5.0], 'blue')
    a2 = DynamicAgent(2, [10.0, 5.0], [0.0, 5.0], 'red')
    agents = [a1, a2]
    
    print("Computing Initial RRT* Bridge Paths...")

    # Minkowski Sum
    robot_radius = 0.35 # Inflate obstacles based on robot's radius
    
    rrt_obstacles = [
        (x - robot_radius, y - robot_radius, w + (2 * robot_radius), h + (2 * robot_radius)) 
        for (x, y, w, h) in obstacles
    ]

    for a in agents:
        # Pass C-space obstacles to the global planner
        rrt = RRTStarBridge(a.pos, a.goal, rrt_obstacles, [-2, 12])
        raw_path = rrt.plan()
        if raw_path is None:
            raw_path = np.array([a.pos, a.goal])
        a.global_path = smooth_global_path(raw_path, num_points=100)

    print("Running Background Simulation...")
    apf = TwoModeAPF()
    dt = 0.5 
    
    for step in range(800): 
        # Exclude parked agents so they don't block active ones
        active_agents = [a for a in agents if np.linalg.norm(a.pos - a.goal) > 0.05]
        
        if len(active_agents) == 0:
            break
            
        for i, agent in enumerate(agents):
            dist_to_final_goal = np.linalg.norm(agent.pos - agent.goal)
            
            # Skip parked agents entirely to lock them in place
            if dist_to_final_goal <= 0.05:
                agent.history.append(agent.pos.copy())
                agent.mode_history.append(1)
                continue 
                
            other_agents = [a for a in active_agents if a.id != agent.id]
            target_wp = get_current_target(agent)
            
            v_des = apf.get_desired_velocity(agent, target_wp, other_agents, obstacles)
            
            # By-pass B-spline for perfect 0.0 margin zero-velocity stops
            if np.linalg.norm(v_des) == 0:
                agent.history.append(agent.pos.copy())
                agent.mode_history.append(1)
                continue
                
            short_term_goal = agent.pos + (v_des * dt * 2.0) 
            planner = TubeBSplineShort(agent.pos, short_term_goal, obstacles)
            safe_traj = planner.plan()
            
            agent.pos = safe_traj[1] 
            agent.history.append(agent.pos.copy())
            agent.mode_history.append(agent.mode)

    print("Rendering Animation...")
    fig, ax = plt.subplots(figsize=(8, 8)) 
    
    for (ox, oy, w, h) in obstacles[:2]:
        ax.add_patch(patches.Rectangle((ox, oy), w, h, color='black', alpha=0.8))
        
    for a in agents:
        ax.plot(a.global_path[:,0], a.global_path[:,1], color=a.color, linestyle=':', alpha=0.3)
        ax.scatter(a.history[0][0], a.history[0][1], marker='o', s=100, color=a.color, edgecolors='black', zorder=4)
        
        hist = np.array(a.history)
        ax.scatter(hist[-1,0], hist[-1,1], marker='*', s=300, color='yellow', edgecolors=a.color, linewidths=2, zorder=5)

    ax.set_title("Stochastic Symmetry Breaking: 2-Mode Hybrid APF", fontsize=14)
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

        m1_str = "Chaos" if agents[0].mode_history[frame] == 2 else "Stable"
        m2_str = "Chaos" if agents[1].mode_history[frame] == 2 else "Stable"
        mode_text.set_text(f"Step: {frame} | Agent 1: {m1_str} | Agent 2: {m2_str}")
        return lines + heads + [mode_text]

    max_frames = max(len(a.history) for a in agents)
    ani = FuncAnimation(fig, update, frames=max_frames, blit=False, interval=100, repeat=False)
    
    plt.show()

if __name__ == "__main__":
    run_simulation()
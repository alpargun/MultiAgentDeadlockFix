import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
from scipy.integrate import solve_ivp

# Custom imports
from environments import get_scenario
from rrt_bridge import RRTStarBridge

np.random.seed(42)

# ======================================================================================================================
# CONFIGURATION
# ======================================================================================================================
ROBOT_RADIUS = 0.35
GOAL_TOLERANCE = 0.05 # Parking brake trigger radius
V_MAX = 1.5 # Maximum speed
T_MAX = 90.0 # Maximum simulation time
GAP = 1.6 # Minimum gap between walls in the maze and corridor scenarios

# RRT* parameters
RRT_MAX_ITER = 2500
RRT_EXPAND_DIS = 0.5 # Branch segment length

# Continuous Dynamics (ODE & APF)
LOOKAHEAD_DIST = 0.8 # How far ahead on the global path the ODE's nominal point (P2) is placed
APF_BASE_BUFFER = 0.05 # Minimum buffer around agents and walls at zero speed
ODE_MAX_STEP = 0.1 # Maximum step size for the ODE solver (smaller = more accurate but slower)
MIN_CLEAR_DIST = 0.01 # Floor so repulsion stays finite on contact

DIR_OUTPUT = "output"

# ======================================================================================================================
# Global Path Tracking Function
# ======================================================================================================================
def get_lookahead_target(pos, path, lookahead=1.0):
    closest_dist = float('inf')
    best_i, best_t = 0, 0.0

    for i in range(len(path) - 1):
        seg_vec = path[i+1] - path[i]
        seg_len = np.linalg.norm(seg_vec)
        if seg_len == 0: continue
        seg_dir = seg_vec / seg_len

        t_clamped = np.clip(np.dot(pos - path[i], seg_dir), 0, seg_len)
        dist_to_seg = np.linalg.norm(pos - (path[i] + t_clamped * seg_dir))

        if dist_to_seg < closest_dist:
            closest_dist = dist_to_seg
            best_i, best_t = i, t_clamped

    # Walk the remaining arc length forward, spilling over into later segments
    remaining = lookahead
    i, t = best_i, best_t
    while i < len(path) - 1:
        seg_vec = path[i+1] - path[i]
        seg_len = np.linalg.norm(seg_vec)
        if seg_len == 0:
            i += 1
            continue
        if t + remaining <= seg_len:
            return path[i] + (t + remaining) * (seg_vec / seg_len)
        remaining -= (seg_len - t)
        i += 1
        t = 0.0

    return path[-1]

def get_closest_point_on_rect(pos, rect):
    rx, ry, rw, rh = rect
    cx = np.clip(pos[0], rx, rx + rw)
    cy = np.clip(pos[1], ry, ry + rh)
    return np.array([cx, cy])

# ======================================================================================================================
# CONTINUOUS-TIME DYNAMICS
# ======================================================================================================================
def continuous_multi_agent_dynamics(t, state_vector, agents, obstacles):
    num_agents = len(agents)
    dstate_dt = np.zeros_like(state_vector)
    
    positions = state_vector[0::3]
    positions_y = state_vector[1::3]
    weights = state_vector[2::3]
    
    for i, agent in enumerate(agents):
        p_i = np.array([positions[i], positions_y[i]])
        W_i = weights[i] 
        
        dist_to_final_goal = np.linalg.norm(agent.goal - p_i)
        
        if dist_to_final_goal <= GOAL_TOLERANCE:
            dstate_dt[i*3 : i*3 + 3] = [0.0, 0.0, -2.0 * W_i] 
            continue
          
        arrival_radius = 0.2 
        if dist_to_final_goal < arrival_radius:
            target_speed = max(0.1, V_MAX * (dist_to_final_goal / arrival_radius))
        else:
            target_speed = V_MAX
            
        P0 = p_i
        P2 = get_lookahead_target(P0, agent.global_path, lookahead=LOOKAHEAD_DIST)
        
        dir_to_P2 = P2 - P0
        dist_P2 = np.linalg.norm(dir_to_P2)
        dir_norm = dir_to_P2 / dist_P2 if dist_P2 > 0 else np.zeros(2)
        
        v_att = dir_norm * target_speed
        
        # Kinematic APF: Velocity-Scaled Safety Bubbles
        estimated_speed = target_speed * (1.0 - W_i)
        
        # Lowered base buffer to 5cm so they can squeeze tightly at low speeds
        dynamic_buffer = APF_BASE_BUFFER + (0.3 * estimated_speed) 
        
        v_rep_agent = np.zeros(2)
        v_rep_wall = np.zeros(2)
        
        for j in range(num_agents):
            if i == j: continue
            p_j = np.array([positions[j], positions_y[j]])
            dist_center_to_center = np.linalg.norm(p_i - p_j)
            if dist_center_to_center < 1e-9: continue
            
            clear_dist = dist_center_to_center - (2.0 * ROBOT_RADIUS)
            
            if clear_dist < dynamic_buffer:
                # Floor the distance instead of skipping, so the barrier never switches off on contact
                d = max(clear_dist, MIN_CLEAR_DIST)
                rep_mag = 5.0 * ((1.0/d - 1.0/dynamic_buffer)**3) * (1.0/(d**2))
                v_rep_agent += rep_mag * ((p_i - p_j) / dist_center_to_center)
                
        for obs in obstacles:
            closest_pt = get_closest_point_on_rect(p_i, obs)
            dist_center_to_wall = np.linalg.norm(p_i - closest_pt)
            
            clear_dist_wall = dist_center_to_wall - ROBOT_RADIUS
            wall_buffer = APF_BASE_BUFFER + (0.2 * estimated_speed)
            
            if clear_dist_wall < wall_buffer:
                d = max(clear_dist_wall, MIN_CLEAR_DIST)
                if dist_center_to_wall > 1e-9:
                    n_hat = (p_i - closest_pt) / dist_center_to_wall
                else:
                    # Center is inside the rectangle, so push out along the rectangle center
                    n_hat = p_i - np.array([obs[0] + obs[2]/2.0, obs[1] + obs[3]/2.0])
                    n_hat = n_hat / max(np.linalg.norm(n_hat), 1e-9)
                rep_mag = 8.0 * ((1.0/d - 1.0/wall_buffer)**3) * (1.0/(d**2))
                v_rep_wall += rep_mag * n_hat
                
        # ============================================================
        # SAFETY CLAMPS
        # ============================================================
        # Cap the repulsions so a near-contact spike can't blow up the integrator
        mag_agent = np.linalg.norm(v_rep_agent)
        if mag_agent > 15.0:
            v_rep_agent = (v_rep_agent / mag_agent) * 15.0
            
        mag_wall = np.linalg.norm(v_rep_wall)
        if mag_wall > 25.0:
            v_rep_wall = (v_rep_wall / mag_wall) * 25.0
        
        # ============================================================
        # ESCAPE PERTURBATION
        # ============================================================
        # Push direction is held for one trial period, then redrawn if still stuck
        k = min(int((t + agent.trial_offset) // agent.trial_period), len(agent.trial_angles) - 1)
        theta = agent.trial_angles[k]
        
        # Stretch of the control point, gated by W_i so it collapses to v_att when free
        lambda_t = 0.5 + 0.3 * W_i * np.sin(agent.random_stretch_freq * t + agent.random_stretch_phase)
        
        # Quadratic Bezier tangent at P0, shifted by the push while W_i > 0
        P1_nominal = P0 + (dir_norm * (target_speed * lambda_t)) 
        
        pull_vector = np.array([np.cos(theta), np.sin(theta)]) * 2.0 * W_i
        P1_shifted = P1_nominal + pull_vector
        
        v_spline_tangent = 2.0 * (P1_shifted - P0)
        v_final = v_spline_tangent + v_rep_agent + v_rep_wall
        
        speed = np.linalg.norm(v_final)
        if speed > target_speed:
            v_final = (v_final / speed) * target_speed

        # ============================================================
        # DEADLOCK WEIGHT
        # ============================================================
        # Progress of the nominal law toward the lookahead point; excludes the
        # perturbation so W can't feed itself
        v_nominal = v_att + v_rep_agent + v_rep_wall
        speed_nom = np.linalg.norm(v_nominal)
        if speed_nom > target_speed:
            v_nominal = (v_nominal / speed_nom) * target_speed
        actual_forward_progress = np.dot(v_nominal, dir_norm)

        if dist_to_final_goal < 0.5:
            W_target = 0.0
            dW_dt = 10.0 * (W_target - W_i)
        else:
            exponent = np.clip(15.0 * (actual_forward_progress - 0.2), -500, 500)
            W_target = 1.0 / (1.0 + np.exp(exponent))

            if W_target > W_i:
                dW_dt = 10.0 * (W_target - W_i)
            else:
                dW_dt = 3.0 * (W_target - W_i)

        dstate_dt[i*3 : i*3 + 3] = [v_final[0], v_final[1], dW_dt]
        
    return dstate_dt

# ======================================================================================================================
# SIMULATION SETUP & EXECUTION
# ======================================================================================================================
def run_continuous_simulation(scenario):
    
    agents, obstacles = get_scenario(scenario, gap=GAP)
    
    print(f"Computing Exact RRT* Paths for '{scenario.upper()}'...")
    rrt_obstacles = [(x - ROBOT_RADIUS, y - ROBOT_RADIUS, w + (2 * ROBOT_RADIUS), h + (2 * ROBOT_RADIUS)) for (x, y, w, h) in obstacles]

    for a in agents:
        rrt = RRTStarBridge(a.pos, a.goal, rrt_obstacles, [-2, 12], expand_dis=RRT_EXPAND_DIS, max_iter=RRT_MAX_ITER)
        raw_path = rrt.plan()
        if raw_path is None: raw_path = np.array([a.pos, a.goal])
        a.global_path = raw_path

    initial_state = []
    for a in agents:
        initial_state.extend([a.pos[0], a.pos[1], 0.0]) 
        
    def termination_event(t, y):
        distances = []
        for i, a in enumerate(agents):
            pos = np.array([y[i*3], y[i*3+1]])
            distances.append(np.linalg.norm(pos - a.goal))
        return max(distances) - GOAL_TOLERANCE
    termination_event.terminal = True
    termination_event.direction = -1

    print("Running True Continuous-Time ODE Solver (solve_ivp)...")

    sol = solve_ivp(
        fun=lambda t, y: continuous_multi_agent_dynamics(t, y, agents, obstacles),
        t_span=(0, T_MAX),
        y0=initial_state,
        events=termination_event,
        dense_output=True, 
        max_step=ODE_MAX_STEP
    )
    
    print(f"ODE Solver finished at t = {sol.t[-1]:.2f} seconds.")

    # ==================================================================================================================
    # FIGURE 1: RENDER ANIMATION
    # ==================================================================================================================
    print("Preparing Visuals...")
    t_frames = np.arange(0, sol.t[-1], 0.05)
    y_frames = sol.sol(t_frames)
    
    fig, ax = plt.subplots(figsize=(8, 8)) 
    for (ox, oy, w, h) in obstacles:
        ax.add_patch(patches.Rectangle((ox, oy), w, h, color='black', alpha=0.8))
        
    for a in agents:
        ax.plot(a.global_path[:,0], a.global_path[:,1], color=a.color, linestyle=':', alpha=0.5, lw=2)
        ax.scatter(a.goal[0], a.goal[1], marker='*', s=300, color='yellow', edgecolors=a.color, zorder=5)

    ax.set_title(f"Continuous ODE - {scenario.capitalize()}", fontsize=14)
    ax.set_xlim(-1.0, 11.0)
    ax.set_ylim(-1.0, 11.0)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)

    mode_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    lines, physical_bodies, apf_bubbles = [], [], []
    for a in agents:
        body = patches.Circle((a.pos[0], a.pos[1]), radius=ROBOT_RADIUS, color=a.color, alpha=0.9, zorder=5)
        ax.add_patch(body)
        physical_bodies.append(body)
        
        bubble = patches.Circle((a.pos[0], a.pos[1]), radius=ROBOT_RADIUS + 0.55, color=a.color, alpha=0.15, zorder=4)
        ax.add_patch(bubble)
        apf_bubbles.append(bubble)
        
        line, = ax.plot([], [], color=a.color, lw=2.5, zorder=3)
        lines.append(line)

    def update(frame_idx):
        for i, a in enumerate(agents):
            hist_x = y_frames[i*3][:frame_idx+1]
            hist_y = y_frames[i*3+1][:frame_idx+1]
            lines[i].set_data(hist_x, hist_y)
            
            physical_bodies[i].center = (hist_x[-1], hist_y[-1])
            apf_bubbles[i].center = (hist_x[-1], hist_y[-1])
            
            # Extract the exact Deadlock Weight (W_i) from the ODE state vector
            W_i = y_frames[i*3+2][frame_idx]
            
            dist_to_goal = np.linalg.norm(np.array([hist_x[-1], hist_y[-1]]) - a.goal)
            target_speed = max(0.1, V_MAX * (dist_to_goal / 0.2)) if dist_to_goal < 0.2 else V_MAX
            
            estimated_speed = target_speed * (1.0 - W_i)
            dynamic_buffer = APF_BASE_BUFFER + (0.3 * estimated_speed)
            apf_bubbles[i].set_radius(ROBOT_RADIUS + dynamic_buffer)

        distances = []
        for i, a in enumerate(agents):
            current_pos = np.array([y_frames[i*3][frame_idx], y_frames[i*3+1][frame_idx]])
            dist = np.linalg.norm(current_pos - a.goal)
            distances.append(dist)
            
        strs = [f"A{i+1}: {d:.2f}m" for i, d in enumerate(distances)]
        mode_text.set_text(f"Time: {t_frames[frame_idx]:.2f}s | " + " | ".join(strs))
        return lines + physical_bodies + apf_bubbles + [mode_text]

    ani = FuncAnimation(fig, update, frames=len(t_frames), blit=False, interval=50, repeat=True)
    os.makedirs(DIR_OUTPUT, exist_ok=True)
    print("Saving animation...")
    ani.save(f"{DIR_OUTPUT}/{scenario}.mp4", writer="ffmpeg", fps=20, dpi=150)

    # ==================================================================================================================
    # CLEARANCE DATA (Safety Proof Module)
    # ==================================================================================================================
    print("Calculating Minimum Clearance Data...")
    min_agent_clearance_history = []
    min_wall_clearance_history = []
    
    # Sweep every single frame of the simulation to guarantee no physics clipping occurred
    for frame_idx in range(len(t_frames)):
        current_agent_clearance = float('inf')
        current_wall_clearance = float('inf')
        
        # 1. Check all Agent-to-Agent distances
        for i in range(len(agents)):
            p_i = np.array([y_frames[i*3][frame_idx], y_frames[i*3+1][frame_idx]])
            for j in range(i + 1, len(agents)):
                p_j = np.array([y_frames[j*3][frame_idx], y_frames[j*3+1][frame_idx]])
                
                # Clear distance is edge-to-edge
                clear_dist = np.linalg.norm(p_i - p_j) - (2.0 * ROBOT_RADIUS)
                if clear_dist < current_agent_clearance:
                    current_agent_clearance = clear_dist
                    
            # 2. Check all Agent-to-Wall distances
            for obs in obstacles:
                closest_pt = get_closest_point_on_rect(p_i, obs)
                clear_dist_wall = np.linalg.norm(p_i - closest_pt) - ROBOT_RADIUS
                if clear_dist_wall < current_wall_clearance:
                    current_wall_clearance = clear_dist_wall
                    
        min_agent_clearance_history.append(current_agent_clearance)
        min_wall_clearance_history.append(current_wall_clearance)

    # ==================================================================================================================
    # FIGURE 2: TIME SERIES (convergence, clearance, deadlock weight on a shared time axis)
    # ==================================================================================================================
    fig_ts, (ax_error, ax_clearance, ax_w) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    # Panel 1: distance to goal
    for i, a in enumerate(agents):
        distances = [np.linalg.norm(np.array([x, y]) - a.goal) for x, y in zip(y_frames[i*3], y_frames[i*3+1])]
        ax_error.plot(t_frames, distances, label=f'Agent {a.id}', color=a.color, lw=2)
    ax_error.set_title(f"Continuous Distance to Goal ({scenario.capitalize()})", fontsize=13, fontweight='bold')
    ax_error.set_ylabel("Distance to Goal (m)", fontsize=11)
    ax_error.grid(True, linestyle='--', alpha=0.6)
    ax_error.legend()

    # Panel 2: minimum clearance
    ax_clearance.plot(t_frames, min_agent_clearance_history, label='Min Agent-to-Agent', color='purple', lw=2)
    ax_clearance.plot(t_frames, min_wall_clearance_history, label='Min Agent-to-Wall', color='orange', lw=2)
    ax_clearance.set_title("Safety Verification: Minimum Clearance", fontsize=13, fontweight='bold')
    ax_clearance.set_ylabel("Clearance (m)", fontsize=11)
    ax_clearance.set_yscale('log')
    ax_clearance.axhline(APF_BASE_BUFFER, color='gray', linestyle='--', alpha=0.7, label=f'Base Buffer ({APF_BASE_BUFFER}m)')
    ax_clearance.set_ylim(0.02, 20.0)
    ax_clearance.grid(True, linestyle='--', alpha=0.6)
    ax_clearance.legend(loc='upper right', fontsize=9)

    # Panel 3: deadlock weight
    for i, a in enumerate(agents):
        ax_w.plot(t_frames, y_frames[i*3+2], label=f'Agent {a.id}', color=a.color, lw=2)
    ax_w.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='W = 0.5')
    ax_w.set_title("Deadlock Weight W(t)", fontsize=13, fontweight='bold')
    ax_w.set_xlabel("Time (seconds)", fontsize=11)
    ax_w.set_ylabel("Deadlock Weight", fontsize=11)
    ax_w.set_ylim(-0.05, 1.05)
    ax_w.grid(True, linestyle='--', alpha=0.6)
    ax_w.legend(fontsize=9)
    fig_ts.savefig(f"{DIR_OUTPUT}/timeseries_{scenario}.png", dpi=200)

    print("Displaying animation and time series...")
    plt.show()
    
    print(f"[{scenario.upper()}] Execution complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Continuous APF Navigation")
    parser.add_argument('--map', type=str, choices=['corridor', 'intersection', 'maze', 'warehouse'], 
                        default='intersection', help="Select the map scenario to run.")
    args = parser.parse_args()
    
    run_continuous_simulation(args.map)
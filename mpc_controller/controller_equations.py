import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

# =========================
# Model parameters
# ========================

# Define the variables already known from the model
m1 = m2 = 2.5       # link masses (Kg)
L1 = L2 = 0.4       # link lengths (m)
H1 = H2 = 0.05      # link widths (m)
g = 9.81            # acceleration due to gravity (m/s**2)
I1 = m1*(L1**2 + H1**2)/12      # moment of inertia of link 1 (Kg*m**2)
I2 = m2*(L2**2 + H2**2)/12      # moment of inertia of link 2 (Kg*m**2)

# Variables for defining a reference trajectory
w = 2 * np.pi/10     # angular frequency of the circular trajectory (rad/s)
r = 0.5             # radius of the circular trajectory (m)

# MPC parameters
H = 50              # prediction horizon
dt = 0.001          # time increment step (s)
N = 10000            # number of time steps in the prediction horizon

# Control constraints
tau_max = 50.0      # maximum torque (N·m)
tau_min = -50.0     # minimum torque (N·m)

# =========================
# Dynamics Functions (as provided)
# =========================

def calculate_mass_matrix(q):
    q1, q2 = q[0], q[1]
    M11 = I1 + I2 + (m1*L1**2)/4 + m2*L1**2 + (m2*L2**2)/4 + m2*L1*L2*ca.cos(q2)
    M12 = I2 + (m2*L2**2)/4 + (m2*L1*L2*ca.cos(q2))/2
    M21 = M12
    M22 = I2 + (m2*L2**2)/4 

    return ca.vertcat(
        ca.horzcat(M11, M12),
        ca.horzcat(M21, M22)
    )

def coriolis_vector(q, dq):
    q1, q2 = q[0], q[1]
    q1d, q2d = dq[0], dq[1]
    C1 = -(m2*L1*L2*q2d*ca.sin(q2) * (2*q1d + q2d))/2
    C2 = (m2*L1*L2*ca.sin(q2)*q1d**2)/2

    return ca.vertcat(C1, C2)

def gravity_vector(q):
    q1, q2 = q[0], q[1]
    G1 = (m2*((L2*ca.cos(q1 + q2))/2 + L1*ca.cos(q1)) + 0.5*m1*L1*ca.cos(q1))*g
    G2 = 0.5*m2*g*L2*ca.cos(q1 + q2)

    return ca.vertcat(G1, G2)

def dynamics(q, dq, tau):
    M = calculate_mass_matrix(q)
    C = coriolis_vector(q, dq)
    G = gravity_vector(q)
    ddq = ca.solve(M, tau - C - G)
    return ddq

# =========================
# Kinematics Functions
# =========================

def forward_kinematics(q):
    """Compute end-effector position from joint angles"""
    q1, q2 = q[0], q[1]
    x = L1 * ca.cos(q1) + L2 * ca.cos(q1 + q2)
    y = L1 * ca.sin(q1) + L2 * ca.sin(q1 + q2)
    return ca.vertcat(x, y)

def inverse_kinematics_numeric(x, y):
    """Compute joint angles from end-effector position (numeric version)"""
    # Using numpy functions for numeric computation
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    # Ensure c2 is within valid range [-1, 1]
    c2 = np.clip(c2, -1, 1)
    s2 = np.sqrt(1 - c2**2)
    q2 = np.arctan2(s2, c2)
    
    k1 = L1 + L2 * c2
    k2 = L2 * s2
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return q1, q2

def inverse_kinematics_symbolic(x, y):
    """Compute joint angles from end-effector position (symbolic version)"""
    # Using CasADi functions for symbolic computation
    c2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    # Ensure c2 is within valid range [-1, 1]
    c2 = ca.fmax(ca.fmin(c2, 1), -1)
    s2 = ca.sqrt(1 - c2**2)
    q2 = ca.atan2(s2, c2)
    
    k1 = L1 + L2 * c2
    k2 = L2 * s2
    q1 = ca.atan2(y, x) - ca.atan2(k2, k1)
    
    return ca.vertcat(q1, q2)

def generate_reference_trajectory(q1_initial, q2_initial, N=N):
    """Generate reference end-effector trajectory"""
    t = np.arange(0, N*dt, dt)
    x_ref = r * np.cos(q1_initial + q2_initial + w * t)
    y_ref = r * np.sin(q1_initial + q2_initial + w * t)
    return x_ref, y_ref

# =========================
# MPC Controller
# =========================

def setup_mpc_controller():
    """Set up the MPC optimization problem"""
    # Define optimization variables
    opti = ca.Opti()
    
    # State variables: [q1, q2, dq1, dq2]
    X = opti.variable(4, H+1)  # state trajectory
    
    # Control variables: [tau1, tau2]
    U = opti.variable(2, H)    # control trajectory
    
    # Parameters: initial state and reference trajectory
    X0 = opti.parameter(4, 1)  # initial state
    X_ref = opti.parameter(4, H+1)  # reference state trajectory
    
    # Weight matrices
    Q = np.diag([200, 200, 10, 10])    # state tracking weights
    R = np.diag([1e-2, 1e-2])        # control effort weights
    
    # Cost function
    cost = 0
    for k in range(H+1):
        # State tracking cost
        state_error = X[:, k] - X_ref[:, k]
        cost += ca.mtimes([state_error.T, Q, state_error])
        
        if k < H:
            # Control effort cost
            cost += ca.mtimes([U[:, k].T, R, U[:, k]])
    
    # Dynamics constraints
    for k in range(H):
        # Current state
        q = X[0:2, k]
        dq = X[2:4, k]
        tau = U[:, k]
        
        # Dynamics
        ddq = dynamics(q, dq, tau)
        
        # Next state using Euler integration
        next_q = q + dq * dt
        next_dq = dq + ddq * dt
        next_state = ca.vertcat(next_q, next_dq)
        
        # Constraint: next state must equal predicted state
        opti.subject_to(X[:, k+1] == next_state)
    
    # Initial condition constraint
    opti.subject_to(X[:, 0] == X0)
    
    # Control constraints
    for k in range(H):
        opti.subject_to(opti.bounded(tau_min, U[:, k], tau_max))
    
    # Setup solver
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 100}
    opti.solver('ipopt', opts)
    
    return opti, X, U, X0, X_ref

def run_mpc_simulation():
    """Run the MPC simulation"""
    # Initial joint angles and velocities
    q1_0, q2_0 = 30*np.pi/180, 10*np.pi/180  # initial joint angles
    dq1_0, dq2_0 = 0, 0            # initial joint velocities
    
    # Generate reference trajectory
    x_ref, y_ref = generate_reference_trajectory(q1_0, q2_0, N=N)
    
    # Convert to joint space reference using inverse kinematics (numeric version)
    q1_ref = np.zeros(N)
    q2_ref = np.zeros(N)
    for i in range(N):
        q1_ref[i], q2_ref[i] = inverse_kinematics_numeric(x_ref[i], y_ref[i])
    
    # Setup MPC controller
    opti, X, U, X0, X_ref = setup_mpc_controller()
    
    # Initialize states
    current_state = np.array([q1_0, q2_0, dq1_0, dq2_0])
    
    # Storage for results
    states_history = np.zeros((4, N))
    controls_history = np.zeros((2, N))
    ee_history = np.zeros((2, N))
    
    # Main simulation loop
    for i in range(N):
        # Set parameters
        opti.set_value(X0, current_state)
        
        # Set reference trajectory (current horizon)
        horizon_end = min(i + H + 1, N)
        horizon_length = horizon_end - i
        
        # Create reference for the horizon
        ref_traj = np.zeros((4, H+1))
        ref_traj[0, :horizon_length] = q1_ref[i:horizon_end]
        ref_traj[1, :horizon_length] = q2_ref[i:horizon_end]
        # Repeat the last reference point if needed
        if horizon_length < H+1:
            ref_traj[0, horizon_length:] = q1_ref[-1]
            ref_traj[1, horizon_length:] = q2_ref[-1]
        
        opti.set_value(X_ref, ref_traj)
        
        # Solve the optimization problem
        try:
            sol = opti.solve()
            x_opt = sol.value(X)
            u_opt = sol.value(U)
            
            # Apply first control input
            control_input = u_opt[:, 0]
            
            # Store results
            states_history[:, i] = current_state
            controls_history[:, i] = control_input
            
            # Calculate end-effector position
            ee_pos = forward_kinematics(current_state[0:2])
            ee_history[0, i] = ee_pos[0].full()[0, 0]  # Extract numeric value from CasADi object
            ee_history[1, i] = ee_pos[1].full()[0, 0]  # Extract numeric value from CasADi object
            
            # Simulate system forward with the control input
            q = current_state[0:2]
            dq = current_state[2:4]
            
            # Convert to CasADi objects for dynamics calculation
            q_ca = ca.DM(q)
            dq_ca = ca.DM(dq)
            tau_ca = ca.DM(control_input)
            
            # Calculate acceleration
            ddq = dynamics(q_ca, dq_ca, tau_ca)
            
            # Euler integration
            next_dq = dq + np.array(ddq).flatten() * dt
            next_q = q + next_dq * dt
            current_state = np.concatenate([next_q, next_dq])
            
        except Exception as e:
            print(f"Solver failed at step {i}: {e}")
            # If solver fails, use the previous control input
            if i > 0:
                control_input = controls_history[:, i-1]
            else:
                control_input = np.zeros(2)
            
            controls_history[:, i] = control_input
            states_history[:, i] = current_state
            
            # Calculate end-effector position
            ee_pos = forward_kinematics(current_state[0:2])
            ee_history[0, i] = ee_pos[0].full()[0, 0] if hasattr(ee_pos[0], 'full') else ee_pos[0]
            ee_history[1, i] = ee_pos[1].full()[0, 0] if hasattr(ee_pos[1], 'full') else ee_pos[1]
            
            # Continue with zero control input
            q = current_state[0:2]
            dq = current_state[2:4]
            next_q = q + dq * dt
            next_dq = dq  # Assume no acceleration if solver fails
            current_state = np.concatenate([next_q, next_dq])
    
    return states_history, controls_history, ee_history, x_ref, y_ref

# =========================
# Visualization
# =========================

def plot_results(states_history, controls_history, ee_history, x_ref, y_ref):
    """Plot the simulation results"""
    t = np.arange(0, N*dt, dt)
    
    # Plot joint angles
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 2, 1)
    plt.plot(t, states_history[0, :], label='q1')
    plt.plot(t, states_history[1, :], label='q2')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Angles (rad)')
    plt.legend()
    plt.title('Joint Angles')
    plt.grid(True)
    
    # Plot joint velocities
    plt.subplot(3, 2, 2)
    plt.plot(t, states_history[2, :], label='dq1')
    plt.plot(t, states_history[3, :], label='dq2')
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Velocities (rad/s)')
    plt.legend()
    plt.title('Joint Velocities')
    plt.grid(True)
    
    # Plot control torques
    plt.subplot(3, 2, 3)
    plt.plot(t, controls_history[0, :], label='tau1')
    plt.plot(t, controls_history[1, :], label='tau2')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (N·m)')
    plt.legend()
    plt.title('Control Torques')
    plt.grid(True)
    
    # Plot end-effector trajectory
    plt.subplot(3, 2, 4)
    plt.plot(ee_history[0, :], ee_history[1, :], label='Actual')
    plt.plot(x_ref, y_ref, 'r--', label='Reference')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.legend()
    plt.title('End-Effector Trajectory')
    plt.axis('equal')
    plt.grid(True)
    
    # Plot tracking error
    tracking_error = np.sqrt((ee_history[0, :] - x_ref)**2 + (ee_history[1, :] - y_ref)**2)
    plt.subplot(3, 2, 5)
    plt.plot(t, tracking_error)
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Tracking Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# =========================
# Main execution
# =========================

if __name__ == "__main__":
    # Run the MPC simulation
    states_history, controls_history, ee_history, x_ref, y_ref = run_mpc_simulation()
    
    # Plot the results
    plot_results(states_history, controls_history, ee_history, x_ref, y_ref)
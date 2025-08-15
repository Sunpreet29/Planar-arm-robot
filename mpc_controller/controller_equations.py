# MPC controller implementation
import numpy as np
import matplotlib.pyplot as plt

# Define the variables already known from the model
m1 = m2 = 2.5       # link masses (Kg)
L1 = L2 = 0.4       # link lengths (m)
H1 = H2 = 0.05      # link widthsb (m)
g = 9.81            # acceleration due to gravity (m/s**2)

# Defining inertia values
I1 = m1*(L1**2 + H1**2)/12
I2 = m2*(L2**2 + H2**2)/12

# Mass matrix function
def calculate_mass_matrix(q):
    q1, q2 = q
    M11 = I1 + I2 + (m1*L1**2)/4 + m2*L1**2 + (m2*L2**2)/4 + m2*L1*L2*np.cos(q2)
    M12 = I2 + (m2*L2**2)/4 + (m2*L1*L2*np.cos(q2))/2
    M21 = M12
    M22 = I2 + (m2*L2**2)/4 

    return np.array([[M11, M12],
                     [M21, M22]])


# Coriolis vector
def coriolis_vector(q, dq):
    q1, q2 = q
    q1d, q2d = dq
    C1 = -(m2*L1*L2*q2d*np.sin(q2) * (2*q1d + q2d))/2
    C2 = (m2*L1*L2*np.sin(q2)*q1d**2)/2

    return np.array([C1, C2])


# Gravity vector
def gravity_vector(q):
    q1, q2 = q
    G1 = (m2*((L2*np.cos(q1 + q2))/2 + L1*np.cos(q1)) + 0.5*m1*L1*np.cos(q1))*g
    G2 = 0.5*m2*g*L2*np.cos(q1 + q2)

    return np.array([G1, G2])


# Model dynamics. Gives the acceleration at the next time step.
def dynamics(q, dq, tau):
    M = calculate_mass_matrix(q)
    C = coriolis_vector(q, dq)
    G = gravity_vector(q)
    ddq = np.linalg.solve(M, tau - C -G)

    return ddq

# Gives the next step angular_displacement and angular_velocity vector
def next_step_kinematics(q, dq, ddq):
    """
    q: angular_displacement vector at time step t
    dq: angular_velocity vector at time step t
    ddq: angular_acceleration vector at time step t+1
    
    return q_next, dq_next (angular_displacement and angular_velocity vector at next time step respectively)
    """
    dq_next = dq + dt*ddq
    q_next = q + dt*dq_next

    return q_next, dq_next

# Variables for defining a reference trajectory
w = 2 * np.pi       # angular frequency of the circular trajectory (rad/s)
r = 0.5             # radius of the circular trajectory (m)
dt = 0.001          # time increment step (s)

# Generates a reference trajectory for the MPC controller
def generate_reference_trajectory(q1_initial, q2_initial, N=900):
    """
    Generates a reference trajectory for the MPC controller.
    
    q1_initial: initial angle of the first link (rad)
    q2_initial: initial angle of the second link (rad)
    dt: time increment step (s)
    N: number of time steps in the prediction horizon
    T: total simulation time (s)
    
    Returns:
        x_ref: x-coordinate of the reference trajectory
        y_ref: y-coordinate of the reference trajectory
        dx_ref: x-velocity of the reference trajectory
        dy_ref: y-velocity of the reference trajectory
    """
    t = np.arange(0, N*dt, dt)
    x_ref = r * np.cos(q1_initial + q2_initial + w * t)          # x-coordinate of the reference trajectory
    y_ref = r * np.sin(q1_initial + q2_initial + w * t)          # y-coordinate of the reference trajectory
    
    return x_ref, y_ref

# Calculate angles using inverse kinematics equations
def inverse_kinematics(x, y):
    """
    Computes the inverse kinematics for the two-link manipulator.
    
    x: x-coordinate of the end-effector
    y: y-coordinate of the end-effector
    
    Returns:
        q1: angle of the first link with respect to the horizontal axis
        q2: angle of the second link with respect to the first link
    """
    
    q2 = np.arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    
    return q1, q2

if __name__ == "__main__":
    x_ref, y_ref = generate_reference_trajectory(30*np.pi/180, 10*np.pi/180)
    q1_trajectory, q2_trajectory = [], []
    for x, y in zip(x_ref, y_ref):
        q1, q2 = inverse_kinematics(x, y)
        print(f"q1: {q1}, q2: {q2}")
        # q1_trajectory.append(q1)
        # q2_trajectory.append(q2)

    # q1_trajectory = np.array(q1_trajectory)
    # q2_trajectory = np.array(q2_trajectory)
    # q_ref = np.vstack((q1_trajectory, q2_trajectory)).T

    plt.figure(figsize=(6, 6))
    plt.scatter(x_ref, y_ref, label='Reference Trajectory')
    plt.plot(0, 0, 'ro', label='Start Point')  # Start point at origin
    plt.xlim(-0.6, 0.6)
    plt.ylim(-0.6, 0.6)
    plt.show()
# MPC controller implementation
import numpy as np

# Define the variables already known from the model
m1 = m2 = 2.5       # link masses (Kg)
L1 = L2 = 0.4       # link lengths (m)
H1 = H2 = 0.05      # link widthsb (m)
g = 9.81            # acceleration due to gravity (m/s**2)

# Defining inertia values
I1 = m1*(L1** + H1**2)/12
I2 = m2*(L2** + H2**2)/12

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

# Define variables necessary for the MPC
dt = 0.1        # time increment step (s)
N = 100          # number of time steps in the prediction horizon

# Variables for defining a reference trajectory
w = 2 * np.pi       # angular frequency of the circular trajectory (rad/s)
r = 0.5             # radius of the circular trajectory (m)

# Generates a reference trajectory for the MPC controller
def generate_reference_trajectory(dt):
    """
    Generates a reference trajectory for the MPC controller.
    
    dt: time increment step (s)
    
    Returns:
        x_ref: x-coordinate of the reference trajectory
        y_ref: y-coordinate of the reference trajectory
        dx_ref: x-velocity of the reference trajectory
        dy_ref: y-velocity of the reference trajectory
    """
    t = np.arange(0, N*dt, dt)
    x_ref = r * np.cos(w * t)          # x-coordinate of the reference trajectory
    y_ref = r * np.sin(w * t)          # y-coordinate of the reference trajectory
    dx_ref = -r * w * np.sin(w * t)    # x-velocity of the reference trajectory
    dy_ref = r * w * np.cos(w * t)     # y-velocity of the reference trajectory
    
    return x_ref, y_ref, dx_ref, dy_ref

def inverse_kinematics(x, y):
    """
    Computes the inverse kinematics for the two-link manipulator.
    
    x: x-coordinate of the end-effector
    y: y-coordinate of the end-effector
    
    Returns:
        q1: angle of the first link
        q2: angle of the second link
    """
    
    # Calculate angles using inverse kinematics equations
    q2 = np.arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    
    return q1, q2

if __name__ == "__main__":
    angular_displacement = np.array([np.pi/4, 0.75*np.pi])
    angular_velocity = np.array([1, 1])
    torque = np.array([0.5, 1])
    
    angular_acceleration = dynamics(angular_displacement, angular_velocity, torque)
    a1, a2 = next_step_kinematics(angular_displacement, angular_velocity, angular_acceleration)
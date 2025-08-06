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
    qdd = np.linalg.solve(M, tau - C -G)

    return qdd

# Define variables necessary for the MPC
dt = 0.1        # time increment step (s)


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

if __name__ == "__main__":
    angular_displacement = np.array([np.pi/4, 0.75*np.pi])
    angular_velocity = np.array([1, 1])
    torque = np.array([0.5, 1])
    
    angular_acceleration = dynamics(angular_displacement, angular_velocity, torque)
    a1, a2 = next_step_kinematics(angular_displacement, angular_velocity, angular_acceleration)
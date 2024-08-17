"""
    Example of system dynamics discretization for non-homogeneous linear systems.

    - excact integration
    - exaact integration of homogeneous part + euler integration of particular solution
    - euler integration
    - taylor series integration

    mail@kaiploeger.net
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm


# dynamics parameters
SPRING_CONSTANT = 20
SPRING_EQUILIBRIUM_LENGTH = 1
DAMPING = 1
MASS = 1
SPRING_MOUNT = 0

# initial state
MASS_START_POS = 1
MASS_START_VEL = -1

# integration parameters
DURATION = 1
NUM_STEPS = 10
DT = DURATION/NUM_STEPS


def discretize_exact(A, b, dt):
    """
        Solution of linear system is of from:
        x(t) = e^{At}x(0) + int_0^t e^{A(t-s)}b ds   (homogeneous + particular solution)

        The integral over matrix exponential can be found by integrating it's series representation:
        int e^{At} dt = A^{-1} e^{At} + C

        So the final solution is:
        x(t) = e^{At}x(0) + A^{-1} (e^{At} - I) b
    """
    Ad = expm(A*dt)
    bd = np.linalg.inv(A) @ (expm(A*dt) - np.eye(2)) @ b
    return Ad, bd


def discretize_exact_hom_euler_part(A, b, dt):
    """
        Solution of linear system is of from:
        x(t) = e^{At}x(0) + int_0^t e^{A(t-s)}b ds   (homogeneous + particular solution)

        The integral over matrix exponential can be approximated as single euler integration step:
        int e^{At} dt = t *e^{At}

        So the final solution is:
        x(t+dt) = e^{Adt}x(t) + dt * e^{Adt} b
    """
    Ad = expm(A*dt)
    bd = dt * expm(A*dt) @ b
    return Ad, bd


def discretize_euler(A, b, dt):
    """
        Euler discretization of linear system:
        x(t+dt) = (I + A*dt) x(t) + b*dt
    """
    Ad = np.eye(2) + A*dt
    bd = b*dt
    return Ad, bd


def discretize_taylor(A, b, dt, order=2):
    """
        Taylor series discretization of linear system:
        x(t+dt) = (I + A*dt + A^2*dt^2/2 + ... ) x(t)   +   b*dt + A*b*dt^2/2 + ... 
    """
    Ad = np.eye(2)
    bd = np.zeros(2)
    for k in range(1, order+1):
        Ad += (dt**k / np.math.factorial(k)) * np.linalg.matrix_power(A, k)
        bd += (dt**k / np.math.factorial(k)) * np.linalg.matrix_power(A, k-1) @ b
    return Ad, bd


def integrate_discrete(Ad, bd, x0):
    traj = np.zeros((NUM_STEPS+1, x0.shape[0]))
    traj[0] = x0
    for i in range(1, NUM_STEPS+1):
        traj[i] = Ad @ traj[i-1] + bd
    return traj


def main():
    # time-continuous system dynamics
    A = np.array([[ 0,            1           ], 
                [-SPRING_CONSTANT/MASS, -DAMPING/MASS]])

    b = np.array([0,
                (SPRING_MOUNT+SPRING_EQUILIBRIUM_LENGTH)*SPRING_CONSTANT/MASS])


    x0 = np.array([MASS_START_POS,
                   MASS_START_VEL])


    # discretize dynamics
    Ad_exact, bd_exact = discretize_exact(A, b, DT)
    Ad_euler, bd_euler = discretize_euler(A, b, DT)
    # Ad_euler, bd_euler = discretize_taylor(A, b, DT, order=1)
    Ad_taylor2, bd_taylor2 = discretize_taylor(A, b, DT, order=2)
    Ad_taylor3, bd_taylor3 = discretize_taylor(A, b, DT, order=3)
    Ad_taylor4, bd_taylor4 = discretize_taylor(A, b, DT, order=4)
    Ad_exact_hom_euler_part, bd_exact_hom_euler_part = discretize_exact_hom_euler_part(A, b, DT)

    # integrate
    follower_traj_exact = integrate_discrete(Ad_exact, bd_exact, x0)
    follower_traj_euler = integrate_discrete(Ad_euler, bd_euler, x0)
    follower_traj_taylor2 = integrate_discrete(Ad_taylor2, bd_taylor2, x0)
    follower_traj_taylor3 = integrate_discrete(Ad_taylor3, bd_taylor3, x0)
    follower_traj_taylor4 = integrate_discrete(Ad_taylor4, bd_taylor4, x0)
    follower_traj_exact_hom_euler_part = integrate_discrete(Ad_exact_hom_euler_part, bd_exact_hom_euler_part, x0)

    # plot pos and vel over steps
    fig, axs = plt.subplots(2, 1)
    axs[0].axhline(y=SPRING_MOUNT, color='black', linestyle='--', label='spring mount')
    axs[0].axhline(y=SPRING_MOUNT+SPRING_EQUILIBRIUM_LENGTH, color='black', linestyle=':', label='equilibrium pos')
    axs[0].plot(follower_traj_exact[:,0], label='exact')
    axs[0].plot(follower_traj_euler[:,0], label='euler')
    axs[0].plot(follower_traj_taylor2[:,0], label='taylor 2nd')
    axs[0].plot(follower_traj_taylor3[:,0], label='taylor 3rd')
    axs[0].plot(follower_traj_taylor4[:,0], label='taylor 4th')
    axs[0].plot(follower_traj_exact_hom_euler_part[:,0], label='exact hom euler part')
    axs[0].set_ylabel('mass position')
    axs[0].legend()
    axs[1].plot(follower_traj_exact[:,1])
    axs[1].plot(follower_traj_euler[:,1])
    axs[1].plot(follower_traj_taylor2[:,1])
    axs[1].plot(follower_traj_taylor3[:,1])
    axs[1].plot(follower_traj_taylor4[:,1])
    axs[1].set_ylabel('mass velocity')
    plt.show()


if __name__ == '__main__':
    main()

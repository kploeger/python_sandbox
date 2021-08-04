"""
    Simple direct trajectory optimization using CasADi.

    dynamical system: simple integrator
    controls:         acceleration
    cost:             squared acceleration

    -> approximates cubic splines

    mail@kaiploeger.net
"""

from casadi import *
import numpy as np
import matplotlib.pyplot as plt


dt = 0.1
num_steps = 200
num_dim = 3



def main():

    opti = Opti()

    # ---- decision variables ----
    X = opti.variable(2*num_dim, num_steps+1) # state trajectory
    U = opti.variable(num_dim, num_steps)
    pos = X[:num_dim, :]
    vel = X[num_dim:, :]
    acc = U

    # ---- cost ----
    cost = sum1(sum2(U**2))
    opti.minimize(cost)

    # ---- dynamics constraints ----
    f = lambda x, u: vertcat(x[num_dim:], u) # = [vel, acc]

    for k in range(num_steps): # loop over control intervals
        # Runge-Kutta 4 integration
        k1 = f(X[:,k],           U[:,k])
        k2 = f(X[:,k] + dt/2*k1, U[:,k])
        k3 = f(X[:,k] + dt/2*k2, U[:,k])
        k4 = f(X[:,k] + dt*k3,   U[:,k])
        x_next = X[:,k] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        opti.subject_to(X[:,k+1]==x_next) # close the gaps

    # ---- boundary conditions ----
    opti.subject_to(pos[:,0]==[0., 0., 0.])
    opti.subject_to(vel[:,0]==[0., 0., 0.])
    opti.subject_to(pos[:,-1]==[1., 2., 3.])
    opti.subject_to(vel[:,-1]==[0., 0., 0.])

    # ---- run solver ---
    opti.solver('ipopt')
    sol = opti.solve()

    # ---- plots ----
    pos = sol.value(pos)
    vel = sol.value(vel)
    acc = sol.value(acc)

    plt.figure()
    plt.title('pos')
    for i in range(pos.shape[0]):
        plt.plot(pos[i, :])
    plt.figure()
    plt.title('vel')
    for i in range(vel.shape[0]):
        plt.plot(vel[i, :])
    plt.figure()
    plt.title('acc')
    for i in range(acc.shape[0]):
        plt.plot(acc[i, :])
    plt.show()




if __name__ == '__main__':
    main()


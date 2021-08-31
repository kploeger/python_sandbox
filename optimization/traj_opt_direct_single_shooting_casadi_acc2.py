"""
    Direct single shooting trajectory optimization using CasADi with
    extra constraints on positions and jerk.

    dynamical system: simple double integrator
    controls:         accelerations
    cost:             sum of squared accelerations

    mail@kaiploeger.net
"""


import casadi as cas
import matplotlib.pyplot as plt
import numpy as np


dt = 0.01
num_steps = 100
num_dims = 3
pos0 = np.array([0, 0, 0.2])
vel0 = np.array([0, 0, 3.68])
kt = 60
post = np.array([0, 0, 0])
posT = np.array([0, 0, 0.2])
velT = np.array([0, 0, 3.68])
accT = np.array([0, 0,-9.81])
max_jerk = 500
cyclic = True


def main():

    opti = cas.Opti()

    # decision variables
    acc = opti.variable(num_dims, num_steps)

    # dynamics / integration
    pos = pos0 + vel0*dt + acc[:,0]*dt**2
    vel = vel0 + acc[:,0]*dt
    for k in range(1, num_steps):
        pos = cas.horzcat(pos, pos[:,-1] + vel[:,-1]*dt + 1/2*acc[:,k]*dt**2)
        vel = cas.horzcat(vel, vel[:,-1] + acc[:,k]*dt)

    # objective
    cost = cas.sum1(cas.sum2(acc**2))
    opti.minimize(cost)

    # constraints
    opti.subject_to(pos[:,kt]==post)
    opti.subject_to(pos[:,-1]==posT)
    opti.subject_to(vel[:,-1]==velT)
    opti.subject_to(acc[:,-1]==accT)

    jerk = cas.horzcat(cas.diff(acc, 1, 1), acc[:,0]-acc[:,-1]) / dt
    opti.subject_to(opti.bounded(-max_jerk, cas.vec(jerk), max_jerk))
    if cyclic:
        opti.subject_to(opti.bounded(
            -max_jerk, cas.vec(acc[:,-1]-acc[:,0])/dt, max_jerk))

    # solver
    opti.solver('ipopt')
    sol = opti.solve()

    # plots
    pos = np.concatenate([pos0.reshape((num_dims, 1)), sol.value(pos)], axis=1)
    vel = np.concatenate([vel0.reshape((num_dims, 1)), sol.value(vel)], axis=1)
    acc = sol.value(acc)

    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    ax[0].set_ylabel('position')
    ax[1].set_ylabel('velocity')
    ax[2].set_ylabel('acceleration')

    for i in range(num_dims):
        ax[0].plot(pos[i, :])
        ax[1].plot(vel[i, :])
        ax[2].step(np.arange(num_steps+1),
                   np.concatenate([acc[:, 0].reshape(num_dims,1),
                                   acc], axis=1)[i,:])
    plt.show()

if __name__ == '__main__':
    main()


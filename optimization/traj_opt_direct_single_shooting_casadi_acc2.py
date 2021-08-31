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


dt = 0.002
num_steps = 250
num_dims = 3

pos0 = np.array([0, 0, 0.2])
vel0 = np.array([0, 0, 3.68])
acc0 = np.array([0, 0, 0])  # the last acc applied  to reaach pos0, vel0

kt = 150
post = np.array([0, 0, 0])

posT = np.array([0, 0, 0.2])
velT = np.array([0, 0, 3.68])
accT = np.array([0, 0,-9.81])

max_jerk = 2500


def main():

    opti = cas.Opti()

    # decision variables
    acc = opti.variable(num_dims, num_steps)

    # dynamics / integration
    pos0_ = opti.parameter(num_dims, 1)
    vel0_ = opti.parameter(num_dims, 1)
    pos = pos0_
    vel = vel0_
    for k in range(0, num_steps):
        pos = cas.horzcat(pos, pos[:,-1] + vel[:,-1]*dt + 1/2*acc[:,k]*dt**2)
        vel = cas.horzcat(vel, vel[:,-1] + acc[:,k]*dt)

    # objective
    cost = cas.sum1(cas.sum2(acc**2))
    opti.minimize(cost)

    # constraints
    opti.set_value(pos0_, pos0)
    opti.set_value(vel0_, vel0)
    opti.subject_to(pos[:,kt]==post)
    opti.subject_to(pos[:,-1]==posT)
    opti.subject_to(vel[:,-1]==velT)
    opti.subject_to(acc[:,-1]==accT)

    jer = cas.horzcat(acc[:,0]-acc0, cas.diff(acc, 1, 1)) / dt
    opti.subject_to(opti.bounded(-max_jerk, cas.vec(jer), max_jerk))

    # solver
    opti.solver('ipopt')
    sol = opti.solve()
    pos, vel = sol.value(pos), sol.value(vel)
    acc, jer = sol.value(acc), sol.value(jer)


    # plots
    fig, ax = plt.subplots(4, 1, figsize=(12,9))
    ax[0].set_ylabel('position')
    ax[1].set_ylabel('velocity')
    ax[2].set_ylabel('acceleration')
    ax[3].set_ylabel('jerk')

    for i in range(num_dims):
        ax[0].plot(pos[i, :])
        ax[1].plot(vel[i, :])
        ax[2].step(np.arange(num_steps+1),
                   np.concatenate([acc[:, 0].reshape(num_dims,1),
                                   acc], axis=1)[i,:])
        ax[3].scatter(np.arange(len(jer[i,:])),jer[i,:], marker='.')
    plt.show()

if __name__ == '__main__':
    main()


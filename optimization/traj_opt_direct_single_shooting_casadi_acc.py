"""
    Simple direct single shooting trajectory optimization using CasADi.

    CasADi does not support compiling NLPs with Ipopt using the Opti interface.
    C code can be generated, but it throws an error while building.

    Just in time compilation works, but compilation time is way to huge.


    mail@kaiploeger.net
"""


import casadi as cas
import matplotlib.pyplot as plt
import numpy as np


dt = 0.1
num_steps = 100
num_dims = 3
pos0 = np.array([0, 0, 0])
vel0 = np.array([0, 0, 0])
posT = np.array([1, 2, 3])
velT = np.array([0, 0, 0])


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
    opti.subject_to(pos[:,-1]==[1., 2., 3.])
    opti.subject_to(vel[:,-1]==[0., 0., 0.])

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


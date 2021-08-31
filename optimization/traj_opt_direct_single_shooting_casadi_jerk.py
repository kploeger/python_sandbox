"""
    Direct single shooting trajectory optimization using CasADi.

    dynamical system: triple integrator
    controls:         jerk
    cost:             sum of squared jerks/accelerations/velocities

    mail@kaiploeger.net
"""


import casadi as cas
import numpy as np
import matplotlib.pyplot as plt


dt = 0.1
num_steps = 100
num_dims = 3


def main():

    opti = cas.Opti()

    # ---- decision variables ----
    jer = opti.variable(num_dims, num_steps)
    acc0 = opti.variable(num_dims, 1)

    # ---- dynamics / integration ----
    pos = np.array([0, 0, 0]).reshape(num_dims, 1)
    vel = np.array([0, 0, 0]).reshape(num_dims, 1)
    acc = acc0
    for k in range(0, num_steps):
        pos = cas.horzcat(pos, pos[:,-1] + vel[:,-1]*dt + 1/2*acc[:,-1]*dt**2 + 1/6*jer[:,k]*dt**3)
        vel = cas.horzcat(vel, vel[:,-1] + acc[:,-1]*dt + 1/2*jer[:,k]*dt**2)
        acc = cas.horzcat(acc, acc[:,-1] + jer[:,k]*dt)

    # ---- cost ----
    # cost = cas.sum1(cas.sum2(vel**2))
    # vel**2 converges to infeasible trajectory

    # cost = cas.sum1(cas.sum2(acc**2)) \
    # acc**2 as cost is problematic. Accelerations are interpolated linearly,
    # so acc[0] and acc[-1] only influence a single interval, while all other
    # acc[i] influence two intervals. Therefore more expensive to accelerate at
    # t=0 and t=T and the corresponding acc values will be lower than the rest.

    cost = cas.sum1(cas.sum2(jer**2))
    opti.minimize(cost)

    # ---- boundary conditions ----
    opti.subject_to(pos[:,-1]==[1., 2., 3.])
    opti.subject_to(vel[:,-1]==[0., 0., 0.])

    # ---- run solver ---
    opti.solver('ipopt')
    sol = opti.solve()

    # ---- plots ----
    pos = sol.value(pos)
    vel = sol.value(vel)
    acc = sol.value(acc)
    jer = sol.value(jer)

    fig, ax = plt.subplots(4, 1, figsize=(12,9))
    y_labels = ['position', 'velocity', 'acceleration', 'jerk']
    for i, name in enumerate(y_labels):
        ax[i].set_ylabel(name)
        ax[i].set_xlim((-5, num_steps+5))

    for i in range(num_dims):
        ax[0].plot(pos[i, :])
        ax[1].plot(vel[i, :])
        ax[2].plot(acc[i, :])
        ax[3].step(np.arange(num_steps+1),
                   np.concatenate([jer[:, 0].reshape(num_dims,1),
                                   jer], axis=1)[i,:])
    plt.show()


if __name__ == '__main__':
    main()


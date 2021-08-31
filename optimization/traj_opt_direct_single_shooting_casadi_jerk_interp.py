"""
    Direct single shooting trajectory optimization using CasADi and
    interpolation of acc/vel/pos within discretization steps.

    dynamical system: triple integrator
    controls:         jerk
    cost:             sum of squared jerks

    mail@kaiploeger.net
"""


import casadi as cas
import numpy as np
import matplotlib.pyplot as plt


dt = 0.1           # duration of discretization interval
T = 1.0            # time horizon
num_steps = 10     # number discretization intervals
num_dims = 3       # number of system dimensions
num_substeps = 10  # number of interpolated substeps


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
    cost = cas.sum1(cas.sum2(jer**2))
    opti.minimize(cost)

    # ---- boundary conditions ----
    opti.subject_to(pos[:,-1]==[1., 2., 3.])
    opti.subject_to(vel[:,-1]==[0., 0., 0.])

    # ---- run solver ---
    opti.solver('ipopt')
    sol = opti.solve()

    # ---- interpolation ----
    pos = sol.value(pos)
    vel = sol.value(vel)
    acc = sol.value(acc)
    jer = sol.value(jer)

    pos_ = np.zeros((num_dims, num_steps*num_substeps+1))
    vel_ = np.zeros((num_dims, num_steps*num_substeps+1))
    acc_ = np.zeros((num_dims, num_steps*num_substeps+1))

    for step in range(num_steps):
        pos_[:,step*num_substeps] = pos[:,step]
        vel_[:,step*num_substeps] = vel[:,step]
        acc_[:,step*num_substeps] = acc[:,step]
        for substep in range(1, num_substeps):
            t = dt*substep/num_substeps
            pos_[:,step*num_substeps+substep] = pos[:,step] \
                + vel[:,step]*t \
                + 1/2*acc[:,step]*t**2 \
                + 1/6*jer[:,step]*t**3
            vel_[:,step*num_substeps+substep] = vel[:,step] \
                + acc[:,step]*t \
                + 1/2*jer[:,step]*t**2
            acc_[:,step*num_substeps+substep] = acc[:,step] \
                + jer[:,step]*t

    pos_[:,-1] = pos[:,-1]
    vel_[:,-1] = vel[:,-1]
    acc_[:,-1] = acc[:,-1]

    # ---- plots ----
    fig, ax = plt.subplots(4, 1, figsize=(12,9))
    y_labels = ['position', 'velocity', 'acceleration', 'jerk']
    for i, name in enumerate(y_labels):
        ax[i].set_ylabel(name)
        ax[i].set_xlim((-0.5, num_steps+0.5))

    for i in range(num_dims):
        ax[0].plot(pos[i, :])
        ax[1].plot(vel[i, :])
        ax[2].plot(acc[i, :])
        ax[3].step(np.arange(num_steps+1),
                   np.concatenate([jer[:, 0].reshape(num_dims,1),
                                   jer], axis=1)[i,:])

    fig, ax = plt.subplots(4, 1, figsize=(12,9))
    y_labels = ['position', 'velocity', 'acceleration', 'jerk']
    for i, name in enumerate(y_labels):
        ax[i].set_ylabel(name)
        ax[i].set_xlim((-5, num_steps*num_substeps+5))

    for i in range(num_dims):
        ax[0].plot(pos_[i, :])
        ax[1].plot(vel_[i, :])
        ax[2].plot(acc_[i, :])
        ax[3].step(np.arange(num_steps+1)*num_substeps,
                   np.concatenate([jer[:, 0].reshape(num_dims,1),
                                   jer], axis=1)[i,:])
    plt.show()


if __name__ == '__main__':
    main()


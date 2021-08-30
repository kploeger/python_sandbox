"""
    How NOT to optimize trajectories:

    - simple single shooting method but with position as decision variable
    - cost on acc
    -> has to calculate 2nd derivative for cost
        -> huge variance
            -> only works for ~30 time steps
            -> converges suboptimally otherwise

    mail@kaiploeger.net
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


SHAPE = (58, 3)  # x in R^(n_steps, n_dim)
                 # start and stop are fixed


def cost(x, x0, xT, dx0, dxT):
    x_opt = x.reshape(SHAPE) # all posititions that get optimized
    pos = np.concatenate(([x0], x_opt, [xT]))
    vel = np.diff(pos, n=1, axis=0)
    vel = np.concatenate(([dx0], vel, [dxT]))
    acc = np.diff(vel, n=1, axis=0)
    cost = 0
    for acc_i in acc:
        cost += acc_i.T @ acc_i
    return cost


def main():

    n = SHAPE[1]

    # start at pos zero with vel zero
    x0, dx0 = np.zeros(n), np.zeros(n)

    # move to random pos with vel zero
    xT, dxT = np.array([1, 2, 3]), np.zeros(n)

    # start with linear interpolation as initial guess
    x = np.linspace(x0, xT, SHAPE[0])

    # quadratic cost on accelerations
    cost_ = lambda x: cost(x, x0, xT, dx0, dxT)

    # no explicit constraints necessary since the optimizer
    # does not get access to x0 and xT
    res = optimize.minimize(cost_, x, method="SLSQP")

    # reconstruct whole trajectory including x0 and xT
    x = np.concatenate(([x0], res.x.reshape(SHAPE), [xT]))
    dx = np.concatenate(([dx0], np.diff(x, n=1, axis=0), [dxT]))
    ddx = np.diff(dx, n=1, axis=0)

    # plot solution
    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    ax[0].set_ylabel('position')
    ax[1].set_ylabel('velocity')
    ax[2].set_ylabel('acceleration')

    for i in range(SHAPE[1]):
        ax[0].plot(x[:, i])
        ax[1].plot(dx[:, i])
        ax[2].plot(ddx[:, i])
    plt.show()



if __name__ == '__main__':
    main()


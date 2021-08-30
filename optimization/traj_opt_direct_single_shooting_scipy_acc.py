"""
    Simple direct single shooting trajectory optimization using SciPy.

    dynamical system: simple integrator
    controls:         acceleration
    cost:             squared acceleration

    -> approximates cubic splines

    mail@kaiploeger.net
"""

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt

import time


dt = 0.1
num_steps = 100
num_dims = 3

# start at pos zero with vel zero
pos0 = np.zeros(num_dims)
vel0 = np.zeros(num_dims)

# move to random pos with vel zero
posT = np.arange(num_dims)+1
velT = np.zeros(num_dims)


def integrate(acc):
    pos = np.zeros((num_steps+1, num_dims))
    vel = np.zeros((num_steps+1, num_dims))
    pos[0] = pos0
    vel[0] = vel0
    for i in range(num_steps):
        pos[i+1] = pos[i] + vel[i]*dt + 1/2*acc[i]*dt**2
        vel[i+1] = vel[i] + acc[i]*dt
    return pos, vel


def cost(acc):
    return np.sum(acc**2)


def constraints(acc):
    acc = acc.reshape((num_steps, num_dims))
    pos, vel = integrate(acc)
    pos_err_T = pos[-1] - posT
    vel_err_T = vel[-1] - velT
    return np.concatenate([pos_err_T, vel_err_T])


def main():
    acc_init = np.zeros((num_steps, num_dims))
    cons = ({'type': 'eq', 'fun':constraints})

    t0 = time.time()
    res = optimize.minimize(cost, acc_init, method="SLSQP", constraints=cons)
    print(f'time taken: {time.time()-t0}')

    acc = res.x.reshape((num_steps, num_dims))
    pos, vel = integrate(acc)

    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    ax[0].set_ylabel('position')
    ax[1].set_ylabel('velocity')
    ax[2].set_ylabel('acceleration')

    for i in range(num_dims):
        ax[0].plot(pos[:, i])
        ax[1].plot(vel[:, i])
        ax[2].step(np.arange(num_steps+1),
                   np.concatenate([acc[0, :].reshape(1,num_dims), acc])[:,i])
    plt.show()


if __name__ == '__main__':
    main()


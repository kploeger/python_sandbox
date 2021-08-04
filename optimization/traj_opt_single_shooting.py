"""
    Simple indirect trajectory optimization.

    dynamical system: simple integrator
    controls:         acceleration
    cost:             squared acceleration

    -> approximates cubic splines

    mail@kaiploeger.net
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


SHAPE = (199, 3)  # ddx in R^(n_steps, n_dim)
dt = 0.05


dim = SHAPE[1]
num_steps = SHAPE[0]

# start at pos zero with vel zero
x0, dx0 = np.zeros(dim), np.zeros(dim)

# move to random pos with vel zero
xT, dxT = np.array([1, 2, 3]), np.zeros(dim)
ddxT = np.zeros(dim)
ddxT[-1] = -1

# constraints in the middle:
# x50 = np.zeros(dim)
# dx100 = np.ones(dim) * (-0.05)


def cost(ddx):
    return np.sum(ddx**2)


def constraints(ddx):
    # integrate TODO: use better integration method
    dim = SHAPE[1]
    ddx = ddx.reshape(SHAPE)
    ddx = np.concatenate([[np.zeros(dim)], ddx])
    dx = np.cumsum(ddx, axis=0) * dt + dx0
    x = np.cumsum(dx, axis=0) * dt + x0

    # position constraints:
    # pos_err_50 = x[50] - x50
    pos_err_T = x[-1] - xT

    # velocity constraints:
    # vel_err_100 = dx[100] - dx100
    vel_err_T = dx[-1] - dxT

    # acceleration constraints:
    acc_err_T = ddx[-1] - ddxT


    return np.concatenate([\
                           # pos_err_50,
                           pos_err_T,
                           # vel_err_100,
                           vel_err_T,
                           # acc_err_T,
                           ])


def plot(x, dx, ddx):
    plt.figure()
    plt.title('pos')
    for i in range(x.shape[1]):
        plt.plot(x[:, i])
    plt.figure()
    plt.title('vel')
    for i in range(dx.shape[1]):
        plt.plot(dx[:, i])
    plt.figure()
    plt.title('acc')
    for i in range(ddx.shape[1]):
        plt.plot(ddx[:, i])
    plt.show()


def main():
    dim = SHAPE[1]
    x0, dx0 = np.zeros(dim), np.zeros(dim)
    ddx = np.zeros(SHAPE)

    cons = ({'type': 'eq', 'fun':constraints})

    res = optimize.minimize(cost, ddx, method="SLSQP", constraints=cons)

    ddx = res.x.reshape(SHAPE)
    ddx = np.concatenate([[np.zeros(dim)], ddx])
    dx = np.cumsum(ddx, axis=0) * dt + dx0
    x = np.cumsum(dx, axis=0) * dt + x0

    plot(x, dx, ddx)


if __name__ == '__main__':
    main()


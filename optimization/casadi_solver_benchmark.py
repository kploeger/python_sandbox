"""
    Benchmarking different linear solvers with IPOPT.
        --> hsl_ma27 and hsl_ma57 are similar and faster than standard mumps

    Direct single shooting trajectory optimization using CasADi.

    dynamical system: triple integrator
    controls:         jerk
    cost:             sum of squared accelerations

    mail@kaiploeger.net
"""


import casadi as cas
import numpy as np
import matplotlib.pyplot as plt

import time

dt = 0.1
num_steps = 100
num_dims = 3

solvers = ['ma27', 'ma57', 'ma77', 'ma86', 'ma97',
           #'pardiso',
           #'sprals',
           #'ampl',
           'mumps']

num_rollouts = 100

results = {}
for solver in solvers:
    results[solver] = []

def main():

    for i in range(num_rollouts):

        for solver in solvers:

            options = {'ipopt.linear_solver': solver}

            opti = cas.Opti()

            # ---- decision variables ----
            jer = opti.variable(num_dims, num_steps)
            acc0 = opti.variable(num_dims, 1)

            # ---- dynamics / integration ----
            pos0 = opti.parameter(num_dims, 1)
            vel0 = opti.parameter(num_dims, 1)
            pos = pos0
            vel = vel0
            acc = acc0
            for k in range(0, num_steps):
                pos = cas.horzcat(pos, pos[:,-1] + vel[:,-1]*dt + 1/2*acc[:,-1]*dt**2 + 1/6*jer[:,k]*dt**3)
                vel = cas.horzcat(vel, vel[:,-1] + acc[:,-1]*dt + 1/2*jer[:,k]*dt**2)
                acc = cas.horzcat(acc, acc[:,-1] + jer[:,k]*dt)

            cost = cas.sum1(cas.sum2(acc**2))
            opti.minimize(cost)

            # ---- boundary conditions ----
            opti.set_value(pos0, [0, 0, 0])
            opti.set_value(vel0, [0, 0, 0])
            opti.subject_to(pos[:,-1]==[1, 2, 3])
            opti.subject_to(vel[:,-1]==[0, 0, 0])

            # ---- run solver ---
            opti.solver('ipopt', options)

            sol = opti.solve() # first run includes building of the problem
            t0 = time.time()
            sol = opti.solve()
            t = time.time() - t0

            results[solver].append(t)

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
            #plt.show()

    for solver in solvers:
        print(f'{solver}:\tmean: {np.mean(results[solver])*1000:.3f}ms\t std: {np.std(results[solver])*1000:.3f}ms')



if __name__ == '__main__':
    main()


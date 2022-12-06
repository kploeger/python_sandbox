"""
    Testing some parameters of the ma57 linear solver.
        --> not seeing any improvement...

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


NUM_ROLLOUTS = 100

"""
https://www.coin-or.org/Bonmin/option_pages/options_list_ipopt.html#sec:MA57LinearSolver
"""

def test(num_rollouts,
         automatic_scaling='no',
         block_size=16,
         node_amalgamation=16,
         pivot_order=5):

    results = []

    options = {'ipopt.linear_solver': 'ma57',
               'ipopt.ma57_automatic_scaling': automatic_scaling,
               'ipopt.ma57_block_size': block_size,
               'ipopt.ma57_node_amalgamation': node_amalgamation,
               'ipopt.ma57_pivot_order': pivot_order}

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

    # ---- cost ----
    # cost = cas.sum1(cas.sum2(vel**2))
    # vel**2 converges to infeasible trajectory

    # cost = cas.sum1(cas.sum2(acc**2)) \
    # acc**2 as cost is problematic. Accelerations are interpolated linearly,
    # so acc[0] and acc[-1] only influence a single interval, while all other
    # acc[i] influence two intervals. Therefore more expensive to accelerate at
    # t=0 and t=T and the corresponding acc values will be lower than the rest.

    cost = cas.sum1(cas.sum2(acc**2))
    opti.minimize(cost)

    # ---- boundary conditions ----
    opti.set_value(pos0, [0, 0, 0])
    opti.set_value(vel0, [0, 0, 0])
    opti.subject_to(pos[:,-1]==[1, 2, 3])
    opti.subject_to(vel[:,-1]==[0, 0, 0])

    # ---- run solver ---
    opti.solver('ipopt', options)
    sol = opti.solve()

    for i in range(num_rollouts):

        t0 = time.time()
        sol = opti.solve()
        t = time.time() - t0
        results.append(t)

    return results


def main():
    res_default = test(num_rollouts=NUM_ROLLOUTS)
    res_scaling = test(num_rollouts=NUM_ROLLOUTS, automatic_scaling='yes')
    res_block_size_1 = test(num_rollouts=NUM_ROLLOUTS, block_size=1)
    res_block_size_100 = test(num_rollouts=NUM_ROLLOUTS, block_size=100)
    res_node_am_1 = test(num_rollouts=NUM_ROLLOUTS, node_amalgamation=1)
    res_node_am_100 = test(num_rollouts=NUM_ROLLOUTS, node_amalgamation=100)
    res_pivot_0 = test(num_rollouts=NUM_ROLLOUTS, pivot_order=0)

    print(f'default:\t{np.mean(res_default)*1000:.3f}+-{np.std(res_default)*1000:.3f}ms')
    print(f'scaling:\t{np.mean(res_scaling)*1000:.3f}+-{np.std(res_scaling)*1000:.3f}ms')
    print(f'block 1:\t{np.mean(res_block_size_1)*1000:.3f}+-{np.std(res_block_size_1)*1000:.3f}ms')
    print(f'block 100:\t{np.mean(res_block_size_100)*1000:.3f}+-{np.std(res_block_size_100)*1000:.3f}ms')
    print(f'node am 1:\t{np.mean(res_node_am_1)*1000:.3f}+-{np.std(res_node_am_1)*1000:.3f}ms')
    print(f'node am 100:\t{np.mean(res_node_am_100)*1000:.3f}+-{np.std(res_node_am_100)*1000:.3f}ms')
    print(f'pivot 0:\t{np.mean(res_pivot_0)*1000:.3f}+-{np.std(res_pivot_0)*1000:.3f}ms')


if __name__ == '__main__':
    main()


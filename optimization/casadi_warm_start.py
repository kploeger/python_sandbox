"""
    Testing the warm start feature of casadi.
        --> significant speedup!

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

pos0 = np.array([0, 0, 0])
vel0 = np.array([0, 0, 0])
posT = np.array([1, 2, 3])
velT = np.array([0, 0, 0])


NUM_ROLLOUTS = 10000
PLOT = False

"""
https://groups.google.com/g/casadi-users/c/DMrWvz3xh5s
"""

def test(num_rollouts,
         warm_start = False):

    results = []

    options_warm = {'ipopt.linear_solver': 'ma57',
                    'ipopt.warm_start_init_point': 'yes'}

    # these options are related, but don't seem to have impact
    # 'ipopt.warm_start_bound_frac' : 1e-16,
    # 'ipopt.warm_start_bound_push' : 1e-16,
    # 'ipopt.warm_start_mult_bound_push' : 1e-16,
    # 'ipopt.warm_start_slack_bound_frac' : 1e-16,
    # 'ipopt.warm_start_slack_bound_push' : 1e-16}

    options_cold = {'ipopt.linear_solver': 'ma57'}


    # ---- decision variables ----
    cacc0 = cas.SX.sym("acc0", num_dims, 1)
    cjer = cas.SX.sym("jer", num_dims, num_steps)
    dec_vars = cas.vertcat(cacc0, cas.vec(cjer))


    # ---- dynamics / integration ----
    cpos = cas.SX(pos0)
    cvel = cas.SX(vel0)
    cacc = cas.SX(cacc0)

    for k in range(0, num_steps):
        cpos = cas.horzcat(cpos, cpos[:,-1] + cvel[:,-1]*dt + 1/2*cacc[:,-1]*dt**2 + 1/6*cjer[:,k]*dt**3)
        cvel = cas.horzcat(cvel, cvel[:,-1] + cacc[:,-1]*dt + 1/2*cjer[:,k]*dt**2)
        cacc = cas.horzcat(cacc, cacc[:,-1] + cjer[:,k]*dt)

    # ---- cost ----
    cost = cas.sum1(cas.sum2(cacc**2)) / num_steps


    # ---- boundary conditions ----
    cons = cas.vertcat(cpos[:,-1] - posT,
                       cvel[:,-1] - velT)
    lbg = np.zeros(cons.shape[0])
    ubg = np.zeros(cons.shape[0])


    # ---- solver ----
    nlp = {"x": dec_vars, "f": cost, "g": cons}
    solver0 = cas.nlpsol('solver', 'ipopt', nlp, options_cold)
    sol = solver0(lbg=lbg, ubg=ubg)
    if warm_start:
        solver_eval = cas.nlpsol('solver', 'ipopt', nlp, options_warm)
    else:
        solver_eval = cas.nlpsol('solver', 'ipopt', nlp, options_cold)


    # ---- evaluate ----
    for i in range(num_rollouts):
        if warm_start:
            t0 = time.time()
            solver_eval(lbg=lbg, ubg=ubg, x0=sol["x"], lam_g0=sol["lam_g"], lam_x0=sol["lam_x"])
            t = time.time() - t0
        else:
            t0 = time.time()
            solver_eval(lbg=lbg, ubg=ubg)
            t = time.time() - t0

        results.append(t)


    # ---- plot ----
    if PLOT:
        pos = np.zeros((num_dims, num_steps+1))
        vel = np.zeros((num_dims, num_steps+1))
        acc = np.zeros((num_dims, num_steps+1))
        jer = np.zeros((num_dims, num_steps))
        pos[:,0] = pos0
        vel[:,0] = vel0
        acc[:,0] = np.array(sol["x"])[:num_dims].flat
        jer = np.array(sol["x"])[num_dims:].reshape(num_steps, num_dims).T

        for k in range(0, num_steps):
            pos[:,k+1] = pos[:,k] + vel[:,k]*dt + 1/2*acc[:,k]*dt**2 + 1/6*jer[:,k]*dt**3
            vel[:,k+1] = vel[:,k] + acc[:,k]*dt + 1/2*jer[:,k]*dt**2
            acc[:,k+1] = acc[:,k] + jer[:,k]*dt

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

    return results


def main():
    res_default = test(num_rollouts=NUM_ROLLOUTS)
    res_warm_start = test(num_rollouts=NUM_ROLLOUTS, warm_start=True)

    print(f'\ndefault:\t{np.mean(res_default)*1000:.3f}+-{np.std(res_default)*1000:.3f}ms')
    print(f'warm start:\t{np.mean(res_warm_start)*1000:.3f}+-{np.std(res_warm_start)*1000:.3f}ms')


if __name__ == '__main__':
    main()


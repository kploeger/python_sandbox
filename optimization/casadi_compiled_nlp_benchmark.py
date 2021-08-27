"""
    Testing performance improvement of compiling CasADi NLPs using the Ipopt
    solver on a simple direct singel shooting problem.

    - compilation takes very long
    - the compiled nlp is only slightly faster ~10%
    - code compilation is not supported for CasADi's "Opti" interface
        -> the regular interface only returns decision variables (not pos, vel)

    => compiling code is definitely not worth it

    mail@kaiploeger.net
"""


import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import timeit


# compile options
recompile = True
name = __file__[:-3] # name of the .c and .so file:

# nlp options
dt = 0.1
num_steps = 500
num_dims = 3
pos0 = np.array([0, 0, 0])
vel0 = np.array([0, 0, 0])
posT = np.array([1, 2, 3])
velT = np.array([0, 0, 0])

# solver options
options = {'print_time': 0,
           'ipopt.print_level': 0}


def main():
    # optimization variables
    acc = cas.MX.sym("acc", num_dims, num_steps)

    # dynamics / integration
    pos = pos0 + vel0*dt + acc[:,0]*dt**2
    vel = vel0 + acc[:,0]*dt
    for k in range(1, num_steps):
        pos = cas.horzcat(pos, pos[:,-1] + vel[:,-1]*dt + acc[:,k]*dt**2)
        vel = cas.horzcat(vel, vel[:,-1] + acc[:,k]*dt)

    # cost
    cost = cas.sum1(cas.sum2(acc**2))

    # constraints
    cons = cas.vertcat(pos[:,-1] - posT,
                       vel[:,-1] - posT)

    lbg = np.array([0, 0, 0, 0, 0, 0])  # lower bound
    ubg = np.array([0, 0, 0, 0, 0, 0])  # upper bound

    # definition of nonlinear problem
    nlp = {"x": cas.vec(acc), "f": cost, "g": cons}

    # uncompiled solver
    solver = cas.nlpsol('solver', 'ipopt', nlp, options)
    T_normal = timeit.timeit(
        lambda: solver(lbg=lbg, ubg=ubg), number=1000)/1000
    print('normal:', T_normal)

    # compiled solver
    if recompile:
        solver.generate_dependencies(name+".c")
        subprocess.Popen(["gcc","-fPIC","-shared", "-O3",
                          name+".c","-o", name+".so"]).wait()

    solver = cas.nlpsol("solver", "ipopt", name+".so", options)
    T_compiled = timeit.timeit(
        lambda: solver(lbg=lbg, ubg=ubg), number=1000)/1000
    print('compiled:', T_compiled)

    # plots
    sol = solver(lbg=lbg, ubg=ubg)
    acc = np.array(sol["x"]).reshape(num_steps, num_dims).T

    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    ax[0].set_ylabel('position')
    ax[1].set_ylabel('velocity')
    ax[2].set_ylabel('acceleration')

    for i in range(num_dims):
        # ax[0].plot(pos[i, :])
        # ax[1].plot(vel[i, :])
        ax[2].step(np.arange(num_steps+1),
                   np.concatenate([acc[:, 0].reshape(num_dims,1),
                                   acc], axis=1)[i,:])
    plt.show()


if __name__ == '__main__':
    main()


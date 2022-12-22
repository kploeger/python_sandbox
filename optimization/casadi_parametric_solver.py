"""
    Testing how to solve a parametric nlp without rebuilding the solver.
    Not rebuilding the solver saves a lot of time.

    mail@kaiploeger.net
"""

import casadi as cas
import time

# decision variables
x = cas.SX.sym("x",2)

# parameters
p = cas.SX.sym("p",2)

# objective
f = p[0]*x[0]**2 + p[1]*x[1]**2

# constraints
g = x[0]+x[1]-10      # constraint

# build solver
t0 = time.time()
nlp = {'x':x, 'f':f, 'g':g, 'p':p}
solver = cas.nlpsol("solver", "ipopt", nlp, {})
T_build = time.time() - t0

# solve problem
x0 = [1, 1]
p = [1, 1]

t0 = time.time()
sol=solver(x0=x0, p=p)
T_solve_1 = time.time() - t0

# solve it again
t0 = time.time()
sol=solver(x0=x0, p=p)
T_solve_2 = time.time() - t0

# how did we do?
print(f'build time: {T_build*1000:.3f}ms')
print(f'first solve: {T_solve_1*1000:.3f}ms')
print(f'second solve: {T_solve_2*1000:.3f}ms')

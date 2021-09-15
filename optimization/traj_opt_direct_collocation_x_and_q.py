"""
    Direct collocation trajectory optimization using CasADi, jointly optimizing
    in joint space and task space.

    dynamical system: 2D reacher with link lengths 1
    controls:         accelerations
    cost:             sum of squared accelerations in joint or task space

    Requires very good initial solution to converge.

    mail@kaiploeger.net
"""


import casadi as cas
import numpy as np
import matplotlib.pyplot as plt


dt = 0.1           # duration of discretization interval
T = 1.0            # time horizon
num_steps = 10     # number discretization intervals
num_dims = 2       # number of system dimensions
num_joints = 2
num_substeps = 10  # number of interpolated substeps


# ---- symbolic and numeric kinematics ----
def fkin_sym(q):
    return cas.vertcat(cas.cos(q[0]) + cas.cos(q[0]+q[1]),
                       cas.sin(q[0]) + cas.sin(q[0]+q[1]))

def fkin(q):
    return np.array([np.cos(q[0]) + np.cos(q[0]+q[1]),
                     np.sin(q[0]) + np.sin(q[0]+q[1])])

def ikin_sym(x):
    l = cas.sqrt(x[0]**2+x[1]**2)
    alpha = cas.arctan(x[1]/x[0])
    return cas.vertcat(alpha + cas.arccos(l/2),
                       cas.arccos(1-l**2/2))

def ikin(x):
    l = np.sqrt(x[0]**2+x[1]**2)
    alpha = np.arctan(x[1]/x[0])
    return np.array([alpha + np.arccos(l/2),
                     np.arccos(1-l**2/2)])

def jac_sym(q):
    return cas.horzcat( \
            cas.vertcat(-cas.sin(q[0])-cas.sin(q[0]+q[1]),-cas.sin(q[0]+q[1])),
            cas.vertcat( cas.cos(q[0])+cas.cos(q[0]+q[1]), cas.cos(q[0]+q[1])))

def jac(q):
    return np.array([[-np.cas.sin(q[0])-np.sin(q[0]+q[1]), -np.sin(q[0]+q[1])],
                     [ np.cos(q[0])+np.cos(q[0]+q[1]),      np.cos(q[0]+q[1])]])



def main():

    opti = cas.Opti()

    # ---- decision variables ----
    acc = opti.variable(num_dims, num_steps)
    vel = opti.variable(num_dims, num_steps+1)
    pos = opti.variable(num_dims, num_steps+1)
    accq = opti.variable(num_joints, num_steps)
    velq = opti.variable(num_joints, num_steps+1)
    posq = opti.variable(num_joints, num_steps+1)


    # ---- cost ----
    # cost = cas.sum1(cas.sum2(accq**2))
    cost = cas.sum1(cas.sum2(acc**2))
    opti.minimize(cost)


    # ---- dynamics / integration ----
    for k in range(num_steps):
        opti.subject_to(pos[:,k+1] == pos[:,k] + vel[:,k]*dt + 1/2*acc[:,k]*dt**2)
        opti.subject_to(vel[:,k+1] == vel[:,k] + acc[:,k]*dt)
        opti.subject_to(posq[:,k+1] == posq[:,k] + velq[:,k]*dt + 1/2*accq[:,k]*dt**2)
        opti.subject_to(velq[:,k+1] == velq[:,k] + accq[:,k]*dt)


    # ---- kinematics ----
    # TODO: constraints on velocities only is not sufficient,
    #       but constraints on positions crash
    # opti.subject_to(posq[:,0]==ikin_sym(pos[:,0]))
    # opti.subject_to(pos[:,0]==fkin_sym(posq[:,0]))
    for k in range(num_steps+1):
        opti.subject_to(vel[:, k]==jac_sym(posq[:, k]) @ velq[:, k])


    # ---- boundary conditions ----
    opti.subject_to(pos[:,0]==[1, 0])
    opti.subject_to(vel[:,0]==[0, 0])
    opti.subject_to(pos[:,-1]==[1.5, 0])
    opti.subject_to(vel[:,-1]==[0, 0])


    # ---- initial guess for joint positions ----
    pos_guess = np.array([[1.        , 1.01363636, 1.05151515, 1.10757576, 1.17575758, 1.25      , 1.32424242, 1.39242424, 1.44848485, 1.48636364, 1.5       ],
                        [0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]])
    for k in range(num_steps):
        opti.set_initial(posq[:,k], ikin(pos_guess[:,k]))


    # ---- run solver ---
    opti.solver('ipopt')
    sol = opti.solve()

    pos = sol.value(pos)
    vel = sol.value(vel)
    acc = sol.value(acc)
    posq = sol.value(posq)
    velq = sol.value(velq)
    accq = sol.value(accq)


    # ---- plots ----
    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    ax[0].set_title('tool - decision vars')
    y_labels = ['position', 'velocity', 'acceleration']
    for i, name in enumerate(y_labels):
        ax[i].set_ylabel(name)
        ax[i].set_xlim((-0.5, num_steps+0.5))

    for i in range(num_dims):
        ax[0].plot(pos[i, :])
        ax[1].plot(vel[i, :])
        ax[2].step(np.arange(num_steps+1),
                   np.concatenate([acc[:, 0].reshape(num_dims,1),
                                   acc], axis=1)[i,:])

    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    ax[0].set_title('joints - decision vars')
    y_labels = ['position', 'velocity', 'acceleration']
    for i, name in enumerate(y_labels):
        ax[i].set_ylabel(name)
        ax[i].set_xlim((-0.5, num_steps+0.5))

    for i in range(num_dims):
        ax[0].plot(posq[i, :])
        ax[1].plot(velq[i, :])
        ax[2].step(np.arange(num_steps+1),
                   np.concatenate([accq[:, 0].reshape(num_dims,1),
                                   accq], axis=1)[i,:])
    plt.show()


if __name__ == '__main__':
    main()


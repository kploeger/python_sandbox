"""
    mail@kaiploeger.net
"""

import pinocchio as pin
from pinocchio import casadi as cpin

import casadi as cas
from casadi import SX
import numpy as np

import matplotlib.pyplot as plt


URDF_PATH = "../robot_description/urdf/robot/wam_4dof.urdf"

num_steps = 50
dt = 0.1

options = {}


def cfkin(cmodel, cdata, frame_id, cq):
    cpin.forwardKinematics(cmodel, cdata, cq)
    cpin.updateFramePlacement(cmodel, cdata, frame_id)
    x = cdata.oMf[frame_id].translation
    return x


def main():
    # model = pin.buildSampleModelHumanoidRandom()
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    tool_id = cmodel.getFrameId('tool')

    cq = SX.sym("q", cmodel.nq, 1)
    cdq = SX.sym("dq", cmodel.nv, 1)
    cddq = SX.sym("ddq", cmodel.nv, 1)

    cfkin(cmodel, cdata, tool_id, cq)

    # constraint values
    q0 = np.array([0, 0, 0, 0])
    dq0 = np.array([0, 0, 0, 0])
    ddq0 = np.array([0, 0, 0, 0])

    qT = np.array([1, 2, 3, 4])
    dqT = np.array([0, 0, 0, 0])
    ddqT = np.array([0, 0, 0, 0])


    # decision vars
    cdddq = cas.SX.sym("dddq", cmodel.nq, num_steps)


    # integration
    # cq = q0 + dq0*dt + 1/2*ddq0*dt**2 + 1/6*cdddq[:,0]*dt**3
    # cdq = dq0 + ddq0*dt + 1/2*cdddq[:,0]*dt**2
    # cddq = ddq0 + cdddq[:,0]*dt

    cq = q0.reshape((4,1))
    cdq = dq0.reshape((4,1))
    cddq = ddq0.reshape((4,1))

    for k in range(0, num_steps):
        cq = cas.horzcat(cq, cq[:,-1] + cdq[:,-1]*dt + 1/2*cddq[:,-1]*dt**2 + 1/6*cdddq[:,k]*dt**3)
        cdq = cas.horzcat(cdq, cdq[:,-1] + cddq[:,-1]*dt + 1/2*cdddq[:,k]*dt**2)
        cddq = cas.horzcat(cddq, cddq[:,-1] + cdddq[:,k]*dt)


    # cost
    # cost = cas.sum1(cas.sum2(cddq**2))
    cost = cas.sum1(cas.sum2(cdddq**2))


    # constraints
    cons = cas.vertcat(cq[:,-1] - qT,
                       cdq[:,-1] - dqT,
                       cddq[:,-1] - ddqT)

    # constraint bounds
    lbg = np.zeros(np.shape(cons)[0])
    ubg = np.zeros(np.shape(cons)[0])

    # nlp definition
    nlp = {"x": cas.vec(cdddq), "f": cost, "g": cons}

    # solver
    solver = cas.nlpsol('solver', 'ipopt', nlp, options)
    sol = solver(lbg=lbg, ubg=ubg)


    # plots
    dddq = np.array(sol["x"]).reshape(num_steps, cmodel.nq).T

    # compute trajectory:
    ddq = np.zeros((cmodel.nq, num_steps+1))
    dq = np.zeros((cmodel.nq, num_steps+1))
    q = np.zeros((cmodel.nq, num_steps+1))
    q[:,0] = q0
    dq[:,0] = dq0
    ddq[:,0] = ddq0
    for k in range(0, num_steps):
        q[:,k+1] = q[:,k] + dq[:,k]*dt + 1/2*ddq[:,k]*dt**2 + 1/6*dddq[:,k]*dt**3
        dq[:,k+1] = dq[:,k] + ddq[:,k]*dt + 1/2*dddq[:,k]*dt**2
        ddq[:,k+1] = ddq[:,k] + dddq[:,k]*dt

    print('q:', q[:,-1])
    print('dq:', dq[:,-1])
    print('ddq:', ddq[:,-1])
    print('dddq:', dddq[:,-1])


    fig, ax = plt.subplots(cmodel.nq, 1, figsize=(12,9))
    ax[0].set_ylabel('position')
    ax[1].set_ylabel('velocity')
    ax[2].set_ylabel('acceleration')
    ax[3].set_ylabel('jerk')

    for i in range(cmodel.nq):
        ax[0].plot(q[i, :])
        ax[1].plot(dq[i, :])
        ax[2].plot(ddq[i,:])
        ax[3].step(np.arange(num_steps+1),
                   np.concatenate([dddq[:, 0].reshape(cmodel.nq,1),
                                   dddq], axis=1)[i,:])
    plt.show()


if __name__ == '__main__':
    main()


"""
    mail@kaiploeger.net
"""

import pinocchio as pin
from pinocchio import casadi as cpin  # requires pinocchio >= 2.9.0

import casadi as cas
from casadi import SX
import numpy as np

import matplotlib.pyplot as plt


URDF_PATH = "../robot_description/urdf/robot/wam_4dof.urdf"

num_steps = 50
dt = 0.1

options = {}


def fkin(model, data, frame_id, q):
    """ numeric forward kinematics """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, frame_id)
    x = data.oMf[frame_id].translation
    return x

def dfkin(model, data, frame_id, q, dq, ddq):
    """ numeric forward kinematics with derivatives """
    ref_frame = pin.LOCAL_WORLD_ALIGNED
    pin.forwardKinematics(model, data, q, dq, ddq)
    pin.updateFramePlacement(model, data, frame_id)
    x = data.oMf[frame_id].translation
    dx = pin.getFrameVelocity(model, data, frame_id, ref_frame).linear
    ddx = pin.getFrameClassicalAcceleration(model, data, frame_id, ref_frame).linear
    return x, dx, ddx

def cfkin(cmodel, cdata, frame_id, cq):
    """ symbolic forward kinematics """
    cpin.forwardKinematics(cmodel, cdata, cq)
    cpin.updateFramePlacement(cmodel, cdata, frame_id)
    x = cdata.oMf[frame_id].translation
    return x

def cdfkin(cmodel, cdata, frame_id, cq, cdq, cddq):
    """ symbolic forward kinematics with derivatives """
    ref_frame = pin.LOCAL_WORLD_ALIGNED
    cpin.forwardKinematics(cmodel, cdata, cq, cdq, cddq)
    cpin.updateFramePlacement(cmodel, cdata, frame_id)
    cx = cdata.oMf[frame_id].translation
    cdx = cpin.getFrameVelocity(cmodel, cdata, frame_id, ref_frame).linear
    cddx = cpin.getFrameClassicalAcceleration(cmodel, cdata, frame_id, ref_frame).linear
    return cx, cdx, cddx


def main():
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    tool_id = cmodel.getFrameId('tool')


    # constraint values
    q0 = np.array([0, 0, 0, 0])
    dq0 = np.array([0, 0, 0, 0])
    ddq0 = np.array([0, 0, 0, 0])

    xT = np.array([0.32, -0.03, 2.04])
    dxT = np.array([0, 0, 0])
    ddxT = np.array([0, 0, 0])

    active_joints = np.array([0, 1, 3])
    num_active_joints = len(active_joints)


    # decision vars
    cdddq_active = cas.SX.sym("dddq", num_active_joints, num_steps)  # plan in jerk
    cqT_active = cas.SX.sym("qT", num_active_joints, 1)              # and state after defect
    cdqT_active = cas.SX.sym("dqT", num_active_joints, 1)
    cddqT_active = cas.SX.sym("ddqT", num_active_joints, 1)
    dec_vars = cas.horzcat(cdddq_active, cqT_active, cdqT_active, cddqT_active)


    # integration
    cdddq = cas.SX.zeros(cmodel.nv, num_steps)
    cqT = cas.SX(q0)
    cdqT = cas.SX.zeros(cmodel.nv, 1)
    cddqT = cas.SX.zeros(cmodel.nv, 1)

    cdddq[active_joints,:] = cdddq_active
    cqT[active_joints,:] = cqT_active
    cdqT[active_joints,:] = cdqT_active
    cddqT[active_joints,:] = cddqT_active

    cq = cas.SX(q0)
    cdq = cas.SX(dq0)
    cddq = cas.SX(ddq0)

    for k in range(0, num_steps-1):
        cq = cas.horzcat(cq, cq[:,-1] + cdq[:,-1]*dt + 1/2*cddq[:,-1]*dt**2 + 1/6*cdddq[:,k]*dt**3)
        cdq = cas.horzcat(cdq, cdq[:,-1] + cddq[:,-1]*dt + 1/2*cdddq[:,k]*dt**2)
        cddq = cas.horzcat(cddq, cddq[:,-1] + cdddq[:,k]*dt)

    cq = cas.horzcat(cq, cqT)
    cdq = cas.horzcat(cdq, cdqT)
    cddq = cas.horzcat(cddq, cddqT)


    # defect constraint
    cons = cas.vertcat(cq[:,-2] + cdq[:,-2]*dt + 1/2*cddq[:,-2]*dt**2 + 1/6*cdddq[:,-1] - cq[:,-1],
                       cdq[:,-2] + cddq[:,-2]*dt + 1/2*cdddq[:,-1]*dt**2 - cdq[:,-1],
                       cddq[:,-2] + cdddq[:,-1]*dt - cddq[:,-1])


    # cost
    cost = 0
    cost += cas.sum1(cas.sum2(cddq**2))  # squared accelerations
    # cost += cas.sum1(cas.sum2(cdddq**2))  # squared jerk


    # constraints
    cxT, cdxT, cddxT = cdfkin(cmodel, cdata, tool_id, cq[:,-1], cdq[:,-1], cddq[:,-1])
    cons = cas.vertcat(cons,
                       cxT - xT,
                       cdxT - dxT,
                       cddxT - ddxT)

    lbg = np.zeros(cons.shape[0])
    ubg = np.zeros(cons.shape[0])

    max_jerk = 0.2
    cons = cas.vertcat(cons, cdddq.reshape((cmodel.nv*num_steps, 1)))
    lbg = np.concatenate((lbg,-max_jerk*np.ones(cmodel.nv*num_steps)))
    ubg = np.concatenate((ubg, max_jerk*np.ones(cmodel.nv*num_steps)))


    # solver
    nlp = {"x": cas.vec(dec_vars), "f": cost, "g": cons}
    solver = cas.nlpsol('solver', 'ipopt', nlp, options)
    sol = solver(lbg=lbg, ubg=ubg)


    # restore solution:
    dddq = np.zeros((cmodel.nq, num_steps))
    ddq = np.zeros((cmodel.nq, num_steps+1))
    dq = np.zeros((cmodel.nq, num_steps+1))
    q = np.zeros((cmodel.nq, num_steps+1))
    dddq[active_joints,:] = np.array(sol["x"])[:num_steps*num_active_joints]\
                                      .reshape(num_steps, num_active_joints).T
    q[:,0] = q0
    dq[:,0] = dq0
    ddq[:,0] = ddq0
    for k in range(0, num_steps):
        q[:,k+1] = q[:,k] + dq[:,k]*dt + 1/2*ddq[:,k]*dt**2 + 1/6*dddq[:,k]*dt**3
        dq[:,k+1] = dq[:,k] + ddq[:,k]*dt + 1/2*dddq[:,k]*dt**2
        ddq[:,k+1] = ddq[:,k] + dddq[:,k]*dt

    x = np.zeros((3, num_steps+1))
    dx = np.zeros((3, num_steps+1))
    ddx = np.zeros((3, num_steps+1))
    for k in range(0, num_steps+1):
        x[:,k], dx[:,k], ddx[:,k] = dfkin(model, data, tool_id, q[:,k], dq[:,k], ddq[:,k])


    print('\nfinal state:')
    print('qT:   ', q[:,-1])
    print('dqT:  ', dq[:,-1])
    print('ddqT: ', ddq[:,-1])
    print('dddqT:', dddq[:,-1])
    print('xT:   ', x[:,-1])
    print('dxT:   ', dx[:,-1])
    print('ddxT:   ', ddx[:,-1])


    # plots
    fig, ax = plt.subplots(cmodel.nq, 1, figsize=(12,9))
    ax[0].set_title('joint space')
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

    # task space
    x_colors = ['red', 'green', 'blue']
    fig, ax = plt.subplots(3, 1, figsize=(12,9))
    ax[0].set_title('task space')
    ax[0].set_ylabel('position')
    ax[1].set_ylabel('velocity')
    ax[2].set_ylabel('acceleration')

    for i in range(3):
        ax[0].plot(x[i, :], color=x_colors[i])
        ax[1].plot(dx[i, :], color=x_colors[i])
        ax[2].plot(ddx[i,:], color=x_colors[i])

    plt.show()


if __name__ == '__main__':
    main()


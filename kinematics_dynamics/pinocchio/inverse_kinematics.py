"""
    mail@kaiploeger.net
"""

import pinocchio as pin
import numpy as np
import time


URDF_PATH = "../robot_description/urdf/robot/wam_4dof.urdf"


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


def ikin(model, data, frame_id, x_des, q0=None):
    """ numeric inverse kineatics """
    ref_frame = pin.LOCAL_WORLD_ALIGNED
    eps      = 1e-6
    max_iter = 100
    alpha    = 0.5
    reg      = 1e-15

    if q0 is None:
        q = pin.neutral(model)
    else:
        q = q0

    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, frame_id)

        dx = data.oMf[frame_id].translation - x_des

        if np.linalg.norm(dx) < eps:
            break

        J = pin.computeFrameJacobian(model, data, q, frame_id, ref_frame)[:3, :]

        # q -= alpha J^T(J J^T)^(-1) err
        # use pin.integrate when also using orientations
        q -= alpha * J.T @ np.linalg.solve(J@J.T+reg*np.eye(3), dx)

    else:
        raise ValueError(f'Could not reach position {x_des}.')

    return q

def dikin(model, data, frame_id, x_des, dx_des, ddx_des, q0=None):
    """ numeric inverse kineatics with derivatives """
    ref_frame = pin.LOCAL_WORLD_ALIGNED
    reg      = 1e-17

    q = ikin(model, data, frame_id, x_des, q0)

    J = pin.computeFrameJacobian(model, data, q, frame_id, ref_frame)[:3, :]
    dq = J.T @ np.linalg.solve(J@J.T+reg*np.eye(3), dx_des)

    pin.forwardKinematics(model, data, q, dq, np.zeros_like(dq))
    dJdq = pin.getFrameClassicalAcceleration(model, data, frame_id, pin.LOCAL_WORLD_ALIGNED).linear
    ddq = J.T @ np.linalg.solve(J@J.T+reg*np.eye(3), ddx_des - dJdq)

    return q, dq, ddq


def main():

    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    frame_id = model.getFrameId('tool')

    x_des = np.array([0.3, 0, 2])

    q0 = np.array([0.1, 0.1, 0.1, 0.1])

    t0 = time.time()
    for _ in range(1000):
        q = ikin(model, data, frame_id, x_des, q0)
    T = time.time() - t0

    print(f'Inverse kinematics take {T*1000:.2f}ns to compute.\n')

    print('Derivatives:')

    x_des = np.array([0.3, 0, 2])
    dx_des = np.array([0.1, 0.2, 0.3])
    ddx_des = np.array([0.2, 0.4, 0.2])

    print('\ndesired cart state:')
    print('x:  ', x_des)
    print('dx: ', dx_des)
    print('ddx:', ddx_des)

    q, dq, ddq = dikin(model, data, frame_id, x_des, dx_des, ddx_des)

    print('\nfound joint state:')
    print('q:  ', q)
    print('dq: ', dq)
    print('ddq:', ddq)

    x, dx, ddx = dfkin(model, data, frame_id, q, dq, ddq)

    print('\nfound cart state:')
    print('x:  ', x)
    print('dx: ', dx)
    print('ddx:', ddx)

    print('\ncart error:')
    print('x:  ', x - x_des)
    print('dx:  ', dx - dx_des)
    print('ddx:  ', ddx - ddx_des)


if __name__ == '__main__':
    main()


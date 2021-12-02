"""
    mail@kaiploeger.net
"""

import pinocchio as pin
import numpy as np
import time


URDF_PATH = "../robot_description/urdf/robot/wam_4dof.urdf"



def ikin(model, data, frame_id, x_des, q0=None):
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

        J = pin.computeFrameJacobian(model, data, q, frame_id)[:3, :]

        # q -= alpha J^T(J J^T)^(-1) err
        q -= alpha * J.T @ np.linalg.solve(J@J.T+reg*np.eye(3), dx)

    else:
        raise ValueError(f'Could not reach position {x_des}.')

    return q



def main():

    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    frame_id = model.getFrameId('tool')

    x_des = np.array([0.3, 0, 2])


    t0 = time.time()
    for _ in range(1000):
        q = ikin(model, data, frame_id, x_des)
    T = time.time() - t0

    print(f'Inverse kinematics take {T:.2f}ms to compute.')


if __name__ == '__main__':
    main()


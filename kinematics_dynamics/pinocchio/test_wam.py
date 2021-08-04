"""
    mail@kaiploeger.net
"""

import numpy as np
import pinocchio as pin

urdf_path = "../robot_description/urdf/robot/wam_4dof.urdf"


def main():

    # load model
    model = pin.buildModelFromUrdf(urdf_path)


    # create data structure that pinocchio stores information in
    data = model.createData()


    # random robot config
    q = pin.randomConfiguration(model)
    dq = np.random.rand(model.nv)
    ddq = np.random.rand(model.nv)


    # get Ids of interesting frames
    idx_tool = model.getFrameId("tool")


    # -- kinematics --
    print('-- KINEMATICS --\n')

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)              # updates all frames...
    pin.updateFramePlacement(model, data, idx_tool)     # ...or only one

    # forward kinematics
    T = data.oMf[idx_tool]  # oMf = frame translations wrt origin
    pos = T.translation
    rot = T.rotation

    print(f'fkin end effector position:\n{pos}\n')
    print(f'fkin end effector rotation:\n{rot}\n')


    # Jacobian
    J = pin.computeFrameJacobian(model, data, q, idx_tool)
    J_pos = J[:3, :]
    J_rot = J[3:, :]

    print(f'end effector Jacobian:\n{J_pos}\n')


    # pseudo-inverse of Jacobian
    J_pinv = J_pos.T @ np.linalg.inv(J_pos@J_pos.T)

    print(f'pseudo-inverse Jacobian:\n{J_pinv}\n')


    # time derivative of the Jacobian...
    # ...can not be computed like this
    dJ = pin.computeJointJacobiansTimeVariation(model, data, q, dq)

    # ...instead use the fkin derivative (ddq=0!!) to compute ddx = dJ * dq
    pin.computeForwardKinematicsDerivatives(model, data, q, dq, np.zeros(model.nv))
    dJ_dq = pin.getFrameAcceleration(model, data, idx_tool).linear

    # ... check with finite differences
    delta_t = 0.0000001
    J_plus  = pin.computeFrameJacobian(model, data, q + delta_t*dq, idx_tool)
    J_minus = pin.computeFrameJacobian(model, data, q - delta_t*dq, idx_tool)
    dJ_fd = (J_plus - J_minus) / (2*delta_t)
    dJ_dq_fd = dJ_fd[:3,:] @ dq  # only linear part

    print('time derivative of Jacobian: (dJ/dt * dq/dt)')
    print(f'analytical:         {dJ_dq}')
    print(f'finite differences: {dJ_dq_fd}')


    # -- dynamics --
    print('\n-- DYNAMICS --\n')

    M = pin.crba(model, data, q)  # TODO: there are other algs implemented for M??
    print(f'mass matrix M:\n{M}\n')

    C = pin.computeCoriolisMatrix(model, data, q, dq)
    print(f'coriolis matrix C:\n{C}\n')

    g = pin.computeGeneralizedGravity(model, data, q)
    print(f'gravity vector g:\n{g}\n')

    b = pin.rnea(model, data, q, dq, np.zeros(model.nv))
    print(f'dynamic drift b = C(q,dq)*dq + g(q)\n{b}\n')





if __name__ == '__main__':
    main()


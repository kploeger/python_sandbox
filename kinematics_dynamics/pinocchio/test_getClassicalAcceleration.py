"""
    mail@kaiploeger.net
"""

import pinocchio as pin
import numpy as np


q = np.array([0, 0])
dq = np.array([0, 1])
ddq = np.array([0, 0])

ref_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED


def main():
    print('q:  ', q)
    print('dq: ', dq)
    print('ddq:', ddq)
    model = pin.buildModelFromUrdf('robot.urdf')
    data = model.createData()

    pin.forwardKinematics(model, data, q, dq, ddq)
    pin.updateFramePlacements(model, data)

    print('joint frame accelerations:\n')
    joint_names = ['fix_world', 'joint1', 'joint2', 'ee']
    for joint in joint_names:
        print(f'ddx {joint}:')
        print(pin.getClassicalAcceleration(model, data, model.getJointId(joint), ref_frame))

    print('LINK FRAME ACCELERATIONS:\n')
    link_names = ['world', 'robot_base', 'link1', 'link2', 'tool']
    for link in link_names:
        print(f'ddx {link}:')
        print(pin.getFrameClassicalAcceleration(model, data, model.getFrameId(link), ref_frame))


if __name__ == '__main__':
    main()


"""
    mail@kaiploeger.net
"""

import mujoco_py
import numpy as np
import pinocchio as pin


# URDF_PATH = "robot_description/urdf/robot/wam_4dof.urdf"
URDF_PATH = "wam4dof/wam4dof.urdf"
XML_PATH = "robot_description/mujoco/4dof_1arm_1balls.xml"


def main():

    # pinocchio model
    pin_model = pin.buildModelFromUrdf(URDF_PATH)
    data = pin_model.createData()
    # idx_tool = pin_model.getFrameId("tool_tip")
    # idx_base = pin_model.getFrameId("wam_base")
    idx_tool = pin_model.getFrameId("tool_plate")
    idx_base = pin_model.getFrameId("base_link")

    # mujoco sim
    mj_model = mujoco_py.load_model_from_path(XML_PATH)
    mj_sim = mujoco_py.MjSim(mj_model, nsubsteps=1)


    # test forward kinematics
    q_test = np.array([0, 0.1, 0, 0])
    q_test = np.random.normal(0, 1, 4)
    dq_test = np.array([0, 0, 0, 0])

    print('q: ', q_test)

    mj_sim.data.qpos[:4] = q_test
    mj_sim.data.qvel[:4] = dq_test
    mj_sim.forward()
    # x_mj = mj_sim.data.get_site_xpos("wam/sites/tool")
    x_mj = mj_sim.data.get_body_xpos("wam/links/tool_base_w_plate")

    pin.forwardKinematics(pin_model, data, q_test)
    # pin.updateFramePlacements(model, data)              # updates all frames...
    pin.updateFramePlacement(pin_model, data, idx_tool)     # ...or only one
    # pin.updateFramePlacement(model, data, idx_base)     # ...or only one
    T = data.oMf[idx_tool]
    x_pin = T.translation


    print('xml: ', x_mj)
    print('urdf:', x_pin)
    print('dif: ', x_mj - x_pin)

    R = np.array([[0, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 1]])

    # test jacobians
    jac_mj = np.zeros(3 * mj_sim.model.nv)
    # mj_sim.data.get_site_jacp("wam/sites/tool", jacp=jac_mj)
    mj_sim.data.get_body_jacp("wam/links/tool_base_w_plate", jacp=jac_mj)
    jac_mj = np.array(jac_mj).reshape((3, mj_sim.model.nv))[:, :4]

    jac_local = pin.computeFrameJacobian(pin_model, data, q_test, idx_tool, pin.LOCAL)[:3]
    jac_pin = T.rotation @ jac_local

    # np.set_printoptions(precision=3)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    print('\nJ_mj:\n', jac_mj)
    print('J_pin:\n', jac_pin)
    print('dif: ', jac_mj - jac_pin)




if __name__ == '__main__':
    main()


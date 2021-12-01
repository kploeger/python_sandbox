"""
    mail@kaiploeger.net
"""

import pinocchio as pin
from pinocchio import casadi as cpin

import casadi
from casadi import SX
import numpy as np


URDF_PATH = "../robot_description/urdf/robot/wam_4dof.urdf"


def main():
    # model = pin.buildSampleModelHumanoidRandom()
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()

    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    cq = SX.sym("q", cmodel.nq, 1)
    cdq = SX.sym("dq", cmodel.nv, 1)
    cddq = SX.sym("ddq", cmodel.nv, 1)

    cpin.forwardKinematics(cmodel, cdata, cq)
    cpin.updateFramePlacements(cmodel, cdata)


    print(len(cdata.oMf))

    frames = ["world",
              "fix_world","wam_footprint",
              "wam_basejoint", "wam_base",
              "wam_j1_joint", "j1",
              "wam_j2_joint", "j2",
              "wam_j3_joint", "j3",
              "wam_j4_joint", "j4",
              "wam_tool_fixed_joint", "tool",
              "wam_tool_col_fixed_joint", "tool_col"]

    for frame in frames:
        print(type(cdata.oMf[cmodel.getFrameId(frame)]))
        print(type(cdata.oMf[cmodel.getFrameId(frame)].translation))
        print(type(cdata.oMf[cmodel.getFrameId(frame)].rotation))

if __name__ == '__main__':
    main()


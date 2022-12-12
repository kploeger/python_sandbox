"""
    Testing mujoco bindings to migrate from mujoco_py

    Interactive tutorial:
    https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/tutorial.ipynb

    mail@kaiploeger.net
"""

import cv2

import numpy as np
import mujoco as mj
from mj_viewer import MujocoViewer


XML_MODEL_PATH = "./robot_description/mujoco/juggling_wam_4dof.xml"


def main():
    """
    doc
    """
    model = mj.MjModel.from_xml_path(XML_MODEL_PATH)
    data = mj.MjData(model)


    output = model.body("endeff_des")
    a = output
    print(type(output))


    geom = model.geom('throwing_tool')
    print(f'Properties of a geometry:\n{geom}')

    types = ['body',
             'jnt', 'joint',
             'geom',
             'site',
             'cam', 'camera',
             'light',
             'mesh',
             'skin',
             'hfield',
             'tex', 'texture',
             'mat', 'material',
             'pair',
             'exclude',
             'eq', 'equality',
             'tendon', 'ten',
             'actuator',
             'sensor',
             'numeric',
             'text',
             'tuple',
             'key', 'keyframe']

    print(f'\nOther available types to get info on:\n{types}')

    # viewer = MujocoViewer(model, data)

    # while data.time < 3:
        # mj.mj_step(model, data)
        # viewer.render()


    # changing colors does not seem to work for viewer
    # model.geom('throwing_tool').rgba[:3] = np.random.rand(3)

    # while data.time < 3:
        # mj.mj_step(model, data)
        # viewer.render()

    # viewer.close()


    # record a video

    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)

    # Simulate and display video.

    renderer = mj.Renderer(model, height=1200, width=1600)
    frames = []
    mj.mj_resetData(model, data)  # Reset state and time.
    # while data.time < duration:
    while False:
      mj.mj_step(model, data)
      if len(frames) < data.time * framerate:
        renderer.update_scene(data)
        pixels = renderer.render().copy()
        frames.append(pixels)

    # height, width, layers = frames[0].shape
    # size = (width,height)

    # fourcc = cv2.VideoWriter_fourcc('m','p','4','r') # FourCC is a 4-byte code used to specify the video codec.
    # video = cv2.VideoWriter('test.mp4', fourcc, float(framerate), (frames[0].shape[0], frames[0].shape[1]))

    # video = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    # print(frames[0].shape)

    # for frame in frames:
    #     video.write(frame)

    # video.release()

if __name__ == '__main__':
    main()

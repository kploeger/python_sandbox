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
from mujoco import mjtObj



XML_MODEL_PATH = "./robot_description/mujoco/juggling_wam_4dof.xml"


def main():

    # ---------- initialize environment ----------
    print('\n# ---------- initialize environment ----------')

    model = mj.MjModel.from_xml_path(XML_MODEL_PATH)
    data = mj.MjData(model)


    # ---------- access things ----------
    print('\n# ---------- access things ----------')

    # names and ids
    tool_id= mj.mj_name2id(model, mjtObj.mjOBJ_GEOM, 'throwing_tool')
    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, 23)
    print(f'name2id: throwing_tool->{tool_id}')
    print(f'id2name: 23->{name}')

    # accessing objects
    geom_from_name = model.geom(name)
    geom_from_id = model.geom(tool_id)
    print(f'\nmodel.geom(name)==model.geom(id): {geom_from_id==geom_from_name}')

    # available object types
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
             'tuple'
             'key', 'keyframe']
    print(f'\nOther available types to get info on:\n{types}')


    # ---------- Visualization ----------
    print('\n# ---------- Visualization ----------')

    # cameras have some interesting settings
    def make_camera(track_body_name=''):
        cam = mj.MjvCamera()
        cam.lookat[2] = 1.2     # where to look at
        cam.distance = 2.5      # how far to stay away
        cam.elevation = -25     # up down angle
        cam.azimuth = 135       # left right angle

        # keep camera pointed at the origin of given body
        if track_body_name:
            cam.type = mj.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, track_body_name)
        return cam

    # real time viewer to render camera
    def make_viewer(model, data, track_body_name=''):
        viewer = MujocoViewer(model, data)
        # MujocoViewer() initializes an MjvCamera, but it can be overwritten
        viewer.cam = make_camera(track_body_name=track_body_name)
        return viewer

    viewer = make_viewer(model, data)
    while data.time < 2:
        mj.mj_step(model, data)
        viewer.render()


    # change properties on the fly
    model.opt.gravity = np.array([0, 9.81, 0]) # to the side
    geom_mdl = model.geom('throwing_tool')
    geom_mdl.rgba[:3] = np.random.rand(3)
    while data.time < 4:
        mj.mj_step(model, data)
        viewer.render()

    # reopen viewer
    viewer.close(); viewer = None # garbage collect after close!
    viewer = make_viewer(model, data, track_body_name="links/tool_base_w_plate")
    while data.time < 6:
        mj.mj_step(model, data)
        viewer.render()

    # apply control:
    Kp = np.diag([100, 100, 100, 100])
    Kd = np.diag([10, 10, 10, 10])
    while data.time < 8:
        data.ctrl = - Kp @ data.qpos - Kd @ data.qvel
        mj.mj_step(model, data)
        viewer.render()
    viewer.close(); viewer = None
    data.ctrl = np.zeros(model.nq) # is not reset in mj_step()


    # ---------- Video Recording ----------
    print('\n# ---------- Video Recording ----------')

    # mj.mj_resetData(model, data)  # Reset state and time.

    # redering
    duration = 2  # (seconds)
    framerate = 60  # (Hz)
    resolution = [2560, 1440]

    renderer = mj.Renderer(model,  width=resolution[0], height=resolution[1])
    option = mj.MjvOption()
    option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
    option.flags[mj.mjtVisFlag.mjVIS_COM] = True
    option.frame = mj.mjtFrame.mjFRAME_GEOM
    cam = make_camera()

    t0 = data.time
    frames = []
    while data.time - t0 < duration:
        mj.mj_step(model, data)
        if len(frames) < (data.time - t0) * framerate:
            renderer.update_scene(data, scene_option=option, camera=cam)
            frames.append(renderer.render().copy())

    # writing
    codecs = ['mp4v', 'DIVX']
    formats = ['mp4', 'avi']

    for codec, format_ in zip(codecs, formats):
        print(f'writing {codec} {format_}')
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(f'mujoco_test_{codec}.{format_}',fourcc , framerate, resolution)
        for frame in frames:
            video.write(np.flip(frame, axis=2))
        video.release()


if __name__ == '__main__':
    main()

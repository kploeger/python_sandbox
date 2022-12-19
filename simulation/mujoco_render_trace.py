"""
    Testing mujoco bindings to migrate from mujoco_py

    Interactive tutorial:
    https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/tutorial.ipynb

    mail@kaiploeger.net
"""

import cv2
import numpy as np
import mujoco as mj
from mujoco_utils import MujocoViewer
from mujoco import mjtObj



XML_MODEL_PATH = "./robot_description/mujoco/juggling_wam_4dof.xml"


def get_site_speed(model, data, site_name):
    """Returns the speed of a geom."""
    site_vel = np.zeros(6)
    site_type = mj.mjtObj.mjOBJ_SITE
    site_id = data.site(site_name).id
    mj.mj_objectVelocity(model, data, site_type, site_id, site_vel, 0)
    return np.linalg.norm(site_vel)


def modify_scene(scn, positions, speeds):
    """Draw position trace, speed modifies width and colors."""
    if len(positions) > 1:
        for i in range(len(positions)-1):
            rgba=np.array((np.clip(speeds[i]/10, 0, 1), np.clip(1-speeds[i]/10, 0, 1), .5, 1.))
            radius=.05/(1+np.sqrt(speeds[i]))
            if scn.ngeom < scn.maxgeom:
                scn.ngeom += 1  # increment ngeom
                mj.mjv_initGeom(scn.geoms[scn.ngeom-1], mj.mjtGeom.mjGEOM_CAPSULE,
                                np.zeros(3), np.zeros(3), np.zeros(9), rgba.astype(np.float32))
                mj.mjv_makeConnector(scn.geoms[scn.ngeom-1], mj.mjtGeom.mjGEOM_CAPSULE, radius,
                                     positions[i][0], positions[i][1], positions[i][2],
                                     positions[i+1][0], positions[i+1][1], positions[i+1][2])


def main():

    model = mj.MjModel.from_xml_path(XML_MODEL_PATH)
    data = mj.MjData(model)

    def make_camera():
        cam = mj.MjvCamera()
        cam.lookat[2] = 1.2
        cam.distance = 2.5
        cam.elevation = -25
        cam.azimuth = 135
        return cam

    def make_viewer(model, data):
        viewer = MujocoViewer(model, data)
        viewer.cam = make_camera()
        return viewer

    tool_x_hist = []
    tool_dx_hist = []

    print('\n# ---------- Interactive Viewer ----------')

    viewer = make_viewer(model, data)
    viewer_cb = lambda m, d, s: modify_scene(s, tool_x_hist, tool_dx_hist)
    viewer.set_pre_render_callback(viewer_cb)

    mj.mj_resetData(model, data)  # Reset state and time.
    mj.mj_forward(model, data)

    while data.time < 2:
        mj.mj_step(model, data)
        tool_x_hist.append(data.site_xpos[data.site("sites/tool").id].copy())
        tool_dx_hist.append(get_site_speed(model, data, "sites/tool"))
        viewer.render()
    viewer.close(); viewer = None


    print('\n# ---------- Video Recording ----------')

    mj.mj_resetData(model, data)  # Reset state and time.
    mj.mj_forward(model, data)

    # redering
    duration = 2  # (seconds)
    framerate = 60  # (Hz)
    resolution = [2560, 1440]

    renderer = mj.Renderer(model,  width=resolution[0], height=resolution[1])
    cam = make_camera()

    t0 = data.time
    frames = []
    tool_x_hist = []
    tool_dx_hist = []

    while data.time - t0 < duration:
        mj.mj_step(model, data)
        tool_x_hist.append(data.site_xpos[data.site("sites/tool").id].copy())
        tool_dx_hist.append(get_site_speed(model, data, "sites/tool"))
        if len(frames) < (data.time - t0) * framerate:
            renderer.update_scene(data, camera=cam)
            modify_scene(renderer.scene, tool_x_hist, tool_dx_hist)  # draw things!
            frames.append(renderer.render().copy())

    # writing
    codecs = ['mp4v', 'DIVX']
    formats = ['mp4', 'avi']

    for codec, format_ in zip(codecs, formats):
        print(f'writing {codec} {format_}')
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(f'mujoco_render_trace_{codec}.{format_}',fourcc , framerate, resolution)
        for frame in frames:
            video.write(np.flip(frame, axis=2))
        video.release()


if __name__ == '__main__':
    main()

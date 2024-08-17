"""
    Parameter estimation using Pinocchio. Loss on closed-loop tracking of a desired trajectory.

    mail@kaiploeger.net
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import pinocchio as pin
from scipy.optimize import minimize

URDF_PATH = (Path(__file__).parent / "test.urdf").resolve().as_posix()

DT = 0.01  # control / integration time step
T = 5.0     # trajectory duration

KP = np.diag([100, 100, 100, 100])
KD = np.diag([10, 10, 10, 10])


def get_desired_trajectory(num_dof, cycle_times, num_steps):
    t = np.linspace(0, num_steps * DT, num_steps+1)
    q = np.zeros((num_steps+1, num_dof))
    dq = np.zeros((num_steps+1, num_dof))
    ddq = np.zeros((num_steps+1, num_dof))

    for i in range(num_dof):
        q[:, i] = np.sin(2 * np.pi / cycle_times[i] * t)
        dq[:, i] = 2 * np.pi / cycle_times[i] * np.cos(2 * np.pi / cycle_times[i] * t)
        ddq[:, i] = -(2 * np.pi / cycle_times[i]) ** 2 * np.sin(2 * np.pi / cycle_times[i] * t)

    return t, q, dq, ddq


def track_trajectory(model, data, cycle_times, num_steps):
    assert len(cycle_times) == model.njoints - 1

    # generate desired trajectory
    _, q_des, dq_des, _ = get_desired_trajectory(model.nv, cycle_times, num_steps)

    q_traj = np.zeros((num_steps+1, model.nv))
    dq_traj = np.zeros((num_steps+1, model.nv))
    q_traj[0] = q_des[0]
    dq_traj[0] = dq_des[0]

    # reset initial state
    data.qpos = q_des[0]
    data.qvel = dq_des[0]

    for i in range(num_steps):
        # compute control input (PD controller)
        tau = KP @ (q_des[i] - data.qpos) + KD @ (dq_des[i] - data.qvel)

        # compute acceleration
        M = pin.crba(model, data, data.qpos)
        b = pin.nonLinearEffects(model, data, data.qpos, data.qvel)
        ddq = np.linalg.solve(M, tau - b)

        # integrate (symplectic Euler)
        data.qvel += ddq * DT
        data.qpos = pin.integrate(model, data.qpos, data.qvel * DT)

        # save data
        q_traj[i+1] = data.qpos
        dq_traj[i+1] = data.qvel

    return q_traj, dq_traj


def main():

    # load true model
    true_model = pin.buildModelFromUrdf(URDF_PATH)
    true_data = true_model.createData()

    cycle_times = [[4.1, 2.8, 1.5, 0.9],
                   [3.2, 2.1, 1.3, 4.0],
                   [2.3, 1.3, 4.1, 3.2],
                   [1.3, 4.2, 3.1, 2.2]]
    
    num_steps = int(T / DT)
    num_dof = true_model.njoints - 1

    times, q_des, dq_des, ddq_des = get_desired_trajectory(num_dof, cycle_times[0], num_steps)
    q, dq = track_trajectory(true_model, true_data, cycle_times[0], num_steps)

    # generate true trajectories
    q_trues, dq_trues = track_trajectory(true_model, true_data, cycle_times[0], num_steps)
    for cycle_time in cycle_times[1:]:
        q_true, dq_true = track_trajectory(true_model, true_data, cycle_time, num_steps)
        q_trues = np.vstack([q_trues, q_true])
        dq_trues = np.vstack([dq_trues, dq_true])

    # generate an incorrect model
    guessed_model = true_model.copy()
    guessed_data = guessed_model.createData()

    # guess of masses:
    masses = np.array([0.8 for _ in range(2)])
    print(f"Initial masses: {masses}\n")

    # define problem
    def loss(masses):
        if np.any(masses < 0):  # should be handled more informative but works
            return 1e6

        # set masses
        guessed_model.inertias[3].mass = masses[0]
        guessed_model.inertias[4].mass = masses[0]

        # track trajectory
        qs, dqs = track_trajectory(guessed_model, guessed_data, cycle_times[0], num_steps)
        for cycle_time in cycle_times[1:]:
            q, dq = track_trajectory(guessed_model, guessed_data, cycle_time, num_steps)
            qs = np.vstack([qs, q])
            dqs = np.vstack([dqs, dq])

        # evaluate error
        q_error = np.linalg.norm(qs - q_trues, axis=1).mean()
        dq_error = np.linalg.norm(dqs - dq_trues, axis=1).mean()
        total_error = q_error + dq_error

        print(f"masses: {masses} | error: {total_error}")
        return total_error

    # optimize masses
    start_time = time.time()    
    result = minimize(loss, masses, method='Nelder-Mead', options={'disp': True})
    duration = time.time() - start_time
    print(f"Optimization took {duration:.3f} seconds")

    print(f"\nOptimized masses: {result.x}")
    print(f"True masses:      {[true_model.inertias[3].mass, true_model.inertias[4].mass]}")

    # plot results
    times, q_des, dq_des, ddq_des = get_desired_trajectory(num_dof, cycle_times[0], num_steps)
    q_true, dq_true = track_trajectory(true_model, true_data, cycle_times[0], num_steps)
    q_guessed, dq_guessed = track_trajectory(guessed_model, guessed_data, cycle_times[0], num_steps)


    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    for i in [0, 1, 2, 3]:
        # axs[0].plot(times, q_des[:, i], label=f"q_des{i}")
        axs[0].plot(times, q_true[:, i], label=f"q_true{i}")
        axs[0].plot(times, q_guessed[:, i], label=f"q_guessed{i}")
        # axs[1].plot(times, dq_des[:, i], label=f"dq_des{i}")
        axs[1].plot(times, dq_true[:, i], label=f"dq_true{i}")
        axs[1].plot(times, dq_guessed[:, i], label=f"dq_guessed{i}")
    axs[0].legend()
    axs[1].legend()
    plt.show()


if __name__ == '__main__':
    main()

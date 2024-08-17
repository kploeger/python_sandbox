"""
    Simple example of parameter estimation using Pinocchio.

    mail@kaiploeger.net
"""

import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from scipy.optimize import minimize

URDF_PATH = (Path(__file__).parent / "test.urdf").resolve().as_posix()


def main():

    # load true model
    true_model = pin.buildModelFromUrdf(URDF_PATH)
    true_data = true_model.createData()

    # generate training data
    n_samples = 100
    qs = np.array([pin.randomConfiguration(true_model) for _ in range(n_samples)])
    dqs = np.array([np.random.rand(true_model.nv) for _ in range(n_samples)]) * 10
    ddqs = np.array([np.random.rand(true_model.nv) for _ in range(n_samples)]) * 100
    taus = np.array([pin.rnea(true_model, true_data, q, dq, dds) for q, dq, dds in zip(qs, dqs, ddqs)])

    # generate an incorrect model
    guessed_model = true_model.copy()
    guessed_data = guessed_model.createData()

    # guess of masses:
    masses = np.array([0.5 for _ in range(true_model.njoints - 1)])
    print(f"Initial masses: {masses}\n")

    # optimize masses
    def loss(masses):
        for i in range(1, true_model.njoints):
            guessed_model.inertias[i].mass = masses[i - 1]
        taus_guessed = np.array([pin.rnea(guessed_model, guessed_data, q, dq, dds) for q, dq, dds in zip(qs, dqs, ddqs)])
        return np.linalg.norm(taus - taus_guessed, axis=1).mean()

    start_time = time.time()    
    result = minimize(loss, masses, method='Nelder-Mead', options={'disp': True})
    duration = time.time() - start_time
    print(f"Optimization took {duration:.3f} seconds")

    # print results
    print(f"\nOptimized masses: {result.x}")
    print(f"True masses:      {[true_model.inertias[i].mass for i in range(1, true_model.njoints)]}")
    print("(first two masses do not affect the dynamics)")



if __name__ == '__main__':
    main()

"""Verify: does STEPS_PER_FRAME affect physics?
Run 2000 steps (2s) of free-fall with controller OFF,
batching 1, 16, and 24 steps per frame. Compare final state."""
import os, sys
_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"): os.add_dll_directory(_lib)

import pybullet as p
import numpy as np

# Minimal V10 setup (headless)
M1, M2, L1, L2 = 30.0, 40.0, 0.85, 0.80
BD = 0.25
GRAV = 9.81
DT = 0.001
POLE_H = 4.5; HALF_POLE = 2.0; FOOT_Z = 3.0

def capsule_quat(d):
    d = np.array(d, dtype=float, copy=True)
    dl = np.linalg.norm(d); d /= dl
    ax = np.cross([0,0,1], d)
    s = np.linalg.norm(ax)
    if s < 1e-8: return [0,0,0,1]
    ax /= s; c = d[2]
    ha = np.arctan2(s,c)/2
    return [*(ax*np.sin(ha)), np.cos(ha)]

def box_inertia(m, w, d, h):
    return [m*(d*d+h*h)/12, m*(w*w+h*h)/12, m*(w*w+d*d)/12]

ra_end = np.array([0, -1.875, -1.5])
rb_end = np.array([0, -1.875,  1.5])

def run_sim(steps_per_frame, total_steps=2000):
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -GRAV)
    p.setTimeStep(DT)

    robot = p.createMultiBody(
        baseMass=0, basePosition=[0, HALF_POLE, POLE_H],
        linkMasses=[0.001, M1, M2],
        linkCollisionShapeIndices=[-1]*3,
        linkVisualShapeIndices=[-1]*3,
        linkPositions=[[0,0,0], ra_end.tolist(), [0,-BD/2,L1]],
        linkOrientations=[[0,0,0,1]]*3,
        linkInertialFramePositions=[[0,0,0],[0,-BD/2,L1/2],[0,0,L2/2]],
        linkInertialFrameOrientations=[[0,0,0,1]]*3,
        linkParentIndices=[0,1,2],
        linkJointTypes=[p.JOINT_REVOLUTE]*3,
        linkJointAxis=[[0,1,0]]*3,
    )
    for j in range(3):
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)
    p.changeDynamics(robot, 0, jointDamping=2.0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot, 1, mass=M1, localInertiaDiagonal=box_inertia(M1,0.18,BD,L1),
                     jointDamping=1.0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot, 2, mass=M2, localInertiaDiagonal=box_inertia(M2,0.30,BD,L2),
                     jointDamping=0.5, linearDamping=0, angularDamping=0)

    # Initial pose: theta0 = 0.15
    p.resetJointState(robot, 0, 0, 0)
    p.resetJointState(robot, 1, 0, 0)
    p.resetJointState(robot, 2, 0.15, 0)

    # Run
    samples = []
    n = 0
    while n < total_steps:
        batch = min(steps_per_frame, total_steps - n)
        for _ in range(batch):
            p.stepSimulation()
            n += 1
        # Record at specific times
        if n in [500, 1000, 1500, 2000]:
            phi = p.getJointState(robot, 0)[0]
            ls1 = p.getLinkState(robot, 1, computeForwardKinematics=True)
            ls2 = p.getLinkState(robot, 2, computeForwardKinematics=True)
            alpha = p.getEulerFromQuaternion(ls1[5])[1]
            theta = p.getEulerFromQuaternion(ls2[5])[1]
            samples.append((n*DT, np.degrees(phi), np.degrees(alpha), np.degrees(theta)))

    p.disconnect()
    return samples

print("=== Verifying STEPS_PER_FRAME independence ===\n")
for spf in [1, 16, 24, 100]:
    results = run_sim(spf)
    print(f"STEPS_PER_FRAME = {spf:3d}:")
    for t, phi, alpha, theta in results:
        print(f"  t={t:.1f}s  phi={phi:+8.3f}  alpha={alpha:+8.3f}  theta={theta:+8.3f}")
    print()

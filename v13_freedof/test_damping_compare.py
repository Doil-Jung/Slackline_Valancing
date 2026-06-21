"""
Damping comparison test - headless (no GUI).
Runs 3 damping settings with identical initial conditions and LQR K.
Plots phi/alpha/theta over time.
"""
import sys, os
_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib) and _lib not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib)

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

# ---- V10 parameters (same as v13_pybullet.py) ----
M1=30.0; M2=40.0; L1=0.85; L2=0.80
LOWER_W=0.18; UPPER_W=0.30; BD=0.25
GRAV=9.81; DT=0.001; TAU_MAX=1500.0
L_POLE=4.0; POLE_H=4.5; R_TARGET=1.5
HALF_POLE=L_POLE/2; ROPE_HORIZ=HALF_POLE-BD/2; ROPE_VERT=R_TARGET
FOOT_Z=POLE_H-ROPE_VERT
LC1=L1/2; LC2=L2/2; I1=M1*L1**2/12; I2=M2*L2**2/12
ROPE_MASS=1.0; ALPHA0=0.0; THETA0=0.15

# Fixed K from V10 (with phi sign flip)
K_LQR = np.array([6263.3, -5736.9, -2081.0, 1272.7, -1063.3, -562.2])

def capsule_quat(d):
    d = np.array(d, dtype=float, copy=True); dl = np.linalg.norm(d)
    if dl < 1e-8: return [0,0,0,1]
    d /= dl; z = np.array([0,0,1.])
    dot = np.clip(z @ d, -1, 1)
    if dot > .9999: return [0,0,0,1]
    if dot < -.9999: return [1,0,0,0]
    ax = np.cross(z, d); ax /= np.linalg.norm(ax)
    s = np.sin(np.arccos(dot)/2)
    return [ax[0]*s, ax[1]*s, ax[2]*s, np.cos(np.arccos(dot)/2)]

def box_inertia(m, w, d, h):
    return [m*(d**2+h**2)/12, m*(w**2+h**2)/12, m*(w**2+d**2)/12]

def run_sim(b_phi, b_alpha, b_theta, duration=5.0, ctrl_on=True):
    """Run headless sim, return time series."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -GRAV)
    p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)
    p.loadURDF("plane.urdf")

    ra_end = np.array([0, -ROPE_HORIZ, -ROPE_VERT])
    rb_end = np.array([0, -ROPE_HORIZ,  ROPE_VERT])

    robot_id = p.createMultiBody(
        baseMass=0, basePosition=[0, HALF_POLE, POLE_H],
        linkMasses=[ROPE_MASS, M1, M2, ROPE_MASS],
        linkCollisionShapeIndices=[-1]*4,
        linkVisualShapeIndices=[-1]*4,
        linkPositions=[[0,0,0], ra_end.tolist(), [0,-BD/2,L1], [0,-BD,0]],
        linkOrientations=[[0,0,0,1]]*4,
        linkInertialFramePositions=[[0,0,0],[0,-BD/2,L1/2],[0,0,L2/2],[0,0,0]],
        linkInertialFrameOrientations=[[0,0,0,1]]*4,
        linkParentIndices=[0, 1, 2, 2],
        linkJointTypes=[p.JOINT_SPHERICAL, p.JOINT_SPHERICAL, p.JOINT_REVOLUTE, p.JOINT_SPHERICAL],
        linkJointAxis=[[0,0,0],[0,0,0],[0,1,0],[0,0,0]],
    )
    ROPE_A, ANKLE, HIP, ROPE_B = 0, 1, 2, 3

    pole_b_pos = [0, -HALF_POLE, POLE_H]
    con = p.createConstraint(robot_id, ROPE_B, -1, -1, p.JOINT_POINT2POINT,
                              [0,0,0], rb_end.tolist(), pole_b_pos)
    p.changeConstraint(con, maxForce=50000, erp=0.8)

    p.changeDynamics(robot_id, ANKLE, mass=M1,
                     localInertiaDiagonal=box_inertia(M1, LOWER_W, BD, L1),
                     jointDamping=b_alpha, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, HIP, mass=M2,
                     localInertiaDiagonal=box_inertia(M2, UPPER_W, BD, L2),
                     jointDamping=b_theta, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, ROPE_A, jointDamping=b_phi,
                     localInertiaDiagonal=[0.1,0.1,0.1], linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, ROPE_B, jointDamping=b_phi,
                     localInertiaDiagonal=[0.1,0.1,0.1], linearDamping=0, angularDamping=0)

    p.setJointMotorControl2(robot_id, HIP, p.VELOCITY_CONTROL, force=0)
    for sj in [ROPE_A, ANKLE, ROPE_B]:
        p.setJointMotorControlMultiDof(robot_id, sj, p.POSITION_CONTROL,
                                       targetPosition=[0,0,0,1], force=[0,0,0])

    for i in range(-1, 4):
        for j in range(-1, 4):
            if i != j: p.setCollisionFilterPair(robot_id, robot_id, i, j, 0)
        p.setCollisionFilterPair(robot_id, 0, i, -1, 0)

    # Initial pose
    hip_rel0 = THETA0 - ALPHA0
    p.resetJointStateMultiDof(robot_id, ROPE_A, [0,0,0,1], [0,0,0])
    ha = ALPHA0 / 2
    qa = [0, np.sin(ha), 0, np.cos(ha)]
    p.resetJointStateMultiDof(robot_id, ANKLE, qa, [0,0,0])
    p.resetJointState(robot_id, HIP, hip_rel0, 0)
    p.resetJointStateMultiDof(robot_id, ROPE_B, [0,0,0,1], [0,0,0])

    # Run
    N = int(duration / DT)
    rec = {'t':[], 'phi':[], 'alpha':[], 'theta':[], 'tau':[]}

    for step in range(N):
        # State
        ra = p.getJointStateMultiDof(robot_id, ROPE_A)
        phi = p.getEulerFromQuaternion(ra[0])[1]
        phi_dot = ra[1][1]
        lo = p.getLinkState(robot_id, ANKLE, computeForwardKinematics=True, computeLinkVelocity=True)
        up = p.getLinkState(robot_id, HIP, computeForwardKinematics=True, computeLinkVelocity=True)
        alpha = p.getEulerFromQuaternion(lo[5])[1]
        theta = p.getEulerFromQuaternion(up[5])[1]
        alpha_dot = lo[7][1]
        theta_dot = up[7][1]
        x = np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])

        if np.any(np.isnan(x)):
            print(f"  NaN at t={step*DT:.3f}s")
            break

        tau = 0.0
        if ctrl_on:
            if abs(x[1]) < np.radians(60) and abs(x[2]) < np.radians(60):
                tau = -float(K_LQR @ x)
                tau = np.clip(tau, -TAU_MAX, TAU_MAX)
            p.setJointMotorControl2(robot_id, HIP, p.TORQUE_CONTROL, force=tau)

        p.stepSimulation()

        if step % 10 == 0:
            rec['t'].append(step*DT)
            rec['phi'].append(np.degrees(phi))
            rec['alpha'].append(np.degrees(alpha))
            rec['theta'].append(np.degrees(theta))
            rec['tau'].append(tau)

    p.disconnect()
    return {k: np.array(v) for k, v in rec.items()}

# ---- Run 3 cases ----
cases = [
    ("Damping=0",       0.0, 0.0, 0.0),
    ("V10 (2,1,0.5)",   2.0, 1.0, 0.5),
    ("High (5,3,1.5)",  5.0, 3.0, 1.5),
]

results = {}
for name, bp, ba, bt in cases:
    print(f"Running: {name} ...")
    results[name] = run_sim(bp, ba, bt, duration=5.0, ctrl_on=True)
    n = len(results[name]['t'])
    print(f"  -> {n} samples, final t={results[name]['t'][-1]:.2f}s")

# ---- Plot ----
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
colors = ['#e74c3c', '#3498db', '#2ecc71']

for idx, (name, _bp, _ba, _bt) in enumerate(cases):
    d = results[name]
    axes[0].plot(d['t'], d['phi'],   color=colors[idx], label=name, alpha=0.8)
    axes[1].plot(d['t'], d['alpha'], color=colors[idx], label=name, alpha=0.8)
    axes[2].plot(d['t'], d['theta'], color=colors[idx], label=name, alpha=0.8)
    axes[3].plot(d['t'], d['tau'],   color=colors[idx], label=name, alpha=0.8)

titles = ['phi (rope swing)', 'alpha (lower body)', 'theta (upper body)', 'tau (hip torque)']
ylabels = ['deg', 'deg', 'deg', 'Nm']
for ax, title, yl in zip(axes, titles, ylabels):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(yl)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)

axes[-1].set_xlabel('Time (s)')
fig.suptitle('V13 Free-DOF: Damping Comparison (Fixed K from V10)', fontsize=14, fontweight='bold')
plt.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'damping_compare.png')
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")
plt.show()

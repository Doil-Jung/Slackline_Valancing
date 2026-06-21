"""Compare V10 (analytical EOM + RK4) vs PyBullet free-fall trajectories."""
import numpy as np
import os, sys

# ===== V10 Analytical Model =====
M1, M2, L1, L2 = 30.0, 40.0, 0.85, 0.80
R = 1.5;  g = 9.81;  DT = 0.001
ell1 = L1/2;  ell2 = L2/2
I1 = M1*L1**2/12;  I2 = M2*L2**2/12
p1 = M1*ell1 + M2*L1
p2 = M2*ell2
p3 = M2*L1*ell2
Mtotal = M1 + M2
M11 = Mtotal * R**2
M22 = M1*ell1**2 + I1 + M2*L1**2
M33 = M2*ell2**2 + I2
b_phi, b_alpha, b_theta = 2.0, 1.0, 0.5

def v10_deriv(state, tau=0):
    phi, alpha, theta, pd, ad, td = state
    sinPA = np.sin(phi+alpha); cosPA = np.cos(phi+alpha)
    sinPT = np.sin(phi+theta); cosPT = np.cos(phi+theta)
    sinAT = np.sin(alpha-theta); cosAT = np.cos(alpha-theta)
    
    M = np.array([
        [M11,           R*p1*cosPA,    R*p2*cosPT],
        [R*p1*cosPA,    M22,           p3*cosAT],
        [R*p2*cosPT,    p3*cosAT,      M33]
    ])
    f = np.array([
        R*p1*sinPA*ad**2 + R*p2*sinPT*td**2 - Mtotal*g*R*np.sin(phi) - b_phi*pd,
        R*p1*sinPA*pd**2 - p3*sinAT*td**2 + p1*g*np.sin(alpha) - tau - b_alpha*ad,
        R*p2*sinPT*pd**2 + p3*sinAT*ad**2 + p2*g*np.sin(theta) + tau - b_theta*td,
    ])
    accel = np.linalg.solve(M, f)
    return np.array([pd, ad, td, accel[0], accel[1], accel[2]])

def v10_rk4(state, tau, dt):
    k1 = v10_deriv(state, tau)
    k2 = v10_deriv(state + 0.5*dt*k1, tau)
    k3 = v10_deriv(state + 0.5*dt*k2, tau)
    k4 = v10_deriv(state + dt*k3, tau)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# ===== PyBullet Model =====
_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH","")
    if hasattr(os, "add_dll_directory"): os.add_dll_directory(_lib)
import pybullet as p

def box_inertia(m, w, d, h):
    return [m*(d*d+h*h)/12, m*(w*w+h*h)/12, m*(w*w+d*d)/12]

BD = 0.25
ra_end = np.array([0, -1.875, -1.5])

def run_pybullet(init_state, total_steps=2000):
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -g)
    p.setTimeStep(DT)
    robot = p.createMultiBody(
        baseMass=0, basePosition=[0, 2, 4.5],
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
    p.changeDynamics(robot, 0, jointDamping=b_phi, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot, 1, mass=M1, localInertiaDiagonal=box_inertia(M1,0.18,BD,L1),
                     jointDamping=b_alpha, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot, 2, mass=M2, localInertiaDiagonal=box_inertia(M2,0.30,BD,L2),
                     jointDamping=b_theta, linearDamping=0, angularDamping=0)
    
    phi0, alpha0, theta0 = init_state[:3]
    hip_rel = theta0 - alpha0
    p.resetJointState(robot, 0, phi0, init_state[3])
    p.resetJointState(robot, 1, alpha0, init_state[4])
    p.resetJointState(robot, 2, hip_rel, init_state[5] - init_state[4])
    
    results = []
    for n in range(total_steps):
        p.stepSimulation()
        phi = p.getJointState(robot, 0)[0]
        phi_dot = p.getJointState(robot, 0)[1]
        ls1 = p.getLinkState(robot, 1, computeForwardKinematics=True, computeLinkVelocity=True)
        ls2 = p.getLinkState(robot, 2, computeForwardKinematics=True, computeLinkVelocity=True)
        alpha = p.getEulerFromQuaternion(ls1[5])[1]
        theta = p.getEulerFromQuaternion(ls2[5])[1]
        alpha_dot = ls1[7][1]
        theta_dot = ls2[7][1]
        if (n+1) % 100 == 0:
            results.append((n+1, phi, alpha, theta, phi_dot, alpha_dot, theta_dot))
    p.disconnect()
    return results

# ===== Run comparison =====
init = np.array([0.0, 0.0, 0.15, 0.0, 0.0, 0.0])  # theta0=0.15 rad

# V10
print("=== Free-fall comparison (tau=0, theta0=8.6deg) ===\n")
print(f"{'t':>6s}  {'V10 phi':>8s} {'PB phi':>8s}  {'V10 a':>8s} {'PB a':>8s}  {'V10 th':>8s} {'PB th':>8s}")
print("-"*70)

state_v10 = init.copy()
pb_results = run_pybullet(init, 2000)

pb_idx = 0
for n in range(1, 2001):
    state_v10 = v10_rk4(state_v10, 0, DT)
    if n % 100 == 0:
        t = n * DT
        pb = pb_results[pb_idx]; pb_idx += 1
        print(f"t={t:.1f}s  "
              f"{np.degrees(state_v10[0]):+8.2f} {np.degrees(pb[1]):+8.2f}  "
              f"{np.degrees(state_v10[1]):+8.2f} {np.degrees(pb[2]):+8.2f}  "
              f"{np.degrees(state_v10[2]):+8.2f} {np.degrees(pb[3]):+8.2f}")

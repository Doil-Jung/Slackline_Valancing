"""
V13 Two-Body Architecture
=========================
Two separate multibodies connected by P2P between moving bodies.
This allows phi to swing freely (P2P no longer anchored to fixed world).

Multibody 1 (robot): pole_a → universal(Y+X) → rope_a → spherical → lower_body → revolute_Y → upper_body
Multibody 2 (rope_b): pole_b → universal(Y+X) → rope_b
Constraint: P2P between rope_b end ↔ lower_body ankle_b (both moving)
"""
import sys
import os
import time
import numpy as np

# Conda DLL fix
_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib) and _lib not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib)

import pybullet as p
import pybullet_data

# ============================================================
#  V10 Parameters (hardcoded, matching v5/params.js)
# ============================================================
M1 = 30.0;  M2 = 40.0          # mass (kg)
L1 = 0.85;  L2 = 0.80          # length (m)
LOWER_W = 0.18; UPPER_W = 0.30 # width (m)
BD = 0.50                      # body depth = foot spacing (m) — doubled to resist lateral falling

# Damping: spherical joints must have jointDamping=0 (PyBullet locks them otherwise)
# Only hip (revolute, STS3215 servo with gear reduction) gets real damping
B_PHI_LQR = 0.0; B_ALPHA_LQR = 0.0; B_THETA_LQR = 3.0
B_PHI = 0.0; B_ALPHA = 0.0; B_THETA = 3.0
TAU_MAX = 1500.0               # max hip torque (N·m)
GRAV = 9.81

# Rope geometry → R_rope = rope_vert = 1.5 (matching V10 R)
L_POLE = 4.0; POLE_H = 4.5; R_TARGET = 1.5
HALF_POLE = L_POLE / 2.0
ROPE_HORIZ = HALF_POLE - BD / 2.0          # 1.875
ROPE_VERT = R_TARGET                        # 1.5
HALF_ROPE = np.sqrt(ROPE_HORIZ**2 + ROPE_VERT**2)  # ~2.401
L0 = 2.0 * HALF_ROPE                       # ~4.803
FOOT_Z = POLE_H - ROPE_VERT                # 3.0

# Derived (V10 formulas)
LC1 = L1 / 2; LC2 = L2 / 2
I1 = M1 * L1**2 / 12; I2 = M2 * L2**2 / 12

# Simulation
DT = 0.001          # match V10
ALPHA0 = 0.0; THETA0 = 0.15

# LQR weights (V10 default)
Q_DIAG = [1, 50, 50, 0.1, 5, 5]
R_COST = 0.001  # V10 default

ROPE_R = 0.012; ROPE_MASS = 0.01  # Near-zero mass; NaN prevented by explicit localInertiaDiagonal


# ============================================================
#  LQR (V10 formulation, scipy DARE)
# ============================================================
def compute_lqr():
    """6-state discrete LQR matching V10's linearization exactly."""
    from scipy.linalg import solve_discrete_are, expm

    R = R_TARGET
    p1 = M1 * LC1 + M2 * L1
    p2 = M2 * LC2
    p3 = M2 * L1 * LC2
    Mt = M1 + M2

    M_eq = np.array([
        [Mt*R**2,    R*p1,                     R*p2],
        [R*p1,       M1*LC1**2 + I1 + M2*L1**2, p3],
        [R*p2,       p3,                        M2*LC2**2 + I2],
    ])
    # Linearization uses V10 convention internally
    G_mat = np.array([
        [-Mt*GRAV*R, 0, 0],
        [0, p1*GRAV, 0],
        [0, 0, p2*GRAV],
    ])
    D_mat = np.diag([-B_PHI_LQR, -B_ALPHA_LQR, -B_THETA_LQR])
    E_mat = np.array([[0], [-1], [1]])

    Mi = np.linalg.inv(M_eq)
    n = 6
    Ac = np.zeros((n, n)); Bc = np.zeros((n, 1))
    Ac[:3, 3:] = np.eye(3)
    Ac[3:, :3] = Mi @ G_mat
    Ac[3:, 3:] = Mi @ D_mat
    Bc[3:, :] = Mi @ E_mat

    # Exact discretization via matrix exponential
    Z = np.zeros((n+1, n+1))
    Z[:n, :n] = Ac * DT
    Z[:n, n:] = Bc * DT
    eZ = expm(Z)
    Ad = eZ[:n, :n]; Bd = eZ[:n, n:n+1]

    Q = np.diag(Q_DIAG); Rc = np.array([[R_COST]])
    P = solve_discrete_are(Ad, Bd, Q, Rc)
    K = np.linalg.inv(Rc + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

    # PyBullet phi convention is opposite to V10 (right-hand rule).
    # Coordinate transform S=diag(-1,1,1) → K_pb = K_v10 * diag(-1,1,1,-1,1,1)
    K[0, 0] *= -1  # phi
    K[0, 3] *= -1  # phi_dot

    # Diagnostics
    eigs_o = sorted(abs(e) for e in np.linalg.eigvals(Ad))
    eigs_c = sorted(abs(e) for e in np.linalg.eigvals(Ad - Bd @ K))
    print(f"[LQR] K = [{', '.join(f'{k:.1f}' for k in K[0])}]")
    print(f"[LQR] Open  |eig|: {[f'{e:.4f}' for e in eigs_o]}")
    print(f"[LQR] Close |eig|: {[f'{e:.4f}' for e in eigs_c]}")
    x10 = np.array([0, 0, np.radians(10), 0, 0, 0])
    print(f"[LQR] tau@10deg = {-float(K[0] @ x10):.1f} Nm")
    return K[0]


# ============================================================
#  Helpers
# ============================================================
def capsule_quat(d):
    """Quaternion aligning Z-axis to direction d."""
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


# ============================================================
#  Main Simulation
# ============================================================
def main():
    global R_COST
    print("=" * 60)
    print("  V10 -> PyBullet Verification")
    print(f"  m1={M1} m2={M2}  L1={L1} L2={L2}  R={R_TARGET}")
    print(f"  L_pole={L_POLE} L0={L0:.3f} foot_z={FOOT_Z:.1f}")
    print(f"  Q={Q_DIAG}  R={R_COST}  dt={DT}")
    print("=" * 60)

    K_lqr = compute_lqr()

    # --- PyBullet setup ---
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -GRAV)
    p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)
    p.setRealTimeSimulation(0)  # prevent auto-stepping in GUI mode
    p.resetDebugVisualizerCamera(4.0, 45, -25, [0, 0, FOOT_Z])
    p.loadURDF("plane.urdf")

    # Poles (visual only)
    pv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=POLE_H,
                             rgbaColor=[.5,.5,.5,1])
    sv = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[.9,.9,.9,1])
    for sy in [1, -1]:
        p.createMultiBody(0, baseVisualShapeIndex=pv,
                          basePosition=[0, sy*HALF_POLE, POLE_H/2])
        p.createMultiBody(0, baseVisualShapeIndex=sv,
                          basePosition=[0, sy*HALF_POLE, POLE_H])
    # Store pole_b for constraint
    pole_b_id = p.createMultiBody(0, baseVisualShapeIndex=-1,
                                   basePosition=[0, -HALF_POLE, POLE_H])

    # --- Rope vectors ---
    ra_end = np.array([0, -ROPE_HORIZ, -ROPE_VERT])  # pole_a → ankle_a
    rb_end = np.array([0, -ROPE_HORIZ,  ROPE_VERT])  # ankle_b → pole_b
    rb_vec = np.array([0,  ROPE_HORIZ, -ROPE_VERT])   # pole_b → ankle_b (separate tree)

    # --- Visual / Collision shapes ---
    # All positions relative to link's JOINT frame (empirically confirmed).
    ra_mid = ra_end / 2
    rope_a_vis = p.createVisualShape(
        p.GEOM_CAPSULE, radius=ROPE_R,
        length=np.linalg.norm(ra_end),
        visualFramePosition=ra_mid.tolist(),
        visualFrameOrientation=capsule_quat(ra_end),
        rgbaColor=[1, .8, 0, 1])

    rb_mid = rb_vec / 2
    rope_b_vis = p.createVisualShape(
        p.GEOM_CAPSULE, radius=ROPE_R,
        length=np.linalg.norm(rb_vec),
        visualFramePosition=rb_mid.tolist(),
        visualFrameOrientation=capsule_quat(rb_vec),
        rgbaColor=[1, .8, 0, 1])

    # Lower body: joint at ankle_a, box center at [0, -BD/2, L1/2] from joint
    lower_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[LOWER_W/2, BD/2, L1/2],
        collisionFramePosition=[0, -BD/2, L1/2])
    lower_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[LOWER_W/2, BD/2, L1/2],
        visualFramePosition=[0, -BD/2, L1/2],
        rgbaColor=[.02, .84, .63, .8])

    # Upper body: joint at hip, box center at [0, 0, L2/2] from joint
    upper_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[UPPER_W/2, BD/2, L2/2],
        collisionFramePosition=[0, 0, L2/2])
    upper_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[UPPER_W/2, BD/2, L2/2],
        visualFramePosition=[0, 0, L2/2],
        rgbaColor=[.51, .22, .93, .8])

    hip_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, .75, .04, 1])

    # --- Multibody 1: Robot (pole_a → universal → rope_a → spherical → lower → revolute → upper) ---
    robot_id = p.createMultiBody(
        baseMass=0, basePosition=[0, HALF_POLE, POLE_H],
        baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
        linkMasses=[0.001, ROPE_MASS, M1, M2],
        linkCollisionShapeIndices=[-1, -1, lower_col, upper_col],
        linkVisualShapeIndices=[-1, rope_a_vis, lower_vis, upper_vis],
        linkPositions=[
            [0,0,0],          # L0: dummy at pole_a
            [0,0,0],          # L1: rope_a at pole_a
            ra_end.tolist(),  # L2: lower_body at ankle_a
            [0,-BD/2,L1],     # L3: upper_body at hip
        ],
        linkOrientations=[[0,0,0,1]]*4,
        linkInertialFramePositions=[
            [0,0,0], [0,0,0], [0,-BD/2,L1/2], [0,0,L2/2]
        ],
        linkInertialFrameOrientations=[[0,0,0,1]]*4,
        linkParentIndices=[0, 1, 2, 3],
        linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_SPHERICAL, p.JOINT_REVOLUTE],
        linkJointAxis=[[0,1,0], [1,0,0], [0,0,0], [0,1,0]],
    )
    PHI_J = 0; ROPE_A_J = 1; ANKLE = 2; HIP = 3

    # --- Multibody 2: Rope_b (pole_b → universal → rope_b) ---
    rope_b_id = p.createMultiBody(
        baseMass=0, basePosition=[0, -HALF_POLE, POLE_H],
        baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
        linkMasses=[0.001, ROPE_MASS],
        linkCollisionShapeIndices=[-1, -1],
        linkVisualShapeIndices=[-1, rope_b_vis],
        linkPositions=[[0,0,0], [0,0,0]],
        linkOrientations=[[0,0,0,1]]*2,
        linkInertialFramePositions=[[0,0,0], [0,0,0]],
        linkInertialFrameOrientations=[[0,0,0,1]]*2,
        linkParentIndices=[0, 1],
        linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
        linkJointAxis=[[0,1,0], [1,0,0]],
    )
    RB_DUMMY = 0; RB_ROPE = 1

    # --- P2P between TWO MOVING bodies (not world-fixed!) ---
    # childFramePosition is in COM frame, not joint frame!
    # Lower body COM = [0, -BD/2, L1/2] from joint (ankle_a)
    # ankle_b in joint frame = [0, -BD, 0]
    # ankle_b in COM frame = [0, -BD, 0] - [0, -BD/2, L1/2] = [0, -BD/2, -L1/2]
    con_id = p.createConstraint(
        rope_b_id, RB_ROPE,    # parent: rope_b end
        robot_id, ANKLE,        # child: lower_body ankle_b
        p.JOINT_POINT2POINT,
        [0,0,0],
        rb_vec.tolist(),        # rope_b end in rope_b COM frame
        [0, -BD/2, -L1/2]      # ankle_b in lower_body COM frame
    )
    p.changeConstraint(con_id, maxForce=5000, erp=0.1)

    # --- Dynamics ---
    tiny_inertia = [0.001, 0.001, 0.001]
    # Robot
    p.changeDynamics(robot_id, PHI_J, mass=0.001, localInertiaDiagonal=[0.0001]*3,
                     jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, ROPE_A_J, mass=ROPE_MASS, localInertiaDiagonal=tiny_inertia,
                     jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, ANKLE, mass=M1,
                     localInertiaDiagonal=box_inertia(M1, LOWER_W, BD, L1),
                     jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, HIP, mass=M2,
                     localInertiaDiagonal=box_inertia(M2, UPPER_W, BD, L2),
                     jointDamping=B_THETA, linearDamping=0, angularDamping=0)
    # Rope_b
    for j in [RB_DUMMY, RB_ROPE]:
        p.changeDynamics(rope_b_id, j, localInertiaDiagonal=tiny_inertia,
                         jointDamping=0, linearDamping=0, angularDamping=0)

    # Disable default motors
    for j in [PHI_J, ROPE_A_J, HIP]:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControlMultiDof(robot_id, ANKLE, p.POSITION_CONTROL,
                                   targetPosition=[0,0,0,1], force=[0,0,0])
    for j in [RB_DUMMY, RB_ROPE]:
        p.setJointMotorControl2(rope_b_id, j, p.VELOCITY_CONTROL, force=0)

    # Disable all collisions
    for body in [robot_id, rope_b_id]:
        nj = p.getNumJoints(body)
        for i in range(-1, nj):
            for j in range(-1, nj):
                if i != j: p.setCollisionFilterPair(body, body, i, j, 0)
            p.setCollisionFilterPair(body, 0, i, -1, 0)
    for i in range(-1, p.getNumJoints(robot_id)):
        for j in range(-1, p.getNumJoints(rope_b_id)):
            p.setCollisionFilterPair(robot_id, rope_b_id, i, j, 0)

    # --- State extraction ---
    def get_state():
        # phi from revolute joint (direct, no quaternion conversion!)
        phi_s = p.getJointState(robot_id, PHI_J)
        phi = phi_s[0]; phi_dot = phi_s[1]

        lo = p.getLinkState(robot_id, ANKLE, computeForwardKinematics=True,
                            computeLinkVelocity=True)
        up = p.getLinkState(robot_id, HIP, computeForwardKinematics=True,
                            computeLinkVelocity=True)
        alpha = p.getEulerFromQuaternion(lo[5])[1]
        theta = p.getEulerFromQuaternion(up[5])[1]
        alpha_dot = lo[7][1]
        theta_dot = up[7][1]

        return np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])

    # --- Initial pose ---
    hip_rel0 = THETA0 - ALPHA0
    def reset_pose():
        p.resetJointState(robot_id, PHI_J, 0, 0)
        p.resetJointState(robot_id, ROPE_A_J, 0, 0)
        ha = ALPHA0 / 2
        qa = [0, np.sin(ha), 0, np.cos(ha)]
        p.resetJointStateMultiDof(robot_id, ANKLE, qa, [0,0,0])
        p.resetJointState(robot_id, HIP, hip_rel0, 0)
        p.resetJointState(rope_b_id, RB_DUMMY, 0, 0)
        p.resetJointState(rope_b_id, RB_ROPE, 0, 0)

    reset_pose()

    x0 = get_state()
    print(f"[Init] {', '.join(f'{np.degrees(v):+.1f}d' for v in x0[:3])}")

    # --- Debug UI (right side) ---
    TX = 3.5  # X offset for text (right side)
    p.addUserDebugText(f"V10 Pybullet  m1={M1} m2={M2}  tau_max={TAU_MAX}Nm",
                       [TX,0,POLE_H+.5], textColorRGB=[.5,1,.5], textSize=1.2)
    p.addUserDebugText("Space=Pause R=Reset C=Ctrl Q=Quit",
                       [TX,0,POLE_H+.3], textColorRGB=[1,1,.5], textSize=1.0)
    status_id = p.addUserDebugText("PAUSED", [TX,0,FOOT_Z-.3],
                                    textColorRGB=[1,.3,.3], textSize=1.5)
    torque_id = p.addUserDebugText("", [TX,0,FOOT_Z-.5],
                                    textColorRGB=[.5,.8,1], textSize=1.0)
    time_id = p.addUserDebugText("t=0.000s", [TX,0,FOOT_Z-.7],
                                  textColorRGB=[1,1,1], textSize=1.2)

    # K display
    V10_K = [-6263.3, -5736.9, -2081.0, -1272.7, -1063.3, -562.2]  # V10 R=0.001
    k_fmt = lambda K: '[' + ', '.join(f'{v:.0f}' for v in K) + ']'
    p.addUserDebugText(f"V10 K={k_fmt(V10_K)}",
                       [TX,0,POLE_H+.1], textColorRGB=[.6,.6,.6], textSize=0.8)
    k_display_id = p.addUserDebugText(
        f"K={k_fmt(K_lqr)}",
        [TX,0,POLE_H-.1], textColorRGB=[.3,1,.8], textSize=0.8)

    def update_k_display():
        nonlocal k_display_id
        p.removeUserDebugItem(k_display_id)
        k_display_id = p.addUserDebugText(
            f"K={k_fmt(K_lqr)}  R={R_COST:.1e}",
            [TX,0,POLE_H-.1], textColorRGB=[.3,1,.8], textSize=0.8)

    def set_status(txt, col):
        nonlocal status_id
        p.removeUserDebugItem(status_id)
        status_id = p.addUserDebugText(txt, [TX,0,FOOT_Z-.3],
                                        textColorRGB=col, textSize=1.5)

    def set_torque_txt(tau, x):
        nonlocal torque_id
        p.removeUserDebugItem(torque_id)
        bar = "|" * min(int(abs(tau) / TAU_MAX * 30), 30)
        torque_id = p.addUserDebugText(
            f"tau={tau:+.0f}Nm {bar}  phi={np.degrees(x[0]):+.1f}",
            [TX,0,FOOT_Z-.5], textColorRGB=[.5,.8,1], textSize=1.0)

    # --- CSV log ---
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'v10_pb_log.csv')
    log_f = open(log_path, 'w')
    log_f.write('t,phi,alpha,theta,phi_dot,alpha_dot,theta_dot,tau\n')

    # --- Gain slider (log scale) ---
    # R = 10^slider_val, so slider -5 means R=0.00001
    r_exp_init = np.log10(R_COST)
    gain_slider = p.addUserDebugParameter("log10(R)", -7, -2, r_exp_init)
    last_r_exp = r_exp_init
    # Gain scale slider (direct tau multiplier)
    gs_slider = p.addUserDebugParameter("Gain Scale", 0.1, 5.0, 1.0)
    gain_scale = 1.0

    # --- Simulation loop ---
    paused = True
    ctrl_on = False
    step_n = 0

    print("[Sim] Space=Pause R=Reset C=Toggle_Ctrl Q=Quit")
    print(f"[Sim] Controller OFF (press C)")

    try:
        while p.isConnected():
            keys = p.getKeyboardEvents()

            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                paused = not paused
                if paused:
                    p.setJointMotorControl2(robot_id, HIP,
                                            p.VELOCITY_CONTROL, force=0)
                    x = get_state()
                    set_status("PAUSED", [1,.3,.3])
                    print(f"[PAUSE] t={step_n*DT:.2f}s "
                          f"phi={np.degrees(x[0]):+.1f} "
                          f"a={np.degrees(x[1]):+.1f} "
                          f"th={np.degrees(x[2]):+.1f}")
                else:
                    tag = "LQR" if ctrl_on else "FREE"
                    set_status(f"RUNNING ({tag})", [.3,1,.3])

            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                p.setJointMotorControl2(robot_id, HIP,
                                        p.VELOCITY_CONTROL, force=0)
                reset_pose()
                step_n = 0; paused = True
                set_status("RESET", [1,.3,.3])
                print("[RESET]")

            if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                ctrl_on = not ctrl_on
                print(f"[CTRL] {'ON' if ctrl_on else 'OFF'}")
                if not ctrl_on:
                    p.setJointMotorControl2(robot_id, HIP,
                                            p.VELOCITY_CONTROL, force=0)
                if not paused:
                    tag = "LQR" if ctrl_on else "FREE"
                    set_status(f"RUNNING ({tag})", [.3,1,.3])

            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                break

            if not paused:
                # Check sliders
                r_exp = p.readUserDebugParameter(gain_slider)
                if abs(r_exp - last_r_exp) > 0.01:
                    last_r_exp = r_exp
                    R_COST = 10 ** r_exp
                    K_lqr = compute_lqr()
                    update_k_display()
                gain_scale = p.readUserDebugParameter(gs_slider)

                # V10 uses stepsPerFrame=16: run physics steps per visual frame
                STEPS_PER_FRAME = 24  # > V10's 16 to offset PyBullet API overhead
                for _ in range(STEPS_PER_FRAME):
                    x = get_state()

                    if np.any(np.isnan(x)):
                        print(f"[!] NaN - auto pause. x={x}")
                        paused = True
                        set_status("NaN ERROR", [1,0,0])
                        break

                    tau = 0.0
                    if ctrl_on and K_lqr is not None:
                        if abs(x[1]) < np.radians(60) and abs(x[2]) < np.radians(60):
                            tau = -float(K_lqr @ x) * gain_scale
                            tau = np.clip(tau, -TAU_MAX, TAU_MAX)
                        p.setJointMotorControl2(robot_id, HIP,
                                                p.TORQUE_CONTROL, force=tau)
                    else:
                        p.setJointMotorControl2(robot_id, HIP,
                                                p.VELOCITY_CONTROL, force=0)

                    p.stepSimulation()
                    step_n += 1

                    # Log (every step for first 1s, then every 10th)
                    if step_n < 1000 or step_n % 10 == 0:
                        t = step_n * DT
                        log_f.write(f'{t:.4f},{x[0]:.6f},{x[1]:.6f},{x[2]:.6f},'
                                    f'{x[3]:.4f},{x[4]:.4f},{x[5]:.4f},{tau:.2f}\n')

                # (update_rope_b_visual removed since it is handled by physics engine now)

                if step_n % 128 == 0:  # ~8 frames
                    set_torque_txt(tau, x)
                    # Update time display
                    p.removeUserDebugItem(time_id)
                    t_now = step_n * DT
                    time_id = p.addUserDebugText(
                        f"t={t_now:.2f}s", [TX,0,FOOT_Z-.7],
                        textColorRGB=[1,1,1], textSize=1.2)

                if step_n % 1000 == 0:
                    t = step_n * DT
                    ankle_q = p.getJointStateMultiDof(robot_id, ANKLE)[0]
                    ankle_euler = p.getEulerFromQuaternion(ankle_q)
                    print(f"  t={t:.1f}s  phi={np.degrees(x[0]):+.1f} "
                          f"a={np.degrees(x[1]):+.1f} "
                          f"th={np.degrees(x[2]):+.1f}  tau={tau:+.0f}"
                          f"  ankle_Y={np.degrees(ankle_euler[1]):+.2f}")

            time.sleep(1/60)  # ~60fps frame rate

    except Exception as e:
        print(f"[Error] {e}")
        import traceback; traceback.print_exc()
    finally:
        log_f.close()
        print(f"[LOG] {log_path}")
        try: p.disconnect()
        except: pass
        print("[Done]")


if __name__ == '__main__':
    main()

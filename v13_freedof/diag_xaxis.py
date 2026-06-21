"""
Diagnostic: X-axis drift in V13
================================
1. Run V13 headless with LQR control
2. Monitor ankle spherical joint quaternion → X-axis (roll) component
3. Check P2P constraint violation
4. Check for asymmetric forces/geometry
"""
import sys, os, math
import numpy as np

_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib) and _lib not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_lib)

import pybullet as p
import pybullet_data

# === Parameters (from v13_pybullet.py) ===
M1 = 30.0; M2 = 40.0; L1 = 0.85; L2 = 0.80
LOWER_W = 0.18; UPPER_W = 0.30; BD = 0.50
B_THETA = 3.0; TAU_MAX = 1500.0; GRAV = 9.81
L_POLE = 4.0; POLE_H = 4.5; R_TARGET = 1.5
HALF_POLE = L_POLE / 2.0
ROPE_HORIZ = HALF_POLE - BD / 2.0
ROPE_VERT = R_TARGET
FOOT_Z = POLE_H - ROPE_VERT
LC1 = L1/2; LC2 = L2/2
I1 = M1*L1**2/12; I2 = M2*L2**2/12
DT = 0.001; ROPE_R = 0.012; ROPE_MASS = 0.01
THETA0 = 0.15

def box_inertia(m, w, d, h):
    return [m*(d**2+h**2)/12, m*(w**2+h**2)/12, m*(w**2+d**2)/12]

def compute_lqr():
    from scipy.linalg import solve_discrete_are, expm
    R = R_TARGET
    p1 = M1*LC1 + M2*L1; p2 = M2*LC2; p3 = M2*L1*LC2; Mt = M1+M2
    M_eq = np.array([[Mt*R**2, R*p1, R*p2],[R*p1, M1*LC1**2+I1+M2*L1**2, p3],[R*p2, p3, M2*LC2**2+I2]])
    G_mat = np.array([[-Mt*GRAV*R,0,0],[0,p1*GRAV,0],[0,0,p2*GRAV]])
    D_mat = np.diag([0,0,-B_THETA])
    E_mat = np.array([[0],[-1],[1]])
    Mi = np.linalg.inv(M_eq); n=6
    Ac = np.zeros((n,n)); Bc = np.zeros((n,1))
    Ac[:3,3:] = np.eye(3); Ac[3:,:3] = Mi@G_mat; Ac[3:,3:] = Mi@D_mat; Bc[3:,:] = Mi@E_mat
    Z = np.zeros((n+1,n+1)); Z[:n,:n]=Ac*DT; Z[:n,n:]=Bc*DT
    eZ = expm(Z); Ad=eZ[:n,:n]; Bd=eZ[:n,n:n+1]
    Q = np.diag([1,50,50,0.1,5,5]); Rc = np.array([[0.001]])
    P = solve_discrete_are(Ad, Bd, Q, Rc)
    K = np.linalg.inv(Rc + Bd.T@P@Bd) @ (Bd.T@P@Ad)
    K[0,0] *= -1; K[0,3] *= -1
    return K[0]

def main():
    K_lqr = compute_lqr()
    print(f"[LQR] K = {K_lqr}")

    cid = p.connect(p.DIRECT)  # headless
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -GRAV)
    p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)

    # === Build V13 (same as v13_pybullet.py) ===
    ra_end = np.array([0, -ROPE_HORIZ, -ROPE_VERT])
    rb_vec = np.array([0, ROPE_HORIZ, -ROPE_VERT])

    lower_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[LOWER_W/2, BD/2, L1/2],
        collisionFramePosition=[0, -BD/2, L1/2])
    upper_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[UPPER_W/2, BD/2, L2/2],
        collisionFramePosition=[0, 0, L2/2])

    robot_id = p.createMultiBody(
        baseMass=0, basePosition=[0, HALF_POLE, POLE_H],
        linkMasses=[0.001, ROPE_MASS, M1, M2],
        linkCollisionShapeIndices=[-1, -1, lower_col, upper_col],
        linkVisualShapeIndices=[-1, -1, -1, -1],
        linkPositions=[[0,0,0],[0,0,0], ra_end.tolist(), [0,-BD/2,L1]],
        linkOrientations=[[0,0,0,1]]*4,
        linkInertialFramePositions=[[0,0,0],[0,0,0],[0,-BD/2,L1/2],[0,0,L2/2]],
        linkInertialFrameOrientations=[[0,0,0,1]]*4,
        linkParentIndices=[0,1,2,3],
        linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE, p.JOINT_SPHERICAL, p.JOINT_REVOLUTE],
        linkJointAxis=[[0,1,0],[1,0,0],[0,0,0],[0,1,0]],
    )
    PHI_J=0; ROPE_A_J=1; ANKLE=2; HIP=3

    rope_b_id = p.createMultiBody(
        baseMass=0, basePosition=[0, -HALF_POLE, POLE_H],
        linkMasses=[0.001, ROPE_MASS],
        linkCollisionShapeIndices=[-1,-1],
        linkVisualShapeIndices=[-1,-1],
        linkPositions=[[0,0,0],[0,0,0]],
        linkOrientations=[[0,0,0,1]]*2,
        linkInertialFramePositions=[[0,0,0],[0,0,0]],
        linkInertialFrameOrientations=[[0,0,0,1]]*2,
        linkParentIndices=[0,1],
        linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
        linkJointAxis=[[0,1,0],[1,0,0]],
    )
    RB_DUMMY=0; RB_ROPE=1

    con_id = p.createConstraint(
        rope_b_id, RB_ROPE, robot_id, ANKLE,
        p.JOINT_POINT2POINT, [0,0,0],
        rb_vec.tolist(), [0, -BD/2, -L1/2]
    )
    p.changeConstraint(con_id, maxForce=5000, erp=0.1)

    # Dynamics
    tiny = [0.001,0.001,0.001]
    p.changeDynamics(robot_id, PHI_J, mass=0.001, localInertiaDiagonal=[0.0001]*3, jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, ROPE_A_J, mass=ROPE_MASS, localInertiaDiagonal=tiny, jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, ANKLE, mass=M1, localInertiaDiagonal=box_inertia(M1,LOWER_W,BD,L1), jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot_id, HIP, mass=M2, localInertiaDiagonal=box_inertia(M2,UPPER_W,BD,L2), jointDamping=B_THETA, linearDamping=0, angularDamping=0)
    for j in [RB_DUMMY, RB_ROPE]:
        p.changeDynamics(rope_b_id, j, localInertiaDiagonal=tiny, jointDamping=0, linearDamping=0, angularDamping=0)

    for j in [PHI_J, ROPE_A_J, HIP]:
        p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControlMultiDof(robot_id, ANKLE, p.POSITION_CONTROL, targetPosition=[0,0,0,1], force=[0,0,0])
    for j in [RB_DUMMY, RB_ROPE]:
        p.setJointMotorControl2(rope_b_id, j, p.VELOCITY_CONTROL, force=0)

    # Disable collisions
    for body in [robot_id, rope_b_id]:
        nj = p.getNumJoints(body)
        for i in range(-1, nj):
            for j in range(-1, nj):
                if i!=j: p.setCollisionFilterPair(body, body, i, j, 0)
            p.setCollisionFilterPair(body, 0, i, -1, 0)
    for i in range(-1, p.getNumJoints(robot_id)):
        for j in range(-1, p.getNumJoints(rope_b_id)):
            p.setCollisionFilterPair(robot_id, rope_b_id, i, j, 0)

    # Initial pose
    hip_rel0 = THETA0
    p.resetJointState(robot_id, PHI_J, 0, 0)
    p.resetJointState(robot_id, ROPE_A_J, 0, 0)
    p.resetJointStateMultiDof(robot_id, ANKLE, [0,0,0,1], [0,0,0])
    p.resetJointState(robot_id, HIP, hip_rel0, 0)
    p.resetJointState(rope_b_id, RB_DUMMY, 0, 0)
    p.resetJointState(rope_b_id, RB_ROPE, 0, 0)

    # === DIAGNOSTIC: Geometry check ===
    print("\n" + "="*60)
    print("  GEOMETRY CHECK")
    print("="*60)

    # Check rope_a attachment point (pole_a top)
    base_pos = [0, HALF_POLE, POLE_H]
    print(f"  Pole A top (robot base): {base_pos}")
    print(f"  Pole B top (rope_b base): [0, {-HALF_POLE}, {POLE_H}]")

    # Check ankle_a world position
    lo = p.getLinkState(robot_id, ANKLE, computeForwardKinematics=True)
    print(f"  Ankle (lower body) COM world: [{lo[0][0]:.4f}, {lo[0][1]:.4f}, {lo[0][2]:.4f}]")
    print(f"  Ankle link frame world: [{lo[4][0]:.4f}, {lo[4][1]:.4f}, {lo[4][2]:.4f}]")

    # Check rope_b end world position
    rb = p.getLinkState(rope_b_id, RB_ROPE, computeForwardKinematics=True)
    print(f"  Rope_b link frame world: [{rb[4][0]:.4f}, {rb[4][1]:.4f}, {rb[4][2]:.4f}]")

    # P2P constraint points in world frame
    # Parent pivot (rope_b end): rb_vec from rope_b link frame
    # Child pivot (ankle): [0, -BD/2, -L1/2] from lower body COM frame
    ankle_com = np.array(lo[0])
    ankle_orn = lo[1]
    R_ankle = np.array(p.getMatrixFromQuaternion(ankle_orn)).reshape(3,3)
    ankle_b_world = ankle_com + R_ankle @ np.array([0, -BD/2, -L1/2])

    rb_com = np.array(rb[0])
    rb_orn = rb[1]
    R_rb = np.array(p.getMatrixFromQuaternion(rb_orn)).reshape(3,3)
    rope_b_end_world = rb_com + R_rb @ rb_vec

    constraint_err = np.linalg.norm(rope_b_end_world - ankle_b_world)
    print(f"\n  Ankle_b (from lower body): [{ankle_b_world[0]:.4f}, {ankle_b_world[1]:.4f}, {ankle_b_world[2]:.4f}]")
    print(f"  Rope_b end (from rope_b):  [{rope_b_end_world[0]:.4f}, {rope_b_end_world[1]:.4f}, {rope_b_end_world[2]:.4f}]")
    print(f"  P2P constraint error: {constraint_err:.6f} m")

    # Check symmetry: ankle_a and ankle_b should be symmetric about X-Z plane
    ankle_a_world = np.array(lo[4])  # link frame = joint frame
    print(f"\n  Ankle_a world (joint frame): [{ankle_a_world[0]:.4f}, {ankle_a_world[1]:.4f}, {ankle_a_world[2]:.4f}]")
    print(f"  Ankle_b world (constraint):  [{ankle_b_world[0]:.4f}, {ankle_b_world[1]:.4f}, {ankle_b_world[2]:.4f}]")
    print(f"  Y-midpoint: {(ankle_a_world[1]+ankle_b_world[1])/2:.4f} (should be ~0)")
    print(f"  Z-difference: {abs(ankle_a_world[2]-ankle_b_world[2]):.4f} (should be ~0)")

    # Check rope lengths
    pole_a = np.array([0, HALF_POLE, POLE_H])
    pole_b = np.array([0, -HALF_POLE, POLE_H])
    rope_a_len = np.linalg.norm(ankle_a_world - pole_a)
    rope_b_len = np.linalg.norm(ankle_b_world - pole_b)
    expected_len = np.sqrt(ROPE_HORIZ**2 + ROPE_VERT**2)
    print(f"\n  Rope_a length: {rope_a_len:.4f} (expected: {expected_len:.4f})")
    print(f"  Rope_b length: {rope_b_len:.4f} (expected: {expected_len:.4f})")
    print(f"  Length difference: {abs(rope_a_len-rope_b_len):.6f}")

    # === RUN SIMULATION ===
    print("\n" + "="*60)
    print("  RUNNING SIMULATION (5 seconds with LQR)")
    print("="*60)
    print(f"  {'t':>5s}  {'phi_Y':>8s}  {'alpha_Y':>8s}  {'theta_Y':>8s}  |  {'ankle_X':>8s}  {'ankle_Z':>8s}  {'ropeAx':>8s}  {'P2Perr':>8s}  {'tau':>8s}")

    data = []
    TOTAL_STEPS = int(5.0 / DT)

    for step in range(TOTAL_STEPS):
        # Get state
        phi_s = p.getJointState(robot_id, PHI_J)
        phi = phi_s[0]; phi_dot = phi_s[1]

        # Rope_a X joint (should stay ~0 if no X-axis coupling)
        rope_ax_s = p.getJointState(robot_id, ROPE_A_J)
        rope_ax = rope_ax_s[0]

        lo = p.getLinkState(robot_id, ANKLE, computeForwardKinematics=True, computeLinkVelocity=True)
        up = p.getLinkState(robot_id, HIP, computeForwardKinematics=True, computeLinkVelocity=True)

        # Full Euler angles from ankle (spherical joint)
        ankle_q = p.getJointStateMultiDof(robot_id, ANKLE)[0]
        ankle_euler = p.getEulerFromQuaternion(ankle_q)  # (roll_X, pitch_Y, yaw_Z)

        alpha = p.getEulerFromQuaternion(lo[5])[1]
        theta = p.getEulerFromQuaternion(up[5])[1]
        alpha_dot = lo[7][1]
        theta_dot = up[7][1]

        x = np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])

        # LQR control
        tau = 0.0
        if abs(x[1]) < np.radians(60) and abs(x[2]) < np.radians(60):
            tau = -float(K_lqr @ x)
            tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        p.setJointMotorControl2(robot_id, HIP, p.TORQUE_CONTROL, force=tau)

        # P2P constraint error
        ankle_com = np.array(lo[0])
        R_a = np.array(p.getMatrixFromQuaternion(lo[1])).reshape(3,3)
        ab_w = ankle_com + R_a @ np.array([0, -BD/2, -L1/2])

        rb_s = p.getLinkState(rope_b_id, RB_ROPE, computeForwardKinematics=True)
        rb_com = np.array(rb_s[0])
        R_r = np.array(p.getMatrixFromQuaternion(rb_s[1])).reshape(3,3)
        rbe_w = rb_com + R_r @ rb_vec
        p2p_err = np.linalg.norm(rbe_w - ab_w)

        p.stepSimulation()

        # Log every 500 steps (0.5s)
        if step % 500 == 0:
            t = step * DT
            print(f"  {t:5.2f}s  {np.degrees(phi):+8.3f}  {np.degrees(alpha):+8.3f}  {np.degrees(theta):+8.3f}  |  "
                  f"{np.degrees(ankle_euler[0]):+8.3f}  {np.degrees(ankle_euler[2]):+8.3f}  "
                  f"{np.degrees(rope_ax):+8.3f}  {p2p_err:8.5f}  {tau:+8.1f}")

        # Store data for analysis
        if step % 10 == 0:
            data.append([step*DT, phi, alpha, theta, ankle_euler[0], ankle_euler[2], rope_ax, p2p_err, tau])

    data = np.array(data)

    # === ANALYSIS ===
    print("\n" + "="*60)
    print("  ANALYSIS")
    print("="*60)

    ankle_x = np.degrees(data[:, 4])
    ankle_z = np.degrees(data[:, 5])
    rope_ax_deg = np.degrees(data[:, 6])
    p2p_errs = data[:, 7]

    print(f"  Ankle X (roll) range: [{ankle_x.min():+.3f}, {ankle_x.max():+.3f}] deg")
    print(f"  Ankle X final: {ankle_x[-1]:+.3f} deg")
    print(f"  Ankle X drift rate: {(ankle_x[-1]-ankle_x[0])/5:.3f} deg/s")
    print(f"  Ankle Z (yaw) range: [{ankle_z.min():+.3f}, {ankle_z.max():+.3f}] deg")
    print(f"  Rope_A X-joint range: [{rope_ax_deg.min():+.3f}, {rope_ax_deg.max():+.3f}] deg")
    print(f"  P2P error range: [{p2p_errs.min():.6f}, {p2p_errs.max():.6f}] m")
    print(f"  P2P error mean: {p2p_errs.mean():.6f} m")

    # Check monotonicity (sign of drift)
    if len(ankle_x) > 100:
        first_half = ankle_x[:len(ankle_x)//2].mean()
        second_half = ankle_x[len(ankle_x)//2:].mean()
        direction = "positive (+X)" if second_half > first_half else "negative (-X)"
        print(f"  Drift direction: {direction}")
        print(f"  1st half mean: {first_half:+.3f}, 2nd half mean: {second_half:+.3f}")

    # Plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("V13 X-axis Drift Diagnosis", fontsize=14)

        t = data[:, 0]
        axes[0].plot(t, np.degrees(data[:,1]), label='phi (Y)')
        axes[0].plot(t, np.degrees(data[:,2]), label='alpha (Y)')
        axes[0].plot(t, np.degrees(data[:,3]), label='theta (Y)')
        axes[0].set_ylabel('Y-axis angles (deg)')
        axes[0].legend(); axes[0].grid(True)
        axes[0].set_title('Y-axis (normal balance) — should stabilize')

        axes[1].plot(t, ankle_x, 'r-', label='ankle roll (X)', linewidth=2)
        axes[1].plot(t, rope_ax_deg, 'b--', label='rope_a X-joint')
        axes[1].set_ylabel('X-axis angles (deg)')
        axes[1].legend(); axes[1].grid(True)
        axes[1].set_title('X-axis rotation — monotonic drift = BUG')

        axes[2].plot(t, ankle_z, 'g-', label='ankle yaw (Z)')
        axes[2].set_ylabel('Z-axis angle (deg)')
        axes[2].legend(); axes[2].grid(True)
        axes[2].set_title('Z-axis rotation')

        axes[3].plot(t, p2p_errs * 1000, 'm-', label='P2P error')
        axes[3].set_ylabel('Constraint error (mm)')
        axes[3].set_xlabel('Time (s)')
        axes[3].legend(); axes[3].grid(True)
        axes[3].set_title('P2P Constraint Violation')

        plt.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_xaxis.png')
        plt.savefig(out, dpi=150)
        print(f"\n  Plot saved: {out}")
        plt.close()
    except ImportError:
        print("  (matplotlib not available, skipping plot)")

    p.disconnect()
    print("[Done]")

if __name__ == '__main__':
    main()

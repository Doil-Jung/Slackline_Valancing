"""
Test: Symmetric P2P architecture
Both ankles connected via P2P constraints (no spherical joint in chain).
Robot body = free-floating multibody (lower + upper with hip revolute).
rope_a, rope_b = separate multibodies, each connected to body via P2P.
"""
import sys, os
import numpy as np

_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib) and _lib not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"): os.add_dll_directory(_lib)

import pybullet as p
import pybullet_data

M1=30.0;M2=40.0;L1=0.85;L2=0.80;LOWER_W=0.18;UPPER_W=0.30;BD=0.50
B_THETA=3.0;TAU_MAX=1500.0;GRAV=9.81;L_POLE=4.0;POLE_H=4.5;R_TARGET=1.5
HALF_POLE=L_POLE/2;ROPE_HORIZ=HALF_POLE-BD/2;ROPE_VERT=R_TARGET
FOOT_Z=POLE_H-ROPE_VERT;LC1=L1/2;LC2=L2/2;I1=M1*L1**2/12;I2=M2*L2**2/12
DT=0.001;ROPE_MASS=0.01;THETA0=0.15

def box_inertia(m,w,d,h):
    return [m*(d**2+h**2)/12, m*(w**2+h**2)/12, m*(w**2+d**2)/12]

def compute_lqr():
    from scipy.linalg import solve_discrete_are, expm
    R=R_TARGET;p1=M1*LC1+M2*L1;p2=M2*LC2;p3=M2*L1*LC2;Mt=M1+M2
    M_eq=np.array([[Mt*R**2,R*p1,R*p2],[R*p1,M1*LC1**2+I1+M2*L1**2,p3],[R*p2,p3,M2*LC2**2+I2]])
    G_mat=np.array([[-Mt*GRAV*R,0,0],[0,p1*GRAV,0],[0,0,p2*GRAV]])
    D_mat=np.diag([0,0,-B_THETA]);E_mat=np.array([[0],[-1],[1]])
    Mi=np.linalg.inv(M_eq);n=6
    Ac=np.zeros((n,n));Bc=np.zeros((n,1))
    Ac[:3,3:]=np.eye(3);Ac[3:,:3]=Mi@G_mat;Ac[3:,3:]=Mi@D_mat;Bc[3:,:]=Mi@E_mat
    Z=np.zeros((n+1,n+1));Z[:n,:n]=Ac*DT;Z[:n,n:]=Bc*DT
    eZ=expm(Z);Ad=eZ[:n,:n];Bd=eZ[:n,n:n+1]
    Q=np.diag([1,50,50,0.1,5,5]);Rc=np.array([[0.001]])
    P=solve_discrete_are(Ad,Bd,Q,Rc)
    K=np.linalg.inv(Rc+Bd.T@P@Bd)@(Bd.T@P@Ad)
    K[0,0]*=-1;K[0,3]*=-1
    return K[0]

def make_rope(base_y, rope_vec):
    """Create a rope multibody: base at pole top, universal joint, rigid link."""
    rope_id = p.createMultiBody(
        baseMass=0, basePosition=[0, base_y, POLE_H],
        linkMasses=[0.001, ROPE_MASS],
        linkCollisionShapeIndices=[-1, -1],
        linkVisualShapeIndices=[-1, -1],
        linkPositions=[[0,0,0],[0,0,0]],
        linkOrientations=[[0,0,0,1]]*2,
        linkInertialFramePositions=[[0,0,0],[0,0,0]],
        linkInertialFrameOrientations=[[0,0,0,1]]*2,
        linkParentIndices=[0, 1],
        linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
        linkJointAxis=[[0,1,0], [1,0,0]],
    )
    tiny = [0.001]*3
    for j in [0, 1]:
        p.changeDynamics(rope_id, j, localInertiaDiagonal=tiny,
                         jointDamping=0, linearDamping=0, angularDamping=0)
        p.setJointMotorControl2(rope_id, j, p.VELOCITY_CONTROL, force=0)
    return rope_id

def run_test(label, symmetric=True):
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-GRAV); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)

    ra_vec = np.array([0, -ROPE_HORIZ, -ROPE_VERT])  # pole_a → ankle_a
    rb_vec = np.array([0,  ROPE_HORIZ, -ROPE_VERT])  # pole_b → ankle_b

    lower_col = p.createCollisionShape(p.GEOM_BOX,
        halfExtents=[LOWER_W/2, BD/2, L1/2])
    upper_col = p.createCollisionShape(p.GEOM_BOX,
        halfExtents=[UPPER_W/2, BD/2, L2/2],
        collisionFramePosition=[0, 0, L2/2])

    if symmetric:
        # === SYMMETRIC: Robot body as free-floating, both ropes via P2P ===
        # Robot base = lower body COM at [0, 0, FOOT_Z + L1/2]
        body_com_init = [0, 0, FOOT_Z + L1/2]
        robot_id = p.createMultiBody(
            baseMass=M1,
            basePosition=body_com_init,
            baseCollisionShapeIndex=lower_col,
            baseVisualShapeIndex=-1,
            baseInertialFramePosition=[0, 0, 0],
            linkMasses=[M2],
            linkCollisionShapeIndices=[upper_col],
            linkVisualShapeIndices=[-1],
            linkPositions=[[0, 0, L1/2]],  # hip at top of lower body (from COM)
            linkOrientations=[[0,0,0,1]],
            linkInertialFramePositions=[[0, 0, L2/2]],
            linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],
            linkJointTypes=[p.JOINT_REVOLUTE],
            linkJointAxis=[[0,1,0]],
        )
        HIP = 0

        p.changeDynamics(robot_id, -1, mass=M1,
            localInertiaDiagonal=box_inertia(M1, LOWER_W, BD, L1),
            linearDamping=0, angularDamping=0)
        p.changeDynamics(robot_id, HIP, mass=M2,
            localInertiaDiagonal=box_inertia(M2, UPPER_W, BD, L2),
            jointDamping=B_THETA, linearDamping=0, angularDamping=0)
        p.setJointMotorControl2(robot_id, HIP, p.VELOCITY_CONTROL, force=0)

        # Two rope multibodies
        rope_a_id = make_rope(+HALF_POLE, ra_vec)
        rope_b_id = make_rope(-HALF_POLE, rb_vec)

        # ankle_a in lower body COM frame: [0, +BD/2, -L1/2]
        # ankle_b in lower body COM frame: [0, -BD/2, -L1/2]
        con_a = p.createConstraint(
            rope_a_id, 1, robot_id, -1,  # rope_a end ↔ robot base (lower body)
            p.JOINT_POINT2POINT, [0,0,0],
            ra_vec.tolist(),           # rope_a endpoint in rope_a COM frame
            [0, +BD/2, -L1/2]         # ankle_a in lower body COM frame
        )
        con_b = p.createConstraint(
            rope_b_id, 1, robot_id, -1,
            p.JOINT_POINT2POINT, [0,0,0],
            rb_vec.tolist(),
            [0, -BD/2, -L1/2]
        )
        p.changeConstraint(con_a, maxForce=50000, erp=0.8)
        p.changeConstraint(con_b, maxForce=50000, erp=0.8)

        # Initial pose
        p.resetBasePositionAndOrientation(robot_id, body_com_init, [0,0,0,1])
        p.resetBaseVelocity(robot_id, [0,0,0], [0,0,0])
        p.resetJointState(robot_id, HIP, THETA0, 0)
        for rid in [rope_a_id, rope_b_id]:
            p.resetJointState(rid, 0, 0, 0)
            p.resetJointState(rid, 1, 0, 0)

        # Disable collisions
        all_bodies = [robot_id, rope_a_id, rope_b_id, 0]
        for b1 in [robot_id, rope_a_id, rope_b_id]:
            nj = p.getNumJoints(b1)
            for i in range(-1, nj):
                for j in range(-1, nj):
                    if i != j: p.setCollisionFilterPair(b1, b1, i, j, 0)
                for b2 in all_bodies:
                    if b1 != b2:
                        nj2 = max(p.getNumJoints(b2), 0)
                        for j in range(-1, nj2):
                            p.setCollisionFilterPair(b1, b2, i, j, 0)

    else:
        # === ORIGINAL V13 (asymmetric) ===
        ra_end = ra_vec
        robot_id = p.createMultiBody(baseMass=0, basePosition=[0, HALF_POLE, POLE_H],
            linkMasses=[0.001, ROPE_MASS, M1, M2],
            linkCollisionShapeIndices=[-1,-1,-1,-1],
            linkVisualShapeIndices=[-1,-1,-1,-1],
            linkPositions=[[0,0,0],[0,0,0],ra_end.tolist(),[0,-BD/2,L1]],
            linkOrientations=[[0,0,0,1]]*4,
            linkInertialFramePositions=[[0,0,0],[0,0,0],[0,-BD/2,L1/2],[0,0,L2/2]],
            linkInertialFrameOrientations=[[0,0,0,1]]*4,
            linkParentIndices=[0,1,2,3],
            linkJointTypes=[p.JOINT_REVOLUTE,p.JOINT_REVOLUTE,p.JOINT_SPHERICAL,p.JOINT_REVOLUTE],
            linkJointAxis=[[0,1,0],[1,0,0],[0,0,0],[0,1,0]])
        HIP=3
        tiny=[0.001]*3
        p.changeDynamics(robot_id,0,mass=0.001,localInertiaDiagonal=[0.0001]*3,jointDamping=0,linearDamping=0,angularDamping=0)
        p.changeDynamics(robot_id,1,mass=ROPE_MASS,localInertiaDiagonal=tiny,jointDamping=0,linearDamping=0,angularDamping=0)
        p.changeDynamics(robot_id,2,mass=M1,localInertiaDiagonal=box_inertia(M1,LOWER_W,BD,L1),jointDamping=0,linearDamping=0,angularDamping=0)
        p.changeDynamics(robot_id,HIP,mass=M2,localInertiaDiagonal=box_inertia(M2,UPPER_W,BD,L2),jointDamping=B_THETA,linearDamping=0,angularDamping=0)
        for j in [0,1,HIP]: p.setJointMotorControl2(robot_id,j,p.VELOCITY_CONTROL,force=0)
        p.setJointMotorControlMultiDof(robot_id,2,p.POSITION_CONTROL,targetPosition=[0,0,0,1],force=[0,0,0])

        rope_b_id = make_rope(-HALF_POLE, rb_vec)
        con_b = p.createConstraint(rope_b_id,1,robot_id,2,p.JOINT_POINT2POINT,[0,0,0],
            rb_vec.tolist(),[0,-BD/2,-L1/2])
        p.changeConstraint(con_b, maxForce=50000, erp=0.8)

        p.resetJointState(robot_id,0,0,0)
        p.resetJointState(robot_id,1,0,0)
        p.resetJointStateMultiDof(robot_id,2,[0,0,0,1],[0,0,0])
        p.resetJointState(robot_id,HIP,THETA0,0)
        p.resetJointState(rope_b_id,0,0,0)
        p.resetJointState(rope_b_id,1,0,0)

        for body in [robot_id,rope_b_id]:
            nj=p.getNumJoints(body)
            for i in range(-1,nj):
                for j in range(-1,nj):
                    if i!=j: p.setCollisionFilterPair(body,body,i,j,0)
                p.setCollisionFilterPair(body,0,i,-1,0)
        for i in range(-1,p.getNumJoints(robot_id)):
            for j in range(-1,p.getNumJoints(rope_b_id)):
                p.setCollisionFilterPair(robot_id,rope_b_id,i,j,0)
        rope_a_id = None

    K_lqr = compute_lqr()
    data = []

    for step in range(int(5.0/DT)):
        if symmetric:
            # Get state from base orientation + hip
            base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
            base_vel, base_angvel = p.getBaseVelocity(robot_id)
            euler = p.getEulerFromQuaternion(base_orn)

            # phi: estimate from base X position relative to expected
            # For symmetric, phi ~ lateral displacement / R
            phi = base_pos[0] / R_TARGET if R_TARGET > 0 else 0
            phi_dot = base_vel[0] / R_TARGET if R_TARGET > 0 else 0
            alpha = euler[1]  # pitch Y
            alpha_dot = base_angvel[1]

            hip_s = p.getJointState(robot_id, HIP)
            hip_angle = hip_s[0]
            theta = alpha + hip_angle
            theta_dot = alpha_dot + hip_s[1]

            x_roll = np.degrees(euler[0])  # roll X
        else:
            phi_s = p.getJointState(robot_id, 0)
            phi = phi_s[0]; phi_dot = phi_s[1]
            lo = p.getLinkState(robot_id, 2, computeForwardKinematics=True, computeLinkVelocity=True)
            up = p.getLinkState(robot_id, HIP, computeForwardKinematics=True, computeLinkVelocity=True)
            alpha = p.getEulerFromQuaternion(lo[5])[1]
            theta = p.getEulerFromQuaternion(up[5])[1]
            alpha_dot = lo[7][1]; theta_dot = up[7][1]
            ankle_q = p.getJointStateMultiDof(robot_id, 2)[0]
            x_roll = np.degrees(p.getEulerFromQuaternion(ankle_q)[0])

        x = np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])
        tau = 0.0
        if abs(x[1]) < np.radians(60) and abs(x[2]) < np.radians(60):
            tau = -float(K_lqr @ x)
            tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        p.setJointMotorControl2(robot_id, HIP, p.TORQUE_CONTROL, force=tau)
        p.stepSimulation()

        if step % 10 == 0:
            data.append([step*DT, x_roll, np.degrees(phi), np.degrees(alpha), np.degrees(theta), tau])

    p.disconnect()
    data = np.array(data)
    print(f"  [{label}] X-roll @3s={data[300,1]:+.2f}°  @5s={data[-1,1]:+.2f}°  "
          f"max|X|={np.max(np.abs(data[:300,1])):.2f}°")
    return data

# === Run ===
print("="*60)
print("  SYMMETRIC vs ASYMMETRIC P2P TEST")
print("="*60)
K_lqr = compute_lqr()
print()

r_orig = run_test("Original V13 (asymmetric)", symmetric=False)
r_sym  = run_test("Symmetric P2P (both sides)", symmetric=True)

# Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Symmetric P2P vs Original V13", fontsize=14)

for label, d, ls in [("Original (asymmetric)", r_orig, '-'), ("Symmetric P2P", r_sym, '--')]:
    axes[0].plot(d[:,0], d[:,1], ls, label=label, linewidth=2)
axes[0].set_ylabel("X-roll (deg)"); axes[0].set_title("X-axis rotation (should stay ~0)")
axes[0].legend(); axes[0].grid(True); axes[0].axhline(0, color='k', lw=0.5)

for label, d, ls in [("Original", r_orig, '-'), ("Symmetric", r_sym, '--')]:
    axes[1].plot(d[:,0], d[:,3], ls, label=f"{label} alpha", alpha=0.8)
    axes[1].plot(d[:,0], d[:,4], ls, label=f"{label} theta", alpha=0.5)
axes[1].set_ylabel("Y-axis (deg)"); axes[1].set_title("Y-axis balance")
axes[1].legend(fontsize=8); axes[1].grid(True)

for label, d, ls in [("Original", r_orig, '-'), ("Symmetric", r_sym, '--')]:
    axes[2].plot(d[:,0], d[:,5], ls, label=label, alpha=0.8)
axes[2].set_ylabel("Torque (Nm)"); axes[2].set_xlabel("Time (s)")
axes[2].set_title("Hip torque"); axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_symmetric.png')
plt.savefig(out, dpi=150)
print(f"\nPlot: {out}")

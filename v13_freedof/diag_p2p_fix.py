"""
Fix test: P2P constraint stiffness
===================================
Test different maxForce & erp values to prevent X-axis drift.
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

# Parameters
M1=30.0; M2=40.0; L1=0.85; L2=0.80
LOWER_W=0.18; UPPER_W=0.30; BD=0.50
B_THETA=3.0; TAU_MAX=1500.0; GRAV=9.81
L_POLE=4.0; POLE_H=4.5; R_TARGET=1.5
HALF_POLE=L_POLE/2; ROPE_HORIZ=HALF_POLE-BD/2; ROPE_VERT=R_TARGET
FOOT_Z=POLE_H-ROPE_VERT; LC1=L1/2; LC2=L2/2
I1=M1*L1**2/12; I2=M2*L2**2/12
DT=0.001; ROPE_MASS=0.01; THETA0=0.15

def box_inertia(m,w,d,h):
    return [m*(d**2+h**2)/12, m*(w**2+h**2)/12, m*(w**2+d**2)/12]

def compute_lqr():
    from scipy.linalg import solve_discrete_are, expm
    R=R_TARGET; p1=M1*LC1+M2*L1; p2=M2*LC2; p3=M2*L1*LC2; Mt=M1+M2
    M_eq=np.array([[Mt*R**2,R*p1,R*p2],[R*p1,M1*LC1**2+I1+M2*L1**2,p3],[R*p2,p3,M2*LC2**2+I2]])
    G_mat=np.array([[-Mt*GRAV*R,0,0],[0,p1*GRAV,0],[0,0,p2*GRAV]])
    D_mat=np.diag([0,0,-B_THETA]); E_mat=np.array([[0],[-1],[1]])
    Mi=np.linalg.inv(M_eq); n=6
    Ac=np.zeros((n,n)); Bc=np.zeros((n,1))
    Ac[:3,3:]=np.eye(3); Ac[3:,:3]=Mi@G_mat; Ac[3:,3:]=Mi@D_mat; Bc[3:,:]=Mi@E_mat
    Z=np.zeros((n+1,n+1)); Z[:n,:n]=Ac*DT; Z[:n,n:]=Bc*DT
    eZ=expm(Z); Ad=eZ[:n,:n]; Bd=eZ[:n,n:n+1]
    Q=np.diag([1,50,50,0.1,5,5]); Rc=np.array([[0.001]])
    P=solve_discrete_are(Ad,Bd,Q,Rc)
    K=np.linalg.inv(Rc+Bd.T@P@Bd)@(Bd.T@P@Ad)
    K[0,0]*=-1; K[0,3]*=-1
    return K[0]

def run_sim(maxForce, erp, label):
    """Run 5s headless sim, return time series of ankle X-roll."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-GRAV); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)

    ra_end=np.array([0,-ROPE_HORIZ,-ROPE_VERT])
    rb_vec=np.array([0,ROPE_HORIZ,-ROPE_VERT])
    lower_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[LOWER_W/2,BD/2,L1/2],collisionFramePosition=[0,-BD/2,L1/2])
    upper_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[UPPER_W/2,BD/2,L2/2],collisionFramePosition=[0,0,L2/2])

    robot_id=p.createMultiBody(baseMass=0,basePosition=[0,HALF_POLE,POLE_H],
        linkMasses=[0.001,ROPE_MASS,M1,M2],linkCollisionShapeIndices=[-1,-1,lower_col,upper_col],
        linkVisualShapeIndices=[-1,-1,-1,-1],
        linkPositions=[[0,0,0],[0,0,0],ra_end.tolist(),[0,-BD/2,L1]],
        linkOrientations=[[0,0,0,1]]*4,
        linkInertialFramePositions=[[0,0,0],[0,0,0],[0,-BD/2,L1/2],[0,0,L2/2]],
        linkInertialFrameOrientations=[[0,0,0,1]]*4,
        linkParentIndices=[0,1,2,3],
        linkJointTypes=[p.JOINT_REVOLUTE,p.JOINT_REVOLUTE,p.JOINT_SPHERICAL,p.JOINT_REVOLUTE],
        linkJointAxis=[[0,1,0],[1,0,0],[0,0,0],[0,1,0]])
    PHI_J=0;ROPE_A_J=1;ANKLE=2;HIP=3

    rope_b_id=p.createMultiBody(baseMass=0,basePosition=[0,-HALF_POLE,POLE_H],
        linkMasses=[0.001,ROPE_MASS],linkCollisionShapeIndices=[-1,-1],
        linkVisualShapeIndices=[-1,-1],linkPositions=[[0,0,0],[0,0,0]],
        linkOrientations=[[0,0,0,1]]*2,
        linkInertialFramePositions=[[0,0,0],[0,0,0]],
        linkInertialFrameOrientations=[[0,0,0,1]]*2,
        linkParentIndices=[0,1],
        linkJointTypes=[p.JOINT_REVOLUTE,p.JOINT_REVOLUTE],
        linkJointAxis=[[0,1,0],[1,0,0]])

    con_id=p.createConstraint(rope_b_id,1,robot_id,ANKLE,p.JOINT_POINT2POINT,[0,0,0],
        rb_vec.tolist(),[0,-BD/2,-L1/2])
    p.changeConstraint(con_id, maxForce=maxForce, erp=erp)

    tiny=[0.001]*3
    p.changeDynamics(robot_id,PHI_J,mass=0.001,localInertiaDiagonal=[0.0001]*3,jointDamping=0,linearDamping=0,angularDamping=0)
    p.changeDynamics(robot_id,ROPE_A_J,mass=ROPE_MASS,localInertiaDiagonal=tiny,jointDamping=0,linearDamping=0,angularDamping=0)
    p.changeDynamics(robot_id,ANKLE,mass=M1,localInertiaDiagonal=box_inertia(M1,LOWER_W,BD,L1),jointDamping=0,linearDamping=0,angularDamping=0)
    p.changeDynamics(robot_id,HIP,mass=M2,localInertiaDiagonal=box_inertia(M2,UPPER_W,BD,L2),jointDamping=B_THETA,linearDamping=0,angularDamping=0)
    for j in [0,1]:
        p.changeDynamics(rope_b_id,j,localInertiaDiagonal=tiny,jointDamping=0,linearDamping=0,angularDamping=0)

    for j in [PHI_J,ROPE_A_J,HIP]:
        p.setJointMotorControl2(robot_id,j,p.VELOCITY_CONTROL,force=0)
    p.setJointMotorControlMultiDof(robot_id,ANKLE,p.POSITION_CONTROL,targetPosition=[0,0,0,1],force=[0,0,0])
    for j in [0,1]:
        p.setJointMotorControl2(rope_b_id,j,p.VELOCITY_CONTROL,force=0)

    for body in [robot_id,rope_b_id]:
        nj=p.getNumJoints(body)
        for i in range(-1,nj):
            for j in range(-1,nj):
                if i!=j: p.setCollisionFilterPair(body,body,i,j,0)
            p.setCollisionFilterPair(body,0,i,-1,0)
    for i in range(-1,p.getNumJoints(robot_id)):
        for j in range(-1,p.getNumJoints(rope_b_id)):
            p.setCollisionFilterPair(robot_id,rope_b_id,i,j,0)

    p.resetJointState(robot_id,PHI_J,0,0)
    p.resetJointState(robot_id,ROPE_A_J,0,0)
    p.resetJointStateMultiDof(robot_id,ANKLE,[0,0,0,1],[0,0,0])
    p.resetJointState(robot_id,HIP,THETA0,0)
    p.resetJointState(rope_b_id,0,0,0)
    p.resetJointState(rope_b_id,1,0,0)

    K_lqr = compute_lqr()
    data = []

    for step in range(int(5.0/DT)):
        phi_s=p.getJointState(robot_id,PHI_J)
        phi=phi_s[0]; phi_dot=phi_s[1]
        lo=p.getLinkState(robot_id,ANKLE,computeForwardKinematics=True,computeLinkVelocity=True)
        up=p.getLinkState(robot_id,HIP,computeForwardKinematics=True,computeLinkVelocity=True)
        alpha=p.getEulerFromQuaternion(lo[5])[1]
        theta=p.getEulerFromQuaternion(up[5])[1]
        alpha_dot=lo[7][1]; theta_dot=up[7][1]
        x=np.array([phi,alpha,theta,phi_dot,alpha_dot,theta_dot])

        ankle_q=p.getJointStateMultiDof(robot_id,ANKLE)[0]
        ankle_euler=p.getEulerFromQuaternion(ankle_q)

        tau=0.0
        if abs(x[1])<np.radians(60) and abs(x[2])<np.radians(60):
            tau=-float(K_lqr@x); tau=np.clip(tau,-TAU_MAX,TAU_MAX)
        p.setJointMotorControl2(robot_id,HIP,p.TORQUE_CONTROL,force=tau)
        p.stepSimulation()

        if step%10==0:
            data.append([step*DT, np.degrees(ankle_euler[0]), np.degrees(phi), np.degrees(alpha), np.degrees(theta)])

    p.disconnect()
    data = np.array(data)
    print(f"  [{label}] maxForce={maxForce}, erp={erp} → X-roll@3s={data[300,1]:+.2f}° final={data[-1,1]:+.2f}°")
    return data

# === Run multiple configurations ===
print("Testing P2P constraint parameters...")
K_lqr_cache = compute_lqr()  # pre-compute

configs = [
    (5000,    0.1,  "Original"),
    (50000,   0.1,  "10x Force"),
    (500000,  0.1,  "100x Force"),
    (5000,    0.8,  "High ERP"),
    (50000,   0.8,  "10x Force + High ERP"),
    (500000,  0.9,  "100x Force + Very High ERP"),
    (5000000, 0.95, "1000x Force + Max ERP"),
]

results = {}
for mf, erp, label in configs:
    results[label] = run_sim(mf, erp, label)

# Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("V13 P2P Constraint Stiffness Test", fontsize=14)

for label, data in results.items():
    axes[0].plot(data[:,0], data[:,1], label=label)
axes[0].set_ylabel("Ankle X-roll (deg)")
axes[0].set_title("X-axis rotation (should stay near 0)")
axes[0].legend(fontsize=8); axes[0].grid(True)
axes[0].axhline(y=0, color='k', linewidth=0.5)

for label, data in results.items():
    axes[1].plot(data[:,0], data[:,3], label=label, alpha=0.7)
axes[1].set_ylabel("Alpha (deg)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Y-axis alpha (balance performance)")
axes[1].legend(fontsize=8); axes[1].grid(True)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_p2p_fix.png')
plt.savefig(out, dpi=150)
print(f"\nPlot saved: {out}")

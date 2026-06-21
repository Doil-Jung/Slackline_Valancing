"""Test: Two separate multibodies + P2P between moving bodies."""
import sys, os
_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"): os.add_dll_directory(_lib)

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt

M1=30; M2=40; L1=0.85; L2=0.8; BD=0.25
POLE_H=4.5; HALF_POLE=2.0; ROPE_HORIZ=1.875; ROPE_VERT=1.5
GRAV=9.81; DT=0.001; THETA0=0.15; RMASS=0.01
ra_end = np.array([0, -ROPE_HORIZ, -ROPE_VERT])  # pole_a → ankle_a
rb_vec = np.array([0,  ROPE_HORIZ, -ROPE_VERT])   # pole_b → ankle_b (note: +Y because pole_b is at -Y)
def box_inertia(m,w,d,h): return [m*(d**2+h**2)/12, m*(w**2+h**2)/12, m*(w**2+d**2)/12]

def run_v12(dur=3.0):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-GRAV); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)
    p.loadURDF("plane.urdf")
    robot = p.createMultiBody(
        baseMass=0, basePosition=[0,HALF_POLE,POLE_H],
        linkMasses=[0.001, M1, M2],
        linkCollisionShapeIndices=[-1]*3, linkVisualShapeIndices=[-1]*3,
        linkPositions=[[0,0,0], ra_end.tolist(), [0,-BD/2,L1]],
        linkOrientations=[[0,0,0,1]]*3,
        linkInertialFramePositions=[[0,0,0],[0,-BD/2,L1/2],[0,0,L2/2]],
        linkInertialFrameOrientations=[[0,0,0,1]]*3,
        linkParentIndices=[0,1,2],
        linkJointTypes=[p.JOINT_REVOLUTE]*3,
        linkJointAxis=[[0,1,0]]*3)
    for j in range(3):
        p.changeDynamics(robot,j,linearDamping=0,angularDamping=0,jointDamping=0)
        p.setJointMotorControl2(robot,j,p.VELOCITY_CONTROL,force=0)
    p.changeDynamics(robot,1,mass=M1,localInertiaDiagonal=box_inertia(M1,0.18,BD,L1))
    p.changeDynamics(robot,2,mass=M2,localInertiaDiagonal=box_inertia(M2,0.30,BD,L2))
    p.resetJointState(robot,0,0,0); p.resetJointState(robot,1,0,0)
    p.resetJointState(robot,2,THETA0,0)
    for i in range(-1,3):
        for j in range(-1,3):
            if i!=j: p.setCollisionFilterPair(robot,robot,i,j,0)
        p.setCollisionFilterPair(robot,0,i,-1,0)
    rec=[]
    for step in range(int(dur/DT)):
        phi=p.getJointState(robot,0)[0]
        lo=p.getLinkState(robot,1,computeForwardKinematics=True,computeLinkVelocity=True)
        up=p.getLinkState(robot,2,computeForwardKinematics=True,computeLinkVelocity=True)
        a=p.getEulerFromQuaternion(lo[5])[1]; t=p.getEulerFromQuaternion(up[5])[1]
        if np.any(np.isnan([phi,a,t])): break
        p.stepSimulation()
        if step%10==0: rec.append([step*DT,np.degrees(phi),np.degrees(a),np.degrees(t)])
    p.disconnect()
    return np.array(rec)

def run_two_body(dur=3.0):
    """Two multibodies: robot hangs from pole_a, rope_b hangs from pole_b.
    P2P connects rope_b free end to lower_body ankle_b (both moving)."""
    tiny = [0.001, 0.001, 0.001]

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-GRAV); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)
    p.loadURDF("plane.urdf")

    # --- Multibody 1: Robot (pole_a → rope_a → lower_body → upper_body) ---
    # Universal joint at pole_a: L0=dummy(REVOLUTE Y), L1=rope_a(REVOLUTE X)
    # Spherical at ankle: L2=lower_body(SPHERICAL)
    # Hip: L3=upper_body(REVOLUTE Y)
    robot = p.createMultiBody(
        baseMass=0, basePosition=[0, HALF_POLE, POLE_H],
        linkMasses=[0.001, RMASS, M1, M2],
        linkCollisionShapeIndices=[-1]*4,
        linkVisualShapeIndices=[-1]*4,
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
    # L0=dummy_phiY, L1=rope_a, L2=lower_body, L3=upper_body
    PHI_J = 0; ROPE_A_J = 1; LOWER_LINK = 2; HIP_J = 3

    # --- Multibody 2: Rope_b (pole_b → rope_b) ---
    # Universal joint at pole_b: L0=dummy(REVOLUTE Y), L1=rope_b(REVOLUTE X)
    rope_b_body = p.createMultiBody(
        baseMass=0, basePosition=[0, -HALF_POLE, POLE_H],
        linkMasses=[0.001, RMASS],
        linkCollisionShapeIndices=[-1]*2,
        linkVisualShapeIndices=[-1]*2,
        linkPositions=[[0,0,0], [0,0,0]],
        linkOrientations=[[0,0,0,1]]*2,
        linkInertialFramePositions=[[0,0,0], [0,0,0]],
        linkInertialFrameOrientations=[[0,0,0,1]]*2,
        linkParentIndices=[0, 1],
        linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
        linkJointAxis=[[0,1,0], [1,0,0]],
    )
    RB_DUMMY = 0; RB_ROPE = 1

    # --- Dynamics ---
    # Robot
    p.changeDynamics(robot, 0, mass=0.001, localInertiaDiagonal=[0.0001]*3,
                     jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot, 1, mass=RMASS, localInertiaDiagonal=tiny,
                     jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot, 2, mass=M1,
                     localInertiaDiagonal=box_inertia(M1,0.18,BD,L1),
                     jointDamping=0, linearDamping=0, angularDamping=0)
    p.changeDynamics(robot, 3, mass=M2,
                     localInertiaDiagonal=box_inertia(M2,0.30,BD,L2),
                     jointDamping=0, linearDamping=0, angularDamping=0)
    # Rope_b
    for j in [0,1]:
        p.changeDynamics(rope_b_body, j, localInertiaDiagonal=tiny,
                         jointDamping=0, linearDamping=0, angularDamping=0)

    # Disable motors
    for j in [PHI_J, ROPE_A_J, HIP_J]:
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControlMultiDof(robot, LOWER_LINK, p.POSITION_CONTROL,
                                   targetPosition=[0,0,0,1], force=[0,0,0])
    for j in [RB_DUMMY, RB_ROPE]:
        p.setJointMotorControl2(rope_b_body, j, p.VELOCITY_CONTROL, force=0)

    # --- P2P between TWO MOVING bodies ---
    # rope_b free end (at rb_vec from pole_b) ↔ lower_body ankle_b (at [0,-BD,0] from ankle_a)
    cid = p.createConstraint(
        rope_b_body, RB_ROPE,   # parent: rope_b's rope link
        robot, LOWER_LINK,       # child: lower_body
        p.JOINT_POINT2POINT,
        [0,0,0],
        rb_vec.tolist(),         # rope_b end position in rope_b frame
        [0, -BD, 0]             # ankle_b position in lower_body frame
    )
    p.changeConstraint(cid, maxForce=5000, erp=0.1)

    # Disable collisions
    for body in [robot, rope_b_body]:
        nj = p.getNumJoints(body)
        for i in range(-1, nj):
            for j in range(-1, nj):
                if i!=j: p.setCollisionFilterPair(body,body,i,j,0)
            p.setCollisionFilterPair(body,0,i,-1,0)
    # Also between robot and rope_b
    for i in range(-1, p.getNumJoints(robot)):
        for j in range(-1, p.getNumJoints(rope_b_body)):
            p.setCollisionFilterPair(robot, rope_b_body, i, j, 0)

    # Initial state
    p.resetJointState(robot, PHI_J, 0, 0)
    p.resetJointState(robot, ROPE_A_J, 0, 0)
    p.resetJointStateMultiDof(robot, LOWER_LINK, [0,0,0,1], [0,0,0])
    p.resetJointState(robot, HIP_J, THETA0, 0)
    p.resetJointState(rope_b_body, RB_DUMMY, 0, 0)
    p.resetJointState(rope_b_body, RB_ROPE, 0, 0)

    rec=[]
    for step in range(int(dur/DT)):
        phi = p.getJointState(robot, PHI_J)[0]
        lo = p.getLinkState(robot, LOWER_LINK, computeForwardKinematics=True, computeLinkVelocity=True)
        up = p.getLinkState(robot, HIP_J, computeForwardKinematics=True, computeLinkVelocity=True)
        a = p.getEulerFromQuaternion(lo[5])[1]
        t = p.getEulerFromQuaternion(up[5])[1]
        if np.any(np.isnan([phi,a,t])): print(f"  NaN at {step*DT:.3f}"); break
        p.stepSimulation()
        if step%10==0: rec.append([step*DT, np.degrees(phi), np.degrees(a), np.degrees(t)])
    p.disconnect()
    return np.array(rec)

print("V12 (reference)..."); d12 = run_v12()
print(f"  {len(d12)} samples")
print("Two-body (rope_b separate)..."); d2b = run_two_body()
print(f"  {len(d2b)} samples")

fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
for i,(ax,tt) in enumerate(zip(axes,['phi','alpha','theta'])):
    ax.plot(d12[:,0],d12[:,i+1],'b-',label='V12 (no rope_b)',lw=2)
    ax.plot(d2b[:,0],d2b[:,i+1],'r-',label='Two-body (rope_b separate)',lw=2)
    ax.set_ylabel('deg'); ax.set_title(tt,fontweight='bold')
    ax.legend(); ax.grid(True,alpha=0.3); ax.axhline(0,color='gray',lw=0.5)
axes[-1].set_xlabel('Time (s)')
fig.suptitle('Two-Body Architecture: rope_b as separate multibody',fontsize=14,fontweight='bold')
plt.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'two_body_test.png')
plt.savefig(out,dpi=150); print(f"Saved: {out}")

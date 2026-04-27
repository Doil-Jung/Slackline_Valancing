"""Quick diagnostic: compare V12 (revolute) vs V13 (spherical) angle readings."""
import sys, os
_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"): os.add_dll_directory(_lib)

import numpy as np
import pybullet as p
import pybullet_data

M1=30; M2=40; L1=0.85; L2=0.8; BD=0.25
POLE_H=4.5; HALF_POLE=2.0; ROPE_HORIZ=1.875; ROPE_VERT=1.5
GRAV=9.81; DT=0.001; THETA0=0.15
ra_end = np.array([0, -ROPE_HORIZ, -ROPE_VERT])

def box_inertia(m,w,d,h): return [m*(d**2+h**2)/12, m*(w**2+h**2)/12, m*(w**2+d**2)/12]

# Fixed K from V10
K = np.array([6263.3, -5736.9, -2081.0, 1272.7, -1063.3, -562.2])

def run_v12():
    """V12-style: all revolute, no rope_b physics."""
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-GRAV); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=50)
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
        p.changeDynamics(robot, j, linearDamping=0, angularDamping=0, jointDamping=0)
        p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)
    p.changeDynamics(robot, 1, mass=M1, localInertiaDiagonal=box_inertia(M1,0.18,BD,L1))
    p.changeDynamics(robot, 2, mass=M2, localInertiaDiagonal=box_inertia(M2,0.30,BD,L2))
    
    p.resetJointState(robot, 0, 0, 0)
    p.resetJointState(robot, 1, 0, 0)
    p.resetJointState(robot, 2, THETA0, 0)
    
    rec = []
    for step in range(2000):
        phi = p.getJointState(robot,0)[0]
        phi_dot = p.getJointState(robot,0)[1]
        lo = p.getLinkState(robot,1,computeForwardKinematics=True,computeLinkVelocity=True)
        up = p.getLinkState(robot,2,computeForwardKinematics=True,computeLinkVelocity=True)
        alpha = p.getEulerFromQuaternion(lo[5])[1]
        theta = p.getEulerFromQuaternion(up[5])[1]
        alpha_dot = lo[7][1]; theta_dot = up[7][1]
        x = np.array([phi,alpha,theta,phi_dot,alpha_dot,theta_dot])
        tau = -float(K @ x); tau = np.clip(tau, -1500, 1500)
        p.setJointMotorControl2(robot, 2, p.TORQUE_CONTROL, force=tau)
        p.stepSimulation()
        if step % 10 == 0:
            rec.append([step*DT] + list(np.degrees(x[:3])) + [tau])
    p.disconnect()
    return np.array(rec)

def run_v13():
    """V13-style: spherical joints + rope_b with P2P."""
    rb_end = np.array([0, -ROPE_HORIZ, ROPE_VERT])
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-GRAV); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)
    p.loadURDF("plane.urdf")
    
    robot = p.createMultiBody(
        baseMass=0, basePosition=[0,HALF_POLE,POLE_H],
        linkMasses=[0.01, M1, M2, 0.01],
        linkCollisionShapeIndices=[-1]*4, linkVisualShapeIndices=[-1]*4,
        linkPositions=[[0,0,0], ra_end.tolist(), [0,-BD/2,L1], [0,-BD,0]],
        linkOrientations=[[0,0,0,1]]*4,
        linkInertialFramePositions=[[0,0,0],[0,-BD/2,L1/2],[0,0,L2/2],[0,0,0]],
        linkInertialFrameOrientations=[[0,0,0,1]]*4,
        linkParentIndices=[0,1,2,2],
        linkJointTypes=[p.JOINT_SPHERICAL, p.JOINT_SPHERICAL, p.JOINT_REVOLUTE, p.JOINT_SPHERICAL],
        linkJointAxis=[[0,0,0],[0,0,0],[0,1,0],[0,0,0]])
    
    cid = p.createConstraint(robot,3,-1,-1,p.JOINT_POINT2POINT,[0,0,0],rb_end.tolist(),[0,-HALF_POLE,POLE_H])
    p.changeConstraint(cid, maxForce=50000, erp=0.8)
    
    p.changeDynamics(robot,1,mass=M1,localInertiaDiagonal=box_inertia(M1,0.18,BD,L1),
                     jointDamping=0,linearDamping=0,angularDamping=0)
    p.changeDynamics(robot,2,mass=M2,localInertiaDiagonal=box_inertia(M2,0.30,BD,L2),
                     jointDamping=3.0,linearDamping=0,angularDamping=0)
    for j in [0,1,3]:
        p.changeDynamics(robot,j,jointDamping=0,localInertiaDiagonal=[0.1,0.1,0.1],
                         linearDamping=0,angularDamping=0)
    
    p.setJointMotorControl2(robot, 2, p.VELOCITY_CONTROL, force=0)
    for sj in [0,1,3]:
        p.setJointMotorControlMultiDof(robot,sj,p.POSITION_CONTROL,
                                       targetPosition=[0,0,0,1],force=[0,0,0])
    
    p.resetJointStateMultiDof(robot,0,[0,0,0,1],[0,0,0])
    p.resetJointStateMultiDof(robot,1,[0,0,0,1],[0,0,0])
    p.resetJointState(robot,2,THETA0,0)
    p.resetJointStateMultiDof(robot,3,[0,0,0,1],[0,0,0])
    
    rec = []
    for step in range(2000):
        ra = p.getJointStateMultiDof(robot,0)
        phi = p.getEulerFromQuaternion(ra[0])[1]
        phi_dot = ra[1][1]
        lo = p.getLinkState(robot,1,computeForwardKinematics=True,computeLinkVelocity=True)
        up = p.getLinkState(robot,2,computeForwardKinematics=True,computeLinkVelocity=True)
        alpha = p.getEulerFromQuaternion(lo[5])[1]
        theta = p.getEulerFromQuaternion(up[5])[1]
        alpha_dot = lo[7][1]; theta_dot = up[7][1]
        x = np.array([phi,alpha,theta,phi_dot,alpha_dot,theta_dot])
        tau = -float(K @ x); tau = np.clip(tau, -1500, 1500)
        p.setJointMotorControl2(robot, 2, p.TORQUE_CONTROL, force=tau)
        p.stepSimulation()
        if step % 10 == 0:
            rec.append([step*DT] + list(np.degrees(x[:3])) + [tau])
    p.disconnect()
    return np.array(rec)

import matplotlib.pyplot as plt

print("Running V12 (revolute)...")
d12 = run_v12()
print("Running V13 (spherical+P2P)...")
d13 = run_v13()

fig, axes = plt.subplots(4,1,figsize=(12,10),sharex=True)
labels = ['phi','alpha','theta','tau']
for i, (ax, lb) in enumerate(zip(axes, labels)):
    ax.plot(d12[:,0], d12[:,i+1], 'b-', label='V12 revolute', alpha=0.8)
    ax.plot(d13[:,0], d13[:,i+1], 'r-', label='V13 spherical', alpha=0.8)
    ax.set_ylabel(lb); ax.legend(); ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.5)
axes[-1].set_xlabel('Time (s)')
fig.suptitle('V12 vs V13: Same K, Same Initial Conditions', fontsize=14, fontweight='bold')
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v12_vs_v13_compare.png')
plt.savefig(out, dpi=150); print(f"Saved: {out}")

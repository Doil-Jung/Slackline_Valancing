"""Test: Universal joints (2x revolute) vs Spherical vs V12 for phi freedom."""
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
GRAV=9.81; DT=0.001; THETA0=0.15
ra_end = np.array([0, -ROPE_HORIZ, -ROPE_VERT])
rb_end = np.array([0, -ROPE_HORIZ,  ROPE_VERT])
def box_inertia(m,w,d,h): return [m*(d**2+h**2)/12, m*(w**2+h**2)/12, m*(w**2+d**2)/12]

def run_v12(dur=3.0):
    """V12: 3 revolute, no rope_b"""
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

def run_universal(dur=3.0):
    """Universal joints: 2x revolute (Y then X) at each rope connection.
    
    Tree structure (8 links):
    Base (pole_a)
    → L0: dummy_phiY  (REVOLUTE Y, zero mass) - phi main swing
    → L1: rope_a      (REVOLUTE X, rope mass)  - rope tilt
    → L2: dummy_ankleY(REVOLUTE Y, zero mass) - ankle Y
    → L3: lower_body  (REVOLUTE X, M1)         - ankle X
    → L4: upper_body  (REVOLUTE Y, M2)         - hip
    Branch from L3:
    → L5: dummy_rbY   (REVOLUTE Y, zero mass, parent=3) - rope_b ankle Y
    → L6: rope_b      (REVOLUTE X, rope mass, parent=5) - rope_b tilt
    P2P: L6 end → pole_b
    """
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-GRAV); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)
    p.loadURDF("plane.urdf")

    RMASS = 0.01
    tiny = [0.001, 0.001, 0.001]
    zero_i = [0.0001, 0.0001, 0.0001]  # dummy links

    robot = p.createMultiBody(
        baseMass=0,
        basePosition=[0, HALF_POLE, POLE_H],
        linkMasses=[
            0.001,    # L0: dummy_phiY
            RMASS,    # L1: rope_a
            0.001,    # L2: dummy_ankleY
            M1,       # L3: lower_body
            M2,       # L4: upper_body (hip)
            0.001,    # L5: dummy_rbY
            RMASS,    # L6: rope_b
        ],
        linkCollisionShapeIndices=[-1]*7,
        linkVisualShapeIndices=[-1]*7,
        linkPositions=[
            [0, 0, 0],           # L0: dummy at pole_a (no offset)
            [0, 0, 0],           # L1: rope_a starts at pole_a
            ra_end.tolist(),     # L2: dummy at ankle_a (end of rope_a)
            [0, 0, 0],          # L3: lower_body at ankle_a
            [0, -BD/2, L1],      # L4: upper_body at hip
            [0, -BD, 0],        # L5: dummy at ankle_b (parent=L3)
            [0, 0, 0],           # L6: rope_b starts at ankle_b
        ],
        linkOrientations=[[0,0,0,1]]*7,
        linkInertialFramePositions=[
            [0,0,0],             # L0
            [0,0,0],             # L1: rope CoM at joint
            [0,0,0],             # L2
            [0,-BD/2,L1/2],      # L3: lower body CoM
            [0,0,L2/2],          # L4: upper body CoM
            [0,0,0],             # L5
            [0,0,0],             # L6: rope CoM at joint
        ],
        linkInertialFrameOrientations=[[0,0,0,1]]*7,
        linkParentIndices=[0, 1, 2, 3, 4, 4, 6],  # L5→L4? No.
        # Wait, need to fix parent indices:
        # L0 parent=base(0), L1 parent=L0(1), L2 parent=L1(2), L3 parent=L2(3)
        # L4 parent=L3(4), L5 parent=L3(4)... 
        # Actually parentIndices use 0-based link index, not 1-based
        # base = implicit, L0=link0, L1=link1, ...
        # L0 parent = 0 (base)
        # L1 parent = 0+1? No. parentIndex 0 = base, 1 = link0, 2=link1...
        # Wait, in PyBullet createMultiBody:
        # parentIndex 0 = base
        # parentIndex 1 = link 0
        # parentIndex 2 = link 1, etc.
        # So: L0→base(0), L1→L0(1), L2→L1(2), L3→L2(3), L4→L3(4), L5→L3(4), L6→L5(6)
        linkJointTypes=[
            p.JOINT_REVOLUTE,  # L0: phi Y
            p.JOINT_REVOLUTE,  # L1: rope_a tilt X
            p.JOINT_REVOLUTE,  # L2: ankle Y
            p.JOINT_REVOLUTE,  # L3: ankle X
            p.JOINT_REVOLUTE,  # L4: hip Y
            p.JOINT_REVOLUTE,  # L5: rope_b ankle Y
            p.JOINT_REVOLUTE,  # L6: rope_b tilt X
        ],
        linkJointAxis=[
            [0,1,0],  # L0: Y
            [1,0,0],  # L1: X
            [0,1,0],  # L2: Y
            [1,0,0],  # L3: X
            [0,1,0],  # L4: Y (hip)
            [0,1,0],  # L5: Y
            [1,0,0],  # L6: X
        ],
    )
    # Fix parent indices
    pass  # Already set above - but let me redo this properly
    p.disconnect()

    # --- Rebuild with correct parent indices ---
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-GRAV); p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=100)
    p.loadURDF("plane.urdf")

    robot = p.createMultiBody(
        baseMass=0,
        basePosition=[0, HALF_POLE, POLE_H],
        linkMasses=[0.001, RMASS, 0.001, M1, M2, 0.001, RMASS],
        linkCollisionShapeIndices=[-1]*7,
        linkVisualShapeIndices=[-1]*7,
        linkPositions=[
            [0,0,0],            # L0: dummy at base
            [0,0,0],            # L1: rope_a at same point
            ra_end.tolist(),    # L2: dummy at ankle (end of rope_a)
            [0,0,0],            # L3: lower_body at ankle
            [0,-BD/2,L1],       # L4: upper_body at hip
            [0,-BD,0],          # L5: dummy at ankle_b (from lower_body)
            [0,0,0],            # L6: rope_b at ankle_b
        ],
        linkOrientations=[[0,0,0,1]]*7,
        linkInertialFramePositions=[
            [0,0,0], [0,0,0], [0,0,0],
            [0,-BD/2,L1/2], [0,0,L2/2],
            [0,0,0], [0,0,0],
        ],
        linkInertialFrameOrientations=[[0,0,0,1]]*7,
        linkParentIndices=[0, 1, 2, 3, 4, 4, 6],
        linkJointTypes=[p.JOINT_REVOLUTE]*7,
        linkJointAxis=[
            [0,1,0], [1,0,0],  # L0,L1: universal at pole_a
            [0,1,0], [1,0,0],  # L2,L3: universal at ankle_a
            [0,1,0],            # L4: hip
            [0,1,0], [1,0,0],  # L5,L6: universal at ankle_b
        ],
    )

    # Dynamics
    for j in range(7):
        p.setJointMotorControl2(robot,j,p.VELOCITY_CONTROL,force=0)
    for j in [0,2,5]:  # dummy links
        p.changeDynamics(robot,j,mass=0.001,localInertiaDiagonal=zero_i,
                         jointDamping=0,linearDamping=0,angularDamping=0)
    for j in [1,6]:  # rope links
        p.changeDynamics(robot,j,mass=RMASS,localInertiaDiagonal=tiny,
                         jointDamping=0,linearDamping=0,angularDamping=0)
    p.changeDynamics(robot,3,mass=M1,localInertiaDiagonal=box_inertia(M1,0.18,BD,L1),
                     jointDamping=0,linearDamping=0,angularDamping=0)
    p.changeDynamics(robot,4,mass=M2,localInertiaDiagonal=box_inertia(M2,0.30,BD,L2),
                     jointDamping=0,linearDamping=0,angularDamping=0)

    # P2P for rope_b end
    cid = p.createConstraint(robot,6,-1,-1,p.JOINT_POINT2POINT,
                              [0,0,0], rb_end.tolist(), [0,-HALF_POLE,POLE_H])
    p.changeConstraint(cid, maxForce=5000, erp=0.1)

    # Initial state
    for j in range(7):
        p.resetJointState(robot,j,0,0)
    p.resetJointState(robot,4,THETA0,0)  # hip = theta0

    # Disable collisions
    for i in range(-1,7):
        for j in range(-1,7):
            if i!=j: p.setCollisionFilterPair(robot,robot,i,j,0)
        p.setCollisionFilterPair(robot,0,i,-1,0)

    PHI_J = 0  # phi = joint 0 (revolute Y at pole_a)
    ANKLE_LINK = 3  # lower body
    HIP_LINK = 4    # upper body

    rec=[]
    for step in range(int(dur/DT)):
        phi = p.getJointState(robot,PHI_J)[0]
        lo = p.getLinkState(robot,ANKLE_LINK,computeForwardKinematics=True,computeLinkVelocity=True)
        up = p.getLinkState(robot,HIP_LINK,computeForwardKinematics=True,computeLinkVelocity=True)
        a = p.getEulerFromQuaternion(lo[5])[1]
        t = p.getEulerFromQuaternion(up[5])[1]
        if np.any(np.isnan([phi,a,t])): print(f"  NaN at {step*DT:.3f}"); break
        p.stepSimulation()
        if step%10==0: rec.append([step*DT,np.degrees(phi),np.degrees(a),np.degrees(t)])
    p.disconnect()
    return np.array(rec)

print("V12 (revolute, no rope_b)...")
d12 = run_v12()
print(f"  {len(d12)} samples")

print("Universal joints (2x revolute per connection)...")
d_uni = run_universal()
print(f"  {len(d_uni)} samples")

fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
titles=['phi (rope swing)','alpha (lower body)','theta (upper body)']
for i,(ax,tt) in enumerate(zip(axes,titles)):
    ax.plot(d12[:,0],d12[:,i+1],'b-',label='V12 revolute (no rope_b)',lw=2)
    ax.plot(d_uni[:,0],d_uni[:,i+1],'r-',label='Universal (2x revolute)',lw=2)
    ax.set_ylabel('deg'); ax.set_title(tt,fontweight='bold')
    ax.legend(); ax.grid(True,alpha=0.3); ax.axhline(0,color='gray',lw=0.5)
axes[-1].set_xlabel('Time (s)')
fig.suptitle('V12 vs Universal Joint (free fall)',fontsize=14,fontweight='bold')
plt.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'universal_test.png')
plt.savefig(out,dpi=150); print(f"Saved: {out}")

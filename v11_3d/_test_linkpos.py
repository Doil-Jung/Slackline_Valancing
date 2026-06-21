"""Check joint axes and rotation behavior"""
import os, sys
_conda = os.path.dirname(sys.executable)
_lib = os.path.join(_conda, "Library", "bin")
if os.path.isdir(_lib):
    os.environ["PATH"] = _lib + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"): os.add_dll_directory(_lib)

import pybullet as p
import numpy as np

cid = p.connect(p.DIRECT)
p.setGravity(0, 0, 0)  # no gravity for clean analysis

BD = 0.25
L1 = 0.85
ra_end = np.array([0, -1.875, -1.5])
rb_end = np.array([0, -1.875,  1.5])

robot = p.createMultiBody(
    baseMass=0,
    basePosition=[0, 2, 4.5],
    linkMasses=[0.001, 30.0, 40.0, 0.001, 0.001],
    linkCollisionShapeIndices=[-1]*5,
    linkVisualShapeIndices=[-1]*5,
    linkPositions=[
        [0, 0, 0],
        ra_end.tolist(),
        [0, -BD/2, L1],
        [0, -BD, 0],
        [0, -BD/2, L1],
    ],
    linkOrientations=[[0,0,0,1]]*5,
    linkInertialFramePositions=[
        [0,0,0], [0,-BD/2,L1/2], [0,0,0.4], [0,0,0], [0,0,0]
    ],
    linkInertialFrameOrientations=[[0,0,0,1]]*5,
    linkParentIndices=[0, 1, 2, 2, 2],
    linkJointTypes=[p.JOINT_REVOLUTE]*4 + [p.JOINT_FIXED],
    linkJointAxis=[[0,1,0]]*5,
)

ROPE_A, ANKLE, HIP, ROPE_B = 0, 1, 2, 3

# Disable all motors
for j in range(5):
    p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

# Print joint info
for i in range(4):
    ji = p.getJointInfo(robot, i)
    print(f"Joint {i} ({ji[1].decode()}): axis={ji[13]}, "
          f"parentIdx={ji[16]}, parentFramePos={ji[14]}")

# Test 1: Set ankle=10deg, check if ankle_b moves in XZ
print("\n--- Test: ankle=10deg, all others=0 ---")
p.resetJointState(robot, ANKLE, np.radians(10), 0)
for i in range(4):
    ls = p.getLinkState(robot, i, computeForwardKinematics=True)
    print(f"  Link {i}: joint_world={tuple(round(x,4) for x in ls[4])}")

# Test 2: Set rope_b=10deg, check tip position
print("\n--- Test: rope_b=10deg, all others=0 ---")
p.resetJointState(robot, ANKLE, 0, 0)
p.resetJointState(robot, ROPE_B, np.radians(10), 0)
ls_rb = p.getLinkState(robot, ROPE_B, computeForwardKinematics=True)
jpos = np.array(ls_rb[4])
jorn = ls_rb[5]
R = np.array(p.getMatrixFromQuaternion(jorn)).reshape(3,3)
tip = jpos + R @ rb_end
print(f"  rope_b joint={tuple(round(x,4) for x in jpos)}")
print(f"  rope_b tip  ={tuple(round(x,4) for x in tip)}")

# Test 3: Set ankle=10deg + rope_b=-10deg (should compensate)
print("\n--- Test: ankle=10deg, rope_b=-10deg ---")
p.resetJointState(robot, ANKLE, np.radians(10), 0)
p.resetJointState(robot, ROPE_B, np.radians(-10), 0)
ls_rb = p.getLinkState(robot, ROPE_B, computeForwardKinematics=True)
jpos = np.array(ls_rb[4])
jorn = ls_rb[5]
R = np.array(p.getMatrixFromQuaternion(jorn)).reshape(3,3)
tip = jpos + R @ rb_end
print(f"  rope_b joint={tuple(round(x,4) for x in jpos)}")
print(f"  rope_b tip  ={tuple(round(x,4) for x in tip)}")
print(f"  pole_b_top  =(0, -2.0, 4.5)")
print(f"  tip error   ={tuple(round(tip[k]-[0,-2,4.5][k],4) for k in range(3))}")

# Test 4: Jacobian check - ∂tip/∂alpha vs ∂tip/∂beta
print("\n--- Jacobian comparison ---")
eps = 0.001
for joint_name, joint_idx in [("ankle", ANKLE), ("rope_b", ROPE_B)]:
    p.resetJointState(robot, ANKLE, 0, 0)
    p.resetJointState(robot, ROPE_B, 0, 0)
    # tip at q=0
    ls0 = p.getLinkState(robot, ROPE_B, computeForwardKinematics=True)
    R0 = np.array(p.getMatrixFromQuaternion(ls0[5])).reshape(3,3)
    tip0 = np.array(ls0[4]) + R0 @ rb_end
    # tip at q=eps
    p.resetJointState(robot, joint_idx, eps, 0)
    ls1 = p.getLinkState(robot, ROPE_B, computeForwardKinematics=True)
    R1 = np.array(p.getMatrixFromQuaternion(ls1[5])).reshape(3,3)
    tip1 = np.array(ls1[4]) + R1 @ rb_end
    dtip = (tip1 - tip0) / eps
    print(f"  ∂tip/∂{joint_name:6s} = [{dtip[0]:+.4f}, {dtip[1]:+.4f}, {dtip[2]:+.4f}]")

p.disconnect()

"""Diagnostic 6: Clean test - does ankle joint rotation show in xquat?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import numpy as np

# ========== Test 1: Minimal model (sanity check) ==========
print("=" * 60)
print("TEST 1: Minimal model - single hinge joint")
print("=" * 60)

minimal_xml = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="parent" pos="0 0 2">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.1 0.1 0.5" mass="10"/>
      <inertial pos="0 0 0.25" mass="10" diaginertia="1 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""
m1 = mujoco.MjModel.from_xml_string(minimal_xml)
d1 = mujoco.MjData(m1)

# Set joint to 15 degrees
d1.qpos[0] = np.radians(15)
mujoco.mj_forward(m1, d1)

bid = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_BODY, 'parent')
q = d1.xquat[bid]
print(f"  qpos = {d1.qpos[0]:.4f} rad ({np.degrees(d1.qpos[0]):.1f} deg)")
print(f"  xquat = [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")

# Manual euler extraction
w, x, y, z = q
pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
print(f"  pitch from xquat = {np.degrees(pitch):.2f} deg  (expect 15.00)")

# ========== Test 2: Full slackline model ==========
print()
print("=" * 60)
print("TEST 2: Full slackline model")
print("=" * 60)

from config import V11Config
from sim.world import build_mjcf

config = V11Config()
config.robot.alpha0 = np.radians(15)
config.robot.theta0 = np.radians(20)

xml = build_mjcf(config)
m2 = mujoco.MjModel.from_xml_string(xml)
d2 = mujoco.MjData(m2)

# Print all joint info
print("\n  Joint map:")
for i in range(m2.njnt):
    name = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_JOINT, i)
    qa = m2.jnt_qposadr[i]
    print(f"    [{i}] '{name}' -> qposadr={qa}")

ankle_id = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, 'ankle')
hip_id = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, 'hip')
aq = m2.jnt_qposadr[ankle_id]
hq = m2.jnt_qposadr[hip_id]

lb_id = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, 'lower_body')

# Test A: all qpos = 0
print("\n  --- Test A: all qpos = 0 ---")
d2.qpos[:] = 0
mujoco.mj_forward(m2, d2)
q = d2.xquat[lb_id]
pitch = np.degrees(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
print(f"  qpos = {list(d2.qpos)}")
print(f"  lower_body xquat = {[f'{v:.6f}' for v in q]}")
print(f"  lower_body pitch = {pitch:.2f} deg  (expect 0.00)")

# Test B: ankle = 15 deg only
print("\n  --- Test B: ankle = 15 deg ---")
d2.qpos[:] = 0
d2.qpos[aq] = np.radians(15)
mujoco.mj_forward(m2, d2)
q = d2.xquat[lb_id]
pitch = np.degrees(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
print(f"  qpos = {list(d2.qpos)}")
print(f"  lower_body xquat = {[f'{v:.6f}' for v in q]}")
print(f"  lower_body pitch = {pitch:.2f} deg  (expect ~15.00)")
print(f"  lower_body xpos  = {d2.xpos[lb_id]}")

# Test C: check if mj_kinematics gives different result
print("\n  --- Test C: mj_kinematics directly ---")
d2.qpos[:] = 0
d2.qpos[aq] = np.radians(15)
mujoco.mj_kinematics(m2, d2)
q = d2.xquat[lb_id]
pitch = np.degrees(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
print(f"  lower_body xquat = {[f'{v:.6f}' for v in q]}")
print(f"  lower_body pitch = {pitch:.2f} deg  (expect ~15.00)")

# Test D: ra_y (rope_a y-axis hinge) which is same axis as ankle
ra_y_id = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, 'ra_y')
ra_y_qa = m2.jnt_qposadr[ra_y_id]
print(f"\n  --- Test D: rope_a y-hinge vs ankle ---")
print(f"  ra_y qposadr = {ra_y_qa}")
print(f"  ankle qposadr = {aq}")
print(f"  NOTE: ankle axis = {m2.jnt_axis[ankle_id]}")
print(f"  NOTE: ra_y axis = {m2.jnt_axis[ra_y_id]}")

# Could ankle be the SAME rotation as ra_y, making them cancel?
# rope_a has joints [ra_x, ra_y, ra_z] giving 3 DOF rotation
# Then lower_body has ankle (y-hinge) giving 1 DOF rotation
# ra_y and ankle are BOTH y-axis hinges on the same kinematic chain
# If ra_y = -alpha and ankle = +alpha, they cancel out!
print("\n  --- Test E: Check if mj_step adjusts ra_y to cancel ankle ---")
d2.qpos[:] = 0
d2.qpos[aq] = np.radians(15)
d2.qvel[:] = 0
mujoco.mj_forward(m2, d2)
print(f"  Before steps: qpos = {[f'{v:.4f}' for v in d2.qpos]}")

for _ in range(10):
    mujoco.mj_step(m2, d2)
print(f"  After 10 steps: qpos = {[f'{v:.4f}' for v in d2.qpos]}")
print(f"  ra_y changed? ra_y = {np.degrees(d2.qpos[ra_y_qa]):.4f} deg")
print(f"  ankle changed? ankle = {np.degrees(d2.qpos[aq]):.4f} deg")

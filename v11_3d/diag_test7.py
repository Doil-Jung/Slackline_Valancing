"""Diagnostic 7: Does launch_passive reset data.qpos?
This test checks if qpos values survive the launch_passive call."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
import mujoco.viewer
import numpy as np
from config import V11Config
from sim.world import build_mjcf

config = V11Config()
config.robot.alpha0 = np.radians(15)
config.robot.theta0 = np.radians(20)

xml = build_mjcf(config)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'ankle')
hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip')
aq = model.jnt_qposadr[ankle_id]
hq = model.jnt_qposadr[hip_id]

# Set initial angles
data.qpos[aq] = np.radians(15)
data.qpos[hq] = np.radians(5)  # theta - alpha
model.qpos0[aq] = np.radians(15)
model.qpos0[hq] = np.radians(5)

mujoco.mj_forward(model, data)

print(f"BEFORE launch_passive:")
print(f"  data.qpos = {[f'{v:.4f}' for v in data.qpos]}")
print(f"  model.qpos0 = {[f'{v:.4f}' for v in model.qpos0]}")

lb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lower_body')
q = data.xquat[lb_id]
pitch = np.degrees(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
print(f"  lower_body pitch = {pitch:.2f} deg")

# Open viewer
print(f"\nOpening viewer...")
viewer = mujoco.viewer.launch_passive(model, data)

print(f"\nAFTER launch_passive:")
print(f"  data.qpos = {[f'{v:.4f}' for v in data.qpos]}")
print(f"  data.time = {data.time}")
q = data.xquat[lb_id]
pitch = np.degrees(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
print(f"  lower_body pitch = {pitch:.2f} deg")

# Step a few times and sync
import time
print(f"\nStepping and syncing...")
for i in range(100):
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.001)

print(f"\nAFTER 100 steps:")
print(f"  data.qpos = {[f'{v:.4f}' for v in data.qpos]}")
print(f"  data.time = {data.time}")
q = data.xquat[lb_id]
pitch = np.degrees(np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1)))
print(f"  lower_body pitch = {pitch:.2f} deg")
print(f"  ankle = {np.degrees(data.qpos[aq]):.2f} deg")

viewer.close()
print("Done.")

"""Diagnostic 5: Set rope_a joint too, see if robot actually tilts and falls"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco, numpy as np
from config import V11Config
from sim.world import SlacklineWorld, quat_to_euler

config = V11Config()
config.robot.alpha0 = np.radians(15)
config.robot.theta0 = np.radians(20)
world = SlacklineWorld(config)
model, data = world.initialize()

# Joint map
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    print(f"  jnt[{i}] '{name}' axis={model.jnt_axis[i]}")

# The rolling axis is x (axis=[1,0,0]) = ra_x
# Set rope_a's x-hinge to alpha0, so the whole rope+robot tilts
ra_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'ra_x')
ra_x_qa = model.jnt_qposadr[ra_x_id]

# Reset all to 0 first
data.qpos[:] = 0
# Set rope_a x-rotation to alpha0 -> this tilts the entire chain
data.qpos[ra_x_qa] = np.radians(15)
# ankle: additional rotation relative to rope_a (can be 0 for now)
# hip: theta0 - alpha0 (upper body relative to lower)
aq = model.jnt_qposadr[world.ankle_jnt_id]
hq = model.jnt_qposadr[world.hip_jnt_id]
data.qpos[hq] = np.radians(20 - 15)  # hip_rel = theta - alpha

mujoco.mj_forward(model, data)

lb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lower_body')
ub_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'upper_body')
lb_euler = quat_to_euler(data.xquat[lb_id])
ub_euler = quat_to_euler(data.xquat[ub_id])

print(f"\n=== With ra_x={np.degrees(data.qpos[ra_x_qa]):.1f} ankle=0 hip=5 ===")
print(f"lower_body pitch = {np.degrees(lb_euler[1]):.2f} deg")
print(f"upper_body pitch = {np.degrees(ub_euler[1]):.2f} deg")
print(f"lower_body xpos  = {data.xpos[lb_id]}")

# Now step and see if it moves
print(f"\nqpos before step: {[f'{v:.4f}' for v in data.qpos]}")
for i in range(100):
    mujoco.mj_step(model, data)
print(f"qpos after 100 steps: {[f'{v:.4f}' for v in data.qpos]}")
print(f"qvel after 100 steps: {[f'{v:.6f}' for v in data.qvel]}")

for i in range(900):
    mujoco.mj_step(model, data)
print(f"qpos after 1000 steps (t={data.time:.3f}s): {[f'{v:.4f}' for v in data.qpos]}")

print("\n=== SUCCESS if ankle/hip qpos changed significantly ===")

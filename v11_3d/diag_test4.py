"""Diagnostic 4: Check - does setting ankle qpos actually tilt the robot visually?
The ankle joint is INSIDE rope_a body. So ankle rotation is LOCAL to rope_a.
If rope_a is at 0,0,0 rotation, then ankle=15deg should tilt the lower_body.
But does gravity acting on the tilted body transfer torque THROUGH ankle to rope_a?

Let's check the body orientations (xquat/xmat) to see if the robot LOOKS tilted.
"""
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

lb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lower_body')
ub_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'upper_body')
ra_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rope_a')

print("=== Initial state ===")
ra_euler = quat_to_euler(data.xquat[ra_id])
lb_euler = quat_to_euler(data.xquat[lb_id])
ub_euler = quat_to_euler(data.xquat[ub_id])
print(f"rope_a orientation (euler deg): roll={np.degrees(ra_euler[0]):.2f} pitch={np.degrees(ra_euler[1]):.2f} yaw={np.degrees(ra_euler[2]):.2f}")
print(f"lower_body orient (euler deg):  roll={np.degrees(lb_euler[0]):.2f} pitch={np.degrees(lb_euler[1]):.2f} yaw={np.degrees(lb_euler[2]):.2f}")
print(f"upper_body orient (euler deg):  roll={np.degrees(ub_euler[0]):.2f} pitch={np.degrees(ub_euler[1]):.2f} yaw={np.degrees(ub_euler[2]):.2f}")
print(f"lower_body xpos: {data.xpos[lb_id]}")
print(f"upper_body xpos: {data.xpos[ub_id]}")
print(f"rope_a xpos:     {data.xpos[ra_id]}")

# The ankle joint axis is (0,1,0) = y-axis = rolling direction
# So ankle rotation should tilt the lower_body about y-axis
# In euler terms, that's 'pitch' (second component)
# 15 degrees of ankle tilt SHOULD show as ~15 deg pitch on lower_body

print("\n=== Comparison: ankle=0 vs ankle=15 ===")
# Reset to 0
aq = model.jnt_qposadr[world.ankle_jnt_id]
data.qpos[aq] = 0
mujoco.mj_forward(model, data)
lb_euler_0 = quat_to_euler(data.xquat[lb_id])
print(f"ankle=0:  lower_body pitch = {np.degrees(lb_euler_0[1]):.2f} deg, xpos={data.xpos[lb_id]}")

data.qpos[aq] = np.radians(15)
mujoco.mj_forward(model, data)
lb_euler_15 = quat_to_euler(data.xquat[lb_id])
print(f"ankle=15: lower_body pitch = {np.degrees(lb_euler_15[1]):.2f} deg, xpos={data.xpos[lb_id]}")

print(f"\nDifference in pitch: {np.degrees(lb_euler_15[1] - lb_euler_0[1]):.2f} deg")
print("(If this is 15 deg, the robot IS visually tilted)")
print("(If this is 0 deg, something is absorbing the rotation)")

# Now also check: what does the VIEWER see?
# The constraint connect with solref is pulling rope_b tip to pole_b
# With ankle tilted, the kinematic chain changes, and the constraint
# solver might be compensating

print("\n=== Energy check ===")
data.qpos[aq] = np.radians(15)
mujoco.mj_forward(model, data)
E0 = data.energy[0] + data.energy[1]  # potential + kinetic
print(f"E with ankle=15: PE={data.energy[0]:.4f} KE={data.energy[1]:.10f} Total={E0:.4f}")

data.qpos[aq] = 0
mujoco.mj_forward(model, data)
E1 = data.energy[0] + data.energy[1]
print(f"E with ankle=0:  PE={data.energy[0]:.4f} KE={data.energy[1]:.10f} Total={E1:.4f}")
print(f"PE difference: {data.energy[0] - E0:.4f} (should be >0 if 15deg raises CoM)")

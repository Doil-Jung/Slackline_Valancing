"""Diagnostic 3: deep physics check"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco, numpy as np
from config import V11Config
from sim.world import SlacklineWorld

config = V11Config()
config.robot.alpha0 = np.radians(15)
config.robot.theta0 = np.radians(20)
world = SlacklineWorld(config)
model, data = world.initialize()

aq = model.jnt_qposadr[world.ankle_jnt_id]
hq = model.jnt_qposadr[world.hip_jnt_id]

print(f"Full initial qpos = {[f'{v:.4f}' for v in data.qpos]}")
print(f"Joint names and addresses:")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qa = model.jnt_qposadr[i]
    da = model.jnt_dofadr[i]
    print(f"  jnt[{i}] '{name}': qposadr={qa}, dofadr={da}, qpos={data.qpos[qa]:.4f}")

# Check body positions
lb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lower_body')
ub_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'upper_body')
print(f"\nlower_body xpos = {data.xpos[lb_id]}")
print(f"upper_body xpos = {data.xpos[ub_id]}")

# 1 step
mujoco.mj_step(model, data)
print(f"\nAfter 1 step:")
print(f"  qvel = {[f'{v:.8f}' for v in data.qvel]}")
print(f"  qpos = {[f'{v:.6f}' for v in data.qpos]}")

# 100 steps
for _ in range(99):
    mujoco.mj_step(model, data)
print(f"\nAfter 100 steps (t={data.time:.3f}s):")
print(f"  qpos = {[f'{v:.6f}' for v in data.qpos]}")
print(f"  qvel = {[f'{v:.6f}' for v in data.qvel]}")

# 1000 steps
for _ in range(900):
    mujoco.mj_step(model, data)
print(f"\nAfter 1000 steps (t={data.time:.3f}s):")
print(f"  qpos = {[f'{v:.6f}' for v in data.qpos]}")

# Constraint info
print(f"\nnefc (active constraints) = {data.nefc}")
print(f"gravity = {model.opt.gravity}")
print(f"timestep = {model.opt.timestep}")

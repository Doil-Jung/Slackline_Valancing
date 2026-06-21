"""Diagnostic: test launch_passive + mj_step without Tkinter"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import mujoco.viewer
import numpy as np
from config import V11Config
from sim.world import SlacklineWorld

config = V11Config()
config.robot.alpha0 = 0.0
config.robot.theta0 = np.radians(8.6)

print(f"[Diag] alpha0={np.degrees(config.robot.alpha0):.1f} deg")
print(f"[Diag] theta0={np.degrees(config.robot.theta0):.1f} deg")

world = SlacklineWorld(config)
model, data = world.initialize()

aq = model.jnt_qposadr[world.ankle_jnt_id]
hq = model.jnt_qposadr[world.hip_jnt_id]

print(f"[Diag] After init: ankle_qpos={data.qpos[aq]:.4f}, hip_qpos={data.qpos[hq]:.4f}")
print(f"[Diag] qpos0:      ankle_qpos0={model.qpos0[aq]:.4f}, hip_qpos0={model.qpos0[hq]:.4f}")
print(f"[Diag] Full qpos = {list(data.qpos)}")

print("[Diag] Opening viewer...")
viewer = mujoco.viewer.launch_passive(model, data)

# Check if viewer changed qpos
print(f"[Diag] After viewer open: ankle_qpos={data.qpos[aq]:.4f}, hip_qpos={data.qpos[hq]:.4f}")
print(f"[Diag] viewer.is_running() = {viewer.is_running()}")

step_count = 0
try:
    while viewer.is_running() and step_count < 3000:  # ~3 seconds then stop
        mujoco.mj_step(model, data)
        viewer.sync()
        step_count += 1
        if step_count % 500 == 0:
            print(f"[Diag] step={step_count}, t={data.time:.3f}s, "
                  f"ankle={np.degrees(data.qpos[aq]):+.2f}deg, "
                  f"hip={np.degrees(data.qpos[hq]):+.2f}deg")
        time.sleep(0.001)
except Exception as e:
    print(f"[Diag] ERROR in loop: {e}")
    import traceback
    traceback.print_exc()

print(f"[Diag] Loop ended. Total steps={step_count}")
print(f"[Diag] viewer.is_running() = {viewer.is_running()}")
try:
    viewer.close()
except:
    pass
print("[Diag] Done.")

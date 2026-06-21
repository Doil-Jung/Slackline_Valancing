"""Minimal test: Does launch_passive preserve data.qpos?
Set a LARGE angle (30 deg) and check if it's visible."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mujoco
import mujoco.viewer
import numpy as np
import time

# Simple model: a pole with one hinge joint
xml = """<mujoco model="tilt_test">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" rgba="0.2 0.2 0.3 1"/>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.5"/>
      <geom type="box" size="0.05 0.05 0.5" rgba="0 0.8 0.5 0.9" mass="5"/>
    </body>
  </worldbody>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hinge')
qi = model.jnt_qposadr[jid]

# Set 30 degree tilt
angle_rad = np.radians(30)
data.qpos[qi] = angle_rad
model.qpos0[qi] = angle_rad
mujoco.mj_forward(model, data)
print(f"[BEFORE launch_passive] qpos={np.degrees(data.qpos[qi]):.2f} deg")
print(f"[BEFORE launch_passive] qpos0={np.degrees(model.qpos0[qi]):.2f} deg")

paused = True

def key_cb(keycode):
    global paused
    if keycode == 32:  # Space
        paused = not paused
        print(f"  -> {'PAUSED' if paused else 'RUNNING'}")
    elif keycode == 259:  # Backspace
        data.qpos[qi] = angle_rad
        data.qvel[:] = 0
        data.time = 0
        mujoco.mj_forward(model, data)
        paused = True
        print(f"  -> RESET to {np.degrees(angle_rad):.1f} deg, PAUSED")

viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_cb)

# Check after launch_passive
print(f"[AFTER launch_passive] qpos={np.degrees(data.qpos[qi]):.2f} deg")

# Re-apply with lock
with viewer.lock():
    print(f"[inside lock BEFORE set] qpos={np.degrees(data.qpos[qi]):.2f} deg")
    data.qpos[qi] = angle_rad
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    print(f"[inside lock AFTER set] qpos={np.degrees(data.qpos[qi]):.2f} deg")

viewer.sync()

print(f"[AFTER sync] qpos={np.degrees(data.qpos[qi]):.2f} deg")
print()
print("=== READY ===")
print("You should see a GREEN POLE tilted 30 degrees.")
print("Space = Play/Pause | Backspace = Reset to 30 deg")
print()

frame = 0
while viewer.is_running():
    if not paused:
        with viewer.lock():
            mujoco.mj_step(model, data)
    viewer.sync()
    
    frame += 1
    if frame % 500 == 0:
        print(f"  t={data.time:.2f}s  qpos={np.degrees(data.qpos[qi]):.2f} deg  {'PAUSED' if paused else 'RUNNING'}")
    
    time.sleep(0.002)

viewer.close()
print("Done.")

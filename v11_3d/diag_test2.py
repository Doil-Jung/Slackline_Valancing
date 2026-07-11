"""Diagnostic 2: test launch_passive INSIDE Tkinter callback (like launcher does)"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
import mujoco
import mujoco.viewer
import numpy as np
from config import V11Config
from sim.world import SlacklineWorld

LOG = []
def log(msg):
    print(msg, flush=True)
    LOG.append(msg)

def do_launch():
    log("[Diag2] _launch called from Tkinter after()")
    
    config = V11Config()
    config.robot.theta0 = np.radians(8.6)
    
    world = SlacklineWorld(config)
    model, data = world.initialize()
    
    aq = model.jnt_qposadr[world.ankle_jnt_id]
    hq = model.jnt_qposadr[world.hip_jnt_id]
    
    log(f"[Diag2] Before viewer: hip_qpos={data.qpos[hq]:.4f}")
    
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
        log(f"[Diag2] Viewer opened. is_running={viewer.is_running()}")
    except Exception as e:
        log(f"[Diag2] Viewer FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    log(f"[Diag2] After viewer: hip_qpos={data.qpos[hq]:.4f}")
    
    step_count = 0
    try:
        while viewer.is_running() and step_count < 2000:
            mujoco.mj_step(model, data)
            viewer.sync()
            step_count += 1
            if step_count % 500 == 0:
                log(f"[Diag2] step={step_count}, t={data.time:.3f}s, "
                    f"hip={np.degrees(data.qpos[hq]):+.2f}deg")
            time.sleep(0.001)
    except Exception as e:
        log(f"[Diag2] ERROR in loop: {e}")
        import traceback
        traceback.print_exc()
    
    log(f"[Diag2] Loop ended. steps={step_count}, viewer.is_running()={viewer.is_running()}")
    try:
        viewer.close()
    except:
        pass
    
    # Write results
    log("[Diag2] Done. Exiting Tkinter.")
    root.destroy()

root = tk.Tk()
root.title("Diag2")
root.geometry("300x100")
tk.Label(root, text="Launching viewer in 500ms...").pack()
root.after(500, do_launch)
log("[Diag2] Starting Tkinter mainloop")
root.mainloop()
log("[Diag2] Mainloop ended")

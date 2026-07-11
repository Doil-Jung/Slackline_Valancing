"""Quick test: does the robot fall?"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
from config import V11Config
from sim.world import SlacklineWorld

c = V11Config()
c.robot.theta0 = 0.15
w = SlacklineWorld(c)
m, d = w.initialize()

print("Stepping 2000 times (2 seconds at dt=0.001)...")
for i in range(2000):
    mujoco.mj_step(m, d)
    if (i+1) % 500 == 0:
        s = w.get_state()
        print(f"  t={d.time:.2f}s  "
              f"alpha={np.degrees(s['alpha']):+7.1f}  "
              f"theta={np.degrees(s['theta']):+7.1f}  "
              f"foot_z={s['foot_pos'][2]:.3f}m")

print("Done. If alpha/theta grew large, robot fell successfully.")

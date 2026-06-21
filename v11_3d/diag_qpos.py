"""Diagnostic: Does model.qpos0 survive through the full pipeline?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import V11Config
from sim.world import build_mjcf, SlacklineWorld
import mujoco
import numpy as np

config = V11Config()
config.robot.alpha0 = np.radians(0)
config.robot.theta0 = np.radians(8.6)

print("=" * 60)
print("  Diagnostic: qpos0 / qpos through initialization pipeline")
print("=" * 60)

# Step 1: Build XML and compile
xml = build_mjcf(config)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

aq = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'ankle')]
hq = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'hip')]

print(f"\n--- Step 1: After XML compile + MjData(model) ---")
print(f"  qpos0 full: {[f'{np.degrees(v):.4f}' for v in model.qpos0]}")
print(f"  qpos  full: {[f'{np.degrees(v):.4f}' for v in data.qpos]}")
print(f"  ankle: qpos0={np.degrees(model.qpos0[aq]):.4f} deg")
print(f"  hip:   qpos0={np.degrees(model.qpos0[hq]):.4f} deg")
print(f"  nkey={model.nkey}")
if model.nkey > 0:
    kq = model.key_qpos[:model.nq]
    print(f"  key_qpos: {[round(np.degrees(float(v)), 4) for v in kq]}")

# Step 2: Manual set (as in initialize())
hip_rel = config.robot.theta0 - config.robot.alpha0
model.qpos0[aq] = config.robot.alpha0
model.qpos0[hq] = hip_rel
data.qpos[aq] = config.robot.alpha0
data.qpos[hq] = hip_rel
mujoco.mj_forward(model, data)

print(f"\n--- Step 2: After manual qpos0/qpos set ---")
print(f"  qpos0 full: {[f'{np.degrees(v):.4f}' for v in model.qpos0]}")
print(f"  qpos  full: {[f'{np.degrees(v):.4f}' for v in data.qpos]}")
print(f"  ankle: qpos0={np.degrees(model.qpos0[aq]):.4f}, qpos={np.degrees(data.qpos[aq]):.4f}")
print(f"  hip:   qpos0={np.degrees(model.qpos0[hq]):.4f}, qpos={np.degrees(data.qpos[hq]):.4f}")

# Step 3: mj_resetData (simulating what launch() might do)
mujoco.mj_resetData(model, data)

print(f"\n--- Step 3: After mj_resetData ---")
print(f"  qpos  full: {[f'{np.degrees(v):.4f}' for v in data.qpos]}")
print(f"  ankle: qpos={np.degrees(data.qpos[aq]):.4f} deg (expect 0.0)")
print(f"  hip:   qpos={np.degrees(data.qpos[hq]):.4f} deg (expect 8.6)")

# Step 4: Full pipeline via SlacklineWorld.initialize()
print(f"\n--- Step 4: Full pipeline via SlacklineWorld.initialize() ---")
world = SlacklineWorld(config)
model2, data2 = world.initialize()
aq2 = model2.jnt_qposadr[world.ankle_jnt_id]
hq2 = model2.jnt_qposadr[world.hip_jnt_id]
print(f"  qpos0 full: {[f'{np.degrees(v):.4f}' for v in model2.qpos0]}")
print(f"  qpos  full: {[f'{np.degrees(v):.4f}' for v in data2.qpos]}")
print(f"  ankle: qpos0={np.degrees(model2.qpos0[aq2]):.4f}, qpos={np.degrees(data2.qpos[aq2]):.4f}")
print(f"  hip:   qpos0={np.degrees(model2.qpos0[hq2]):.4f}, qpos={np.degrees(data2.qpos[hq2]):.4f}")

# Step 5: mj_resetData after full pipeline
mujoco.mj_resetData(model2, data2)
print(f"\n--- Step 5: After mj_resetData on full pipeline model ---")
print(f"  qpos  full: {[f'{np.degrees(v):.4f}' for v in data2.qpos]}")
print(f"  hip:   qpos={np.degrees(data2.qpos[hq2]):.4f} deg (expect 8.6)")

# Verdict
hip_after_reset = np.degrees(data2.qpos[hq2])
if abs(hip_after_reset - 8.6) < 0.1:
    print("\n>>> PASS: qpos0 correctly preserves initial angles through reset")
else:
    print(f"\n>>> FAIL: qpos0 lost! hip={hip_after_reset:.4f} instead of 8.6")
    print(">>> This means launch() would reset angles to zero!")

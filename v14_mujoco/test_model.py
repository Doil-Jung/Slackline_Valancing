"""Test V14: Free fall + LQR control, check X-roll stability"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v14_mujoco import generate_mjcf, get_state, get_x_roll, compute_lqr, THETA0, TAU_MAX
import mujoco
import numpy as np

xml = generate_mjcf()
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
K_lqr = compute_lqr()

hip_qadr = model.jnt_qposadr[model.joint('hip').id]

# Test 1: Free fall (no control) - 3 seconds
print("\n=== TEST 1: Free fall (no control) ===")
mujoco.mj_resetData(model, data)
data.qpos[hip_qadr] = THETA0
mujoco.mj_forward(model, data)

for step in range(3000):
    mujoco.mj_step(model, data)
    if step % 500 == 0:
        x = get_state(model, data)
        xr = np.degrees(get_x_roll(model, data))
        err = np.linalg.norm(data.xpos[model.body('rope_b_end').id] -
                             data.xpos[model.body('ankle_b_target').id])
        print(f"  t={data.time:.1f}s phi={np.degrees(x[0]):+6.1f} "
              f"a={np.degrees(x[1]):+6.1f} th={np.degrees(x[2]):+6.1f}  "
              f"Xroll={xr:+6.1f}  err={err:.4f}")

# Test 2: LQR control - 5 seconds
print("\n=== TEST 2: LQR control (5s) ===")
mujoco.mj_resetData(model, data)
data.qpos[hip_qadr] = THETA0
mujoco.mj_forward(model, data)

for step in range(5000):
    x = get_state(model, data)
    tau = 0.0
    if abs(x[1]) < np.radians(60) and abs(x[2]) < np.radians(60):
        tau = -float(K_lqr @ x)
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
    data.ctrl[0] = tau
    mujoco.mj_step(model, data)

    if step % 500 == 0:
        xr = np.degrees(get_x_roll(model, data))
        err = np.linalg.norm(data.xpos[model.body('rope_b_end').id] -
                             data.xpos[model.body('ankle_b_target').id])
        print(f"  t={data.time:.1f}s phi={np.degrees(x[0]):+6.1f} "
              f"a={np.degrees(x[1]):+6.1f} th={np.degrees(x[2]):+6.1f}  "
              f"Xroll={xr:+6.1f}  tau={tau:+7.0f}  err={err:.4f}")

# Test 3: X-axis perturbation (the KEY test!)
print("\n=== TEST 3: X-axis perturbation (5deg) - should self-restore! ===")
mujoco.mj_resetData(model, data)
# Set small X-axis tilt via rope_a_X joint
rope_ax_qadr = model.jnt_qposadr[model.joint('rope_a_X').id]
data.qpos[rope_ax_qadr] = np.radians(5.0)
mujoco.mj_forward(model, data)

for step in range(5000):
    mujoco.mj_step(model, data)
    if step % 500 == 0:
        x = get_state(model, data)
        xr = np.degrees(get_x_roll(model, data))
        rax = np.degrees(data.qpos[rope_ax_qadr])
        err = np.linalg.norm(data.xpos[model.body('rope_b_end').id] -
                             data.xpos[model.body('ankle_b_target').id])
        print(f"  t={data.time:.1f}s  rope_aX={rax:+6.2f}  Xroll={xr:+6.2f}  "
              f"phi={np.degrees(x[0]):+6.2f}  err={err:.4f}")

print("\nDone!")

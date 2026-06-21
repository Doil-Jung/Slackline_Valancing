"""
V14 Headless Runner -- M_HEAD=0 test
Uses the actual v14_mujoco.py code via environment variables
"""
import os, json, math

# preset_robot.json 로드 후 환경변수 설정 (M_HEAD=0으로 오버라이드)
preset_path = os.path.join(os.path.dirname(__file__), 'preset_robot.json')
with open(preset_path) as f:
    preset = json.load(f)

env_map = {
    'm1': 'V14_M1', 'm2': 'V14_M2', 'mh': 'V14_MHEAD',
    'l1': 'V14_L1', 'l2': 'V14_L2', 'bd': 'V14_BD',
    'lpole': 'V14_LPOLE', 'poleh': 'V14_POLEH', 'rtgt': 'V14_RTGT',
    'btht': 'V14_BTHT', 'taumax': 'V14_TAUMAX', 'rcost': 'V14_RCOST',
    'q0': 'V14_Q0', 'q1': 'V14_Q1', 'q2': 'V14_Q2',
    'q3': 'V14_Q3', 'q4': 'V14_Q4', 'q5': 'V14_Q5',
}

for k, env_k in env_map.items():
    if k in preset:
        val = preset[k]
        if k == 'theta0':
            import math
            os.environ['V14_THETA0'] = str(math.radians(val))
        elif k == 'alpha0':
            os.environ['V14_ALPHA0'] = str(math.radians(val))
        else:
            os.environ[env_k] = str(val)

# M_HEAD = 0 (no head test)
os.environ['V14_MHEAD'] = '0.0'
# theta0, alpha0
os.environ['V14_THETA0'] = str(math.radians(preset.get('theta0', 8.6)))
os.environ['V14_ALPHA0'] = str(math.radians(preset.get('alpha0', 0.0)))

# Now import v14 (it reads env vars at module level)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import v14_mujoco as v14

print(f"=== V14 Headless (M_HEAD={v14.M_HEAD}) ===")
print(f"M1={v14.M1}, M2={v14.M2}, L1={v14.L1}, L2={v14.L2}")
print(f"R={v14.R_TARGET}, B_THETA={v14.B_THETA}, TAU_MAX={v14.TAU_MAX}")
print(f"THETA0={np.degrees(v14.THETA0):.1f} deg")
print(f"M2_EFF={v14.M2_EFF:.4f}, LC2_EFF={v14.LC2_EFF:.4f}, I2_EFF={v14.I2_EFF:.6f}")

xml = v14.generate_mjcf()
K = v14.compute_lqr()

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

hip_jid = model.joint('hip').id
ankle_jid = model.joint('ankle').id
data.qpos[model.jnt_qposadr[hip_jid]] = v14.THETA0 - v14.ALPHA0
data.qpos[model.jnt_qposadr[ankle_jid]] = v14.ALPHA0
mujoco.mj_forward(model, data)

T_END = 3.0
N = int(T_END / v14.DT)
log = np.zeros((N, 8))

for i in range(N):
    x = v14.get_state(model, data)
    tau = -float(K @ x)
    tau = np.clip(tau, -v14.TAU_MAX, v14.TAU_MAX)
    data.ctrl[0] = tau
    log[i] = [data.time, *x, tau]
    mujoco.mj_step(model, data)

# Save CSV
np.savetxt('v14_nohead_log.csv', log, delimiter=',',
           header='t,phi,alpha,theta,phi_dot,alpha_dot,theta_dot,tau',
           comments='')
print("Saved: v14_nohead_log.csv")

# Plot
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle(f'V14 MuJoCo (M_HEAD=0, no head)\n'
             f'm1={v14.M1}, m2={v14.M2}, L1={v14.L1}, L2={v14.L2}, '
             f'R={v14.R_TARGET}, theta0={np.degrees(v14.THETA0):.1f}',
             fontsize=13)

for ax, label, idx, yl in zip(axes,
    ['phi', 'alpha', 'theta', 'tau'],
    [1, 2, 3, 7],
    ['deg', 'deg', 'deg', 'N*m']):
    d = log[:, idx]
    if yl == 'deg': d = np.degrees(d)
    ax.plot(log[:, 0], d, 'b-', lw=1.0)
    ax.set_ylabel(f'{label} [{yl}]')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig('v14_nohead_result.png', dpi=150)
print("Saved: v14_nohead_result.png")
plt.close()

max_theta = np.degrees(np.max(np.abs(log[:, 3])))
print(f"Max |theta|: {max_theta:.1f} deg  ->  {'STABLE' if max_theta < 90 else 'DIVERGED'}")

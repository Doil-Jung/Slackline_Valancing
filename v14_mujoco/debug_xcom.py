# -*- coding: utf-8 -*-
"""디버그: MuJoCo에서 순간 접기 후 x_CoM 시계열 확인"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (generate_mjcf, M1, M2_EFF, LC1, LC2_EFF, L1, GRAV)

g = GRAV; Mt = M1 + M2_EFF
p1 = M1*LC1 + M2_EFF*L1; p2 = M2_EFF*LC2_EFF
h = (p1 + p2) / Mt; c_foot = 0.126

R = 1.0; beta0 = np.radians(1); eps = 1.0
omega = np.sqrt(g/(R+h))
Tq = np.pi/(2*omega)

xml = generate_mjcf(R)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
dt = model.opt.timestep

rope_adr = model.jnt_qposadr[model.joint('phi_Y').id]
ankle_adr = model.jnt_qposadr[model.joint('ankle').id]
hip_adr = model.jnt_qposadr[model.joint('hip').id]

# 순간 접기 설정
delta_fold = h*(1+eps)*beta0/c_foot
phi0 = h*(1+eps)*beta0/R
alpha_new = -eps*beta0 - p2*delta_fold/(p1+p2)

print(f"R={R}, beta0={np.degrees(beta0):.1f}°, eps={eps}")
print(f"delta_fold={np.degrees(delta_fold):.1f}°, phi0={np.degrees(phi0):.3f}°")
print(f"alpha_new={np.degrees(alpha_new):.3f}°")

data.qpos[rope_adr] = phi0
data.qpos[ankle_adr] = alpha_new
data.qpos[hip_adr] = delta_fold
data.qvel[:] = 0
mujoco.mj_forward(model, data)

# 초기 x_CoM 확인
phi = data.qpos[rope_adr]; alpha = data.qpos[ankle_adr]; hip = data.qpos[hip_adr]
theta = alpha + hip
beta = (p1*alpha + p2*theta)/(p1+p2)
x_com0 = R*phi + h*beta  # 선형 근사
x_com0_sin = R*np.sin(phi) + h*np.sin(beta)
print(f"\n초기 x_CoM (선형) = {x_com0*1000:.2f}mm")
print(f"초기 x_CoM (sin)  = {x_com0_sin*1000:.2f}mm")
print(f"이론 x_CoM = h*beta0 = {h*beta0*1000:.2f}mm")

# 자유 진화
times = []; xcoms_lin = []; xcoms_sin = []
phis = []; betas = []
for i in range(int(Tq/dt)):
    phi = data.qpos[rope_adr]; alpha = data.qpos[ankle_adr]; hip = data.qpos[hip_adr]
    theta = alpha + hip
    beta = (p1*alpha + p2*theta)/(p1+p2)
    times.append(data.time)
    xcoms_lin.append(R*phi + h*beta)
    xcoms_sin.append(R*np.sin(phi) + h*np.sin(beta))
    phis.append(phi); betas.append(beta)
    data.ctrl[0] = 0
    mujoco.mj_step(model, data)

times = np.array(times); xcoms_lin = np.array(xcoms_lin)
xcoms_sin = np.array(xcoms_sin); phis = np.array(phis); betas = np.array(betas)

# 이론 x_CoM
x_theory = h*beta0*((1+eps)*np.cos(omega*times) - eps*np.cosh(np.sqrt(Mt*g*h/
            (M1*(LC1**2+L1**2/12)+M2_EFF*(L1+LC2_EFF)**2))*times))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
ax.plot(times, xcoms_lin*1000, 'b-', linewidth=2, label='MuJoCo x_CoM (linear)')
ax.plot(times, xcoms_sin*1000, 'g--', linewidth=2, label='MuJoCo x_CoM (sin)')
ax.plot(times, x_theory*1000, 'r:', linewidth=2, label='Theory x_CoM')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_ylabel('x_CoM [mm]', fontsize=12)
ax.set_title(f'x_CoM after instantaneous fold (R={R}, eps={eps}, beta0={np.degrees(beta0):.0f}°)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(times, np.degrees(phis), 'b-', linewidth=2, label='phi (rope)')
ax.plot(times, np.degrees(betas), 'r-', linewidth=2, label='beta (CoM tilt)')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.set_ylabel('Angle [deg]', fontsize=12)
ax.set_title('Joint angles', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(times, R*np.array(phis)*1000, 'b-', linewidth=2, label='R*phi (foot pos)')
ax.plot(times, h*np.array(betas)*1000, 'r-', linewidth=2, label='h*beta (CoM offset)')
ax.plot(times, xcoms_lin*1000, 'k-', linewidth=2, label='x_CoM = sum')
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.set_xlabel('Time [s]', fontsize=12); ax.set_ylabel('Position [mm]', fontsize=12)
ax.set_title('Decomposition: foot + body', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'debug_xcom.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

# -*- coding: utf-8 -*-
"""여러 R에서 이론 vs MuJoCo x_CoM 비교 - H3 성립 확인"""
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
I_eff = M1*(LC1**2 + L1**2/12) + M2_EFF*((L1+LC2_EFF)**2)
lam = np.sqrt(Mt*g*h / I_eff)

def run_test(R, eps, beta0):
    w = np.sqrt(g/(R+h)); Tq = np.pi/(2*w)
    
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    
    rope_adr = model.jnt_qposadr[model.joint('phi_Y').id]
    ankle_adr = model.jnt_qposadr[model.joint('ankle').id]
    hip_adr = model.jnt_qposadr[model.joint('hip').id]
    
    # 순간 접기
    delta_fold = h*(1+eps)*beta0/c_foot
    phi0 = h*(1+eps)*beta0/R
    alpha_new = -eps*beta0 - p2*delta_fold/(p1+p2)
    
    data.qpos[rope_adr] = phi0
    data.qpos[ankle_adr] = alpha_new
    data.qpos[hip_adr] = delta_fold
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    # 자유 진화
    times = []; xcoms_mj = []; xcoms_th = []
    for i in range(int(Tq*0.8/dt)):
        phi = data.qpos[rope_adr]; alpha = data.qpos[ankle_adr]; hip = data.qpos[hip_adr]
        beta = (p1*alpha + p2*(alpha+hip))/(p1+p2)
        t = data.time
        
        times.append(t)
        xcoms_mj.append(R*phi + h*beta)
        xcoms_th.append(h*beta0*((1+eps)*np.cos(w*t) - eps*np.cosh(lam*t)))
        
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)
    
    return np.array(times), np.array(xcoms_mj), np.array(xcoms_th)

beta0 = np.radians(1)
eps = 0.5  # 작은 ε으로 안전하게
R_list = [0.5, 0.7, 1.0, 1.5, 2.0]

fig, axes = plt.subplots(len(R_list), 1, figsize=(14, 4*len(R_list)))

for ax, R in zip(axes, R_list):
    times, mj, th = run_test(R, eps, beta0)
    
    ax.plot(times, th*1000, 'r--', linewidth=2, label='Theory (decoupled)')
    ax.plot(times, mj*1000, 'b-', linewidth=2, label='MuJoCo (coupled)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    # 오차 계산 (초반 Tq/4 동안)
    w = np.sqrt(g/(R+h)); Tq = np.pi/(2*w)
    mask = times < Tq*0.4
    if mask.sum() > 0:
        err = np.max(np.abs(mj[mask] - th[mask])) / (h*beta0) * 100
    else:
        err = float('nan')
    
    ax.set_ylabel('x_CoM [mm]', fontsize=11)
    ax.set_title(f'R = {R}m  (error = {err:.0f}% of h*beta0 in first 0.4*Tq)',
                 fontsize=12, fontweight='bold',
                 color='green' if err < 20 else 'red')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time [s]', fontsize=12)
plt.suptitle(f'Theory vs MuJoCo: x_CoM after instant fold\n'
             f'beta0={np.degrees(beta0):.0f}°, eps={eps}',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'main_eq_R_compare.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

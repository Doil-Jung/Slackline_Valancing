# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
eps* 검증: 안정 모드 정렬 조건에서 x_CoM 진동 균형
R = 0.3~1.0 (0.1 step), eps = 0.5~3.0 (0.5 step) + eps*
MuJoCo 시뮬레이션 2초, ylim ±200mm
"""
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (
    generate_mjcf, get_state,
    M1, M2_EFF, LC1, LC2_EFF, I2_EFF, L1, GRAV, DT
)

# ============================================================
# 파라미터
# ============================================================
g = GRAV; m2 = M2_EFF; Mt = M1 + m2; I1 = M1*L1**2/12
p1 = M1*LC1 + m2*L1; p2 = m2*LC2_EFF; p3 = m2*L1*LC2_EFF
ps = p1 + p2; h = ps / Mt
T_inv = np.array([[1,0,0],[0,Mt*h/ps,-p2/ps],[0,Mt*h/ps,p1/ps]])
J_aa = M1*LC1**2 + I1 + m2*L1**2
J_tt = m2*LC2_EFF**2 + I2_EFF; J_at = p3

beta0 = np.radians(2)  # 2도

# ============================================================
# eps* 계산
# ============================================================
def compute_eps_star(R):
    """최적 eps: 초기조건이 순수 안정 모드에 정렬되는 eps"""
    M = np.array([[Mt*R**2,R*p1,R*p2],[R*p1,J_aa,J_at],[R*p2,J_at,J_tt]])
    G = np.diag([-Mt*g*R, p1*g, p2*g])
    M_pbd = T_inv.T @ M @ T_inv; G_pbd = T_inv.T @ G @ T_inv
    A2 = np.linalg.solve(M_pbd[:2,:2], G_pbd[:2,:2])
    ev, V = np.linalg.eig(A2)
    idx = np.argsort(ev.real); ev = ev[idx].real; V = V[:,idx].real
    v1 = V[:,0]
    r = v1[0] / v1[1]
    eps_star = -h / (r*R + h)
    omega = np.sqrt(abs(ev[0]))
    lam = np.sqrt(abs(ev[1]))
    return eps_star, omega, lam

# ============================================================
# MuJoCo 시뮬레이션
# ============================================================
def run_wait(R, eps, sim_time=2.0, Kp=10000, Kd=200):
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    ra = model.jnt_qposadr[model.joint('phi_Y').id]
    aa = model.jnt_qposadr[model.joint('ankle').id]
    ha = model.jnt_qposadr[model.joint('hip').id]
    try: rb = model.jnt_qposadr[model.joint('rope_b_Y').id]
    except: rb = None
    
    phi0_mj = -h*(1+eps)*beta0/R
    bn = -eps*beta0
    alpha0 = bn  # delta_fold=0이므로
    
    data.qpos[ra] = phi0_mj
    data.qpos[aa] = alpha0 - phi0_mj  # ankle = alpha_abs - phi
    data.qpos[ha] = 0
    if rb is not None: data.qpos[rb] = phi0_mj
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    n = int(sim_time / DT)
    times = []; xcoms = []
    
    for i in range(n):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > np.radians(80):
            break
        phi=x[0]; alpha=x[1]; theta=x[2]
        delta = theta - alpha; delta_d = x[5] - x[4]
        tau = np.clip(Kp*(0-delta) - Kd*delta_d, -50000, 50000)
        data.ctrl[0] = tau
        
        if i % 10 == 0:
            beta_val = (p1*alpha + p2*theta)/(Mt*h)
            xcom = -R*phi + h*beta_val
            times.append(data.time)
            xcoms.append(xcom)
        
        mujoco.mj_step(model, data)
    
    return np.array(times), np.array(xcoms)

# ============================================================
# 메인 플롯
# ============================================================
R_list = np.arange(0.3, 1.05, 0.1)
eps_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

n_R = len(R_list)
n_eps = len(eps_list) + 1  # +1 for eps*

fig, axes = plt.subplots(n_R, n_eps, figsize=(3.2*n_eps, 2.8*n_R), squeeze=False)

print("=" * 70)
print("  eps* 검증: MuJoCo 2초 시뮬레이션")
print("=" * 70)

for i_R, R in enumerate(R_list):
    eps_star, omega, lam = compute_eps_star(R)
    print(f"\n  R={R:.1f}m: eps*={eps_star:.2f}, omega={omega:.2f}, lam={lam:.2f}")
    
    # eps* 시뮬레이션 (첫 번째 열)
    ax0 = axes[i_R][0]
    t_star, x_star = run_wait(R, eps_star, sim_time=2.0)
    ax0.plot(t_star, x_star*1000, color='#E63946', linewidth=2)
    ax0.axhline(y=0, color='k', linewidth=0.3)
    ax0.set_ylim(-200, 200)
    ax0.set_xlim(0, 2.0)
    ax0.set_title(f'eps*={eps_star:.2f}', fontsize=8, fontweight='bold', color='#E63946')
    if i_R == 0: ax0.set_title(f'eps*={eps_star:.2f}\n(optimal)', fontsize=8, 
                                fontweight='bold', color='#E63946')
    ax0.grid(True, alpha=0.2)
    ax0.tick_params(labelsize=6)
    if i_R == n_R-1: ax0.set_xlabel('t [s]', fontsize=7)
    ax0.set_ylabel(f'R={R:.1f}m', fontsize=8, fontweight='bold')
    
    # 배경색으로 eps* 강조
    ax0.set_facecolor('#FFF0F0')
    
    xmax_star = np.max(np.abs(x_star))*1000 if len(x_star) > 0 else 0
    print(f"    eps*={eps_star:.2f}: xCoM_max={xmax_star:.1f}mm, 지속={t_star[-1]:.2f}s")
    
    # 나머지 eps 스윕
    for i_eps, eps in enumerate(eps_list):
        ax = axes[i_R][i_eps + 1]
        t, x = run_wait(R, eps, sim_time=2.0)
        
        # eps*에 가까울수록 진한 색
        dist = abs(eps - eps_star) / eps_star
        if dist < 0.15:
            color = '#E63946'; lw = 2; bg = '#FFF5F5'
        elif dist < 0.3:
            color = '#457B9D'; lw = 1.5; bg = '#F0F8FF'
        else:
            color = '#264653'; lw = 1.2; bg = 'white'
        
        ax.plot(t, x*1000, color=color, linewidth=lw)
        ax.axhline(y=0, color='k', linewidth=0.3)
        ax.set_ylim(-200, 200)
        ax.set_xlim(0, 2.0)
        ax.set_facecolor(bg)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        
        if i_R == 0:
            ax.set_title(f'eps={eps:.1f}', fontsize=8, fontweight='bold')
        else:
            ax.set_title(f'eps={eps:.1f}', fontsize=7)
        if i_R == n_R-1: ax.set_xlabel('t [s]', fontsize=7)
        
        # y축 레이블은 첫 열만
        if i_eps > 0:
            ax.set_yticklabels([])
        
        xmax = np.max(np.abs(x))*1000 if len(x) > 0 else 999
        survived = t[-1] if len(t) > 0 else 0
        status = "OK" if survived >= 1.95 and xmax < 200 else f"max={xmax:.0f}mm"
        print(f"    eps={eps:.1f}: xCoM_max={xmax:.0f}mm, {status}")

# y축 레이블 위치 조정
for i_R in range(n_R):
    axes[i_R][1].set_yticklabels([])

plt.suptitle('FWE Wait Phase: eps* (안정 모드 정렬) vs eps 스윕\n'
             f'MuJoCo 2초, beta0={np.degrees(beta0):.0f} deg, delta=0 PD 유지',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()

save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'eps_star_scan.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  저장: {save_path}")
print("  완료!")

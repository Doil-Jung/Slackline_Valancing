# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""ε*_gen vs R 그래프"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 파라미터 (derive 스크립트와 동일)
from batch_experiment import M1, M2_EFF, LC1, LC2_EFF, I2_EFF, L1, GRAV
g = GRAV; m2 = M2_EFF; Mt = M1 + m2; I1 = M1*L1**2/12
p1 = M1*LC1 + m2*L1; p2 = m2*LC2_EFF; p3 = m2*L1*LC2_EFF
ps = p1 + p2; h = ps / Mt
T_inv = np.array([[1,0,0],[0,Mt*h/ps,-p2/ps],[0,Mt*h/ps,p1/ps]])
J_aa = M1*LC1**2 + I1 + m2*L1**2
J_tt = m2*LC2_EFF**2 + I2_EFF; J_at = p3

def mode_analysis(R):
    M = np.array([[Mt*R**2,R*p1,R*p2],[R*p1,J_aa,J_at],[R*p2,J_at,J_tt]])
    G = np.diag([-Mt*g*R, p1*g, p2*g])
    Mp = T_inv.T @ M @ T_inv; Gp = T_inv.T @ G @ T_inv
    A2 = np.linalg.solve(Mp[:2,:2], Gp[:2,:2])
    ev, V = np.linalg.eig(A2)
    idx = np.argsort(ev.real); ev = ev[idx].real; V = V[:,idx].real
    Vi = np.linalg.inv(V)
    return np.sqrt(abs(ev[0])), np.sqrt(abs(ev[1])), V[:,0], V[:,1], Vi[0], Vi[1]

def eps_star_stat(R):
    _,_,v1,_,_,_ = mode_analysis(R)
    r = v1[0]/v1[1]
    return -h/(r*R+h)

def eps_star_gen(R, phi_i, beta_i, pdot_i, bdot_i):
    _,lam,_,_,_,u2 = mode_analysis(R)
    q_pred = np.array([phi_i + h*beta_i/R + pdot_i/lam, bdot_i/lam])
    e = np.array([h/R, -1.0])
    num = u2 @ q_pred; den = beta_i * (u2 @ e)
    if abs(den) < 1e-15: return np.nan
    return -num / den

# ============================================================
R_arr = np.linspace(0.2, 3.0, 200)

# 초기조건 세트
ic_sets = [
    ("정지 (ε*)",                0, 2.0,  0,  0, '#2d2d2d', 3.0, '-'),
    ("φ=0.5°",                  0.5, 2.0,  0,  0, '#E63946', 2.0, '-'),
    ("φ=1.0°",                  1.0, 2.0,  0,  0, '#E63946', 2.0, '--'),
    ("β̇=2°/s",                  0, 2.0,  0,  2, '#457B9D', 2.0, '-'),
    ("β̇=5°/s",                  0, 2.0,  0,  5, '#457B9D', 2.0, '--'),
    ("φ̇=3°/s",                  0, 2.0,  3,  0, '#2A9D8F', 2.0, '-'),
    ("φ̇=5°/s",                  0, 2.0,  5,  0, '#2A9D8F', 2.0, '--'),
    ("복합 (φ=0.5, φ̇=3, β̇=2)", 0.5, 2.0, 3,  2, '#F4A261', 2.5, '-'),
    ("복합 (φ=1, φ̇=5, β̇=5)",   1.0, 2.0, 5,  5, '#E76F51', 2.5, '-'),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- (1) ε*_gen vs R ---
ax = axes[0, 0]
for label, pd, bd, pdd, bdd, col, lw, ls in ic_sets:
    pi = np.radians(pd); bi = np.radians(bd)
    pdi = np.radians(pdd); bdi = np.radians(bdd)
    eps_arr = [eps_star_gen(R, pi, bi, pdi, bdi) for R in R_arr]
    ax.plot(R_arr, eps_arr, color=col, lw=lw, ls=ls, label=label)

ax.set_xlabel('R [m]', fontsize=12)
ax.set_ylabel('ε*_gen', fontsize=12)
ax.set_title('ε*_gen vs R (다양한 비정지 초기조건)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)
ax.set_ylim(-2, 8)
ax.axhline(0, color='k', lw=0.5)

# --- (2) Δε = ε*_gen - ε*_stat ---
ax = axes[0, 1]
es_stat_arr = [eps_star_stat(R) for R in R_arr]
for label, pd, bd, pdd, bdd, col, lw, ls in ic_sets:
    if '정지' in label: continue
    pi = np.radians(pd); bi = np.radians(bd)
    pdi = np.radians(pdd); bdi = np.radians(bdd)
    eps_arr = [eps_star_gen(R, pi, bi, pdi, bdi) for R in R_arr]
    delta_arr = [eg - es for eg, es in zip(eps_arr, es_stat_arr)]
    ax.plot(R_arr, delta_arr, color=col, lw=lw, ls=ls, label=label)

ax.set_xlabel('R [m]', fontsize=12)
ax.set_ylabel('Δε = ε*_gen − ε*_stat', fontsize=12)
ax.set_title('속도/변위에 의한 ε 보정량', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)
ax.axhline(0, color='k', lw=0.5)

# --- (3) 접기 후 초기 φ (소각근사 유효 범위 체크) ---
ax = axes[1, 0]
for label, pd, bd, pdd, bdd, col, lw, ls in ic_sets:
    pi = np.radians(pd); bi = np.radians(bd)
    pdi = np.radians(pdd); bdi = np.radians(bdd)
    phi_after_arr = []
    for R in R_arr:
        eg = eps_star_gen(R, pi, bi, pdi, bdi)
        phi_after = pi + h*(1+eg)*bi/R
        phi_after_arr.append(np.degrees(phi_after))
    ax.plot(R_arr, phi_after_arr, color=col, lw=lw, ls=ls, label=label)

ax.axhline(10, color='red', lw=1, ls=':', alpha=0.5, label='소각근사 한계 (~10°)')
ax.axhline(-10, color='red', lw=1, ls=':', alpha=0.5)
ax.set_xlabel('R [m]', fontsize=12)
ax.set_ylabel('접기 후 φ₀ [deg]', fontsize=12)
ax.set_title('접기 후 φ₀ 크기 (소각근사 유효 범위)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.2)
ax.set_ylim(-5, 40)

# --- (4) 감쇠 시간 상수 1/λ_eff ---
ax = axes[1, 1]
tau_arr = []
omega_arr = []
for R in R_arr:
    omega, lam, *_ = mode_analysis(R)
    tau_arr.append(1/lam)
    omega_arr.append(2*np.pi/omega)

ax.plot(R_arr, tau_arr, '#E63946', lw=2.5, label='감쇠 시간 1/λ_eff')
ax.plot(R_arr, omega_arr, '#457B9D', lw=2.5, ls='--', label='안정 진동 주기 2π/ω_eff')
ax.set_xlabel('R [m]', fontsize=12)
ax.set_ylabel('시간 [s]', fontsize=12)
ax.set_title('불안정 모드 감쇠 시간 vs 안정 모드 진동 주기', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.2)

fig.suptitle('일반화 ε*_gen 특성 분석 (β_i = 2°)', fontsize=14, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'eps_star_gen_vs_R.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"저장: {save_path}")

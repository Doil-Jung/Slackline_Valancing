# -*- coding: utf-8 -*-
"""
실효 모델: T_peak(δ 최대 시점)을 T로 사용
메인 식 위에 LQR 점이 있는지 직접 확인
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.optimize import brentq
import mujoco
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (generate_mjcf, compute_lqr, get_state,
                               M1, M2_EFF, LC1, LC2_EFF, L1, DT, TAU_MAX, GRAV)

g = GRAV; Mt = M1 + M2_EFF
p1 = M1*LC1 + M2_EFF*L1; p2 = M2_EFF*LC2_EFF
h = (p1 + p2) / Mt
c_foot = 0.126
I_eff = M1*(LC1**2 + L1**2/12) + M2_EFF*((L1+LC2_EFF)**2)
lam = np.sqrt(Mt*g*h / I_eff)

def omega(R): return np.sqrt(g/(R+h))
def T_quarter(R): return np.pi/(2*omega(R))

def solve_T(R, eps):
    w = omega(R); Tq = T_quarter(R)
    def eq(T): return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    try: return brentq(eq, 1e-6, Tq*0.999)
    except: return None

def measure_lqr(R, beta0_deg=3.0):
    """MuJoCo에서 eps, T_peak, T_quarter 비율 측정"""
    xml = generate_mjcf(R); model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); K = compute_lqr(R)
    b0 = np.radians(beta0_deg)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = b0
    data.qpos[model.jnt_qposadr[model.joint('hip').id]] = 0
    mujoco.mj_forward(model, data)
    max_d = 0; t_p = 0
    for i in range(int(5.0/DT)):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > 1.4: return None
        tau = np.clip(-float(K@x), -TAU_MAX, TAU_MAX); data.ctrl[0] = tau
        d = x[2]-x[1]
        if abs(d) > abs(max_d): max_d=d; t_p=data.time
        mujoco.mj_step(model, data)
    eps = c_foot*abs(max_d)/(h*b0) - 1
    Tq = T_quarter(R)
    # 메인 식 잔차
    w = omega(R)
    main_eq = (1+eps)*np.cos(w*t_p) - eps*np.cosh(lam*t_p)
    return {'R':R, 'eps':eps, 'T_peak':t_p, 'Tq':Tq, 'T_ratio':t_p/Tq,
            'main_eq_residual': main_eq}

R_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]
results = []
print("=" * 80)
print(f"{'R':>5} {'eps':>7} {'T_peak':>7} {'Tq':>7} {'T/Tq':>6} {'main_resid':>11}")
print("=" * 80)
for R in R_list:
    r = measure_lqr(R)
    if r:
        results.append(r)
        print(f"{r['R']:5.1f} {r['eps']:7.3f} {r['T_peak']:7.3f} {r['Tq']:7.3f} "
              f"{r['T_ratio']:6.2f} {r['main_eq_residual']:11.4f}")

Rs = [r['R'] for r in results]
epss = [r['eps'] for r in results]
Ts = [r['T_peak'] for r in results]
Tqs = [r['Tq'] for r in results]
ratios = [r['T_ratio'] for r in results]

# ============================================================
#  그래프
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- A: ε-T 평면, 메인 식 곡선 + LQR 점 ---
ax = axes[0][0]
R_show = [0.5, 0.7, 1.0, 1.5, 2.0]
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
for R, col in zip(R_show, colors):
    eps_r = np.linspace(0.01, 5, 200); T_r = []
    for e in eps_r:
        t = solve_T(R, e)
        T_r.append(t if t else np.nan)
    ax.plot(eps_r, T_r, '-', color=col, linewidth=2, label=f'R={R}')
# LQR 점
for r in results:
    if r['R'] in R_show:
        idx = R_show.index(r['R'])
        ax.plot(r['eps'], r['T_peak'], 'o', color=colors[idx], markersize=14,
                markeredgecolor='black', markeredgewidth=2, zorder=5)
ax.set_xlabel('epsilon', fontsize=12); ax.set_ylabel('T [s]', fontsize=12)
ax.set_title('Main Eq. Curves + LQR Points\nDoes LQR sit ON the curve?', fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(-0.2, 4); ax.set_ylim(0, 0.9)

# --- B: ε vs R ---
ax = axes[0][1]
ax.plot(Rs, epss, 'o-', color='#e74c3c', markersize=10, markeredgecolor='black',
        linewidth=2.5, label='LQR (MuJoCo)')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('R [m]', fontsize=12); ax.set_ylabel('epsilon', fontsize=12)
ax.set_title('LQR epsilon vs R', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3); ax.legend(fontsize=10)

# --- C: T_peak/T_quarter vs R ---
ax = axes[1][0]
ax.plot(Rs, ratios, 'o-', color='#3498db', markersize=10, markeredgecolor='black',
        linewidth=2.5, label='T_peak / T_quarter')
ax.axhline(y=0.5, color='gray', linewidth=1, linestyle='--', alpha=0.5)
ax.axhline(y=0.67, color='orange', linewidth=1, linestyle='--', alpha=0.5, label='2/3')
ax.set_xlabel('R [m]', fontsize=12); ax.set_ylabel('T_peak / T_quarter', fontsize=12)
ax.set_title('Fold Time as Fraction of Quarter Period\n"Hidden 2nd Equation"', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3); ax.legend(fontsize=10); ax.set_ylim(0, 1)

# --- D: 메인 식 잔차 ---
ax = axes[1][1]
resids = [r['main_eq_residual'] for r in results]
ax.bar(Rs, resids, width=0.08, color='#2ecc71', edgecolor='black', alpha=0.8)
ax.axhline(y=0, color='red', linewidth=1, linestyle='-')
ax.set_xlabel('R [m]', fontsize=12); ax.set_ylabel('Residual', fontsize=12)
ax.set_title('Main Eq. Residual at LQR Point\n(0 = perfect match)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Two-Equation Analysis: Main Eq + Finite Fold Time',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'two_equation_analysis.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.show()

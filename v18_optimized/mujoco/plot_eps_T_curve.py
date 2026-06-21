# -*- coding: utf-8 -*-
"""
메인 식의 (ε, T) 곡선 + LQR 작동점
각 R에 대해: 메인 식이 허용하는 (ε, T) 곡선과 LQR이 실제로 선택하는 점
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

g = GRAV
Mt = M1 + M2_EFF
p1 = M1*LC1 + M2_EFF*L1; p2 = M2_EFF*LC2_EFF
h = (p1 + p2) / Mt
c_foot = 0.126
I_eff = M1*(LC1**2 + L1**2/12) + M2_EFF*((L1+LC2_EFF)**2)
lam = np.sqrt(Mt*g*h / I_eff)

def omega(R): return np.sqrt(g/(R+h))
def T_quarter(R): return np.pi/(2*omega(R))

def solve_T(R, eps):
    """메인 식에서 eps → T"""
    w = omega(R); Tq = T_quarter(R)
    def eq(T): return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    try: return brentq(eq, 1e-6, Tq*0.999)
    except: return None

# ============================================================
#  LQR 작동점 측정 (MuJoCo)
# ============================================================
def measure_lqr_point(R, beta0_deg=3.0):
    """MuJoCo에서 LQR의 (eps, T) 측정"""
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    K = compute_lqr(R)
    
    b0 = np.radians(beta0_deg)
    hip_qadr = model.jnt_qposadr[model.joint('hip').id]
    ankle_qadr = model.jnt_qposadr[model.joint('ankle').id]
    data.qpos[ankle_qadr] = b0; data.qpos[hip_qadr] = 0
    mujoco.mj_forward(model, data)
    
    max_delta = 0; t_peak = 0
    for i in range(int(5.0/DT)):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > np.radians(80):
            return None, None
        tau = np.clip(-float(K @ x), -TAU_MAX, TAU_MAX)
        data.ctrl[0] = tau
        delta = x[2] - x[1]
        if abs(delta) > abs(max_delta):
            max_delta = delta; t_peak = data.time
        mujoco.mj_step(model, data)
    
    eps_lqr = c_foot*abs(max_delta)/(h*b0) - 1
    return eps_lqr, t_peak

# ============================================================
#  그래프
# ============================================================
R_list = [0.5, 0.7, 1.0, 1.5, 2.0]
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Plot 1: epsilon-T 평면 ---
ax = axes[0]
for R, color in zip(R_list, colors):
    eps_range = np.linspace(0.01, 5.0, 200)
    Ts = []; eps_valid = []
    for eps in eps_range:
        T = solve_T(R, eps)
        if T is not None:
            Ts.append(T); eps_valid.append(eps)
    if eps_valid:
        ax.plot(eps_valid, Ts, '-', color=color, linewidth=2.5, label=f'R={R}m')

# LQR 작동점
print("LQR 작동점 측정 중...")
for R, color in zip(R_list, colors):
    eps_lqr, t_lqr = measure_lqr_point(R, beta0_deg=3.0)
    if eps_lqr is not None:
        print(f"  R={R:.1f}: eps_LQR={eps_lqr:.3f}, T_LQR={t_lqr:.3f}s")
        ax.plot(eps_lqr, t_lqr, 'o', color=color, markersize=12, 
                markeredgecolor='black', markeredgewidth=2, zorder=5)
        ax.annotate(f'LQR', (eps_lqr, t_lqr), 
                   textcoords='offset points', xytext=(8, 5), fontsize=8)

ax.set_xlabel('epsilon', fontsize=13)
ax.set_ylabel('T [s]', fontsize=13)
ax.set_title('Main Equation Curves + LQR Operating Points\n'
             'Lines = (1+eps)cos(wT) = eps*cosh(lT)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 5)
ax.set_ylim(0, 0.8)

# --- Plot 2: epsilon-R 평면 (LQR 궤적) ---
ax = axes[1]

# 메인 식: 고정 T에서 eps vs R
R_fine = np.linspace(0.3, 3.0, 150)
for T_frac, ls, alpha in [(0.5, '-', 0.3), (0.7, '-', 0.3), (0.9, '-', 0.3)]:
    eps_curve = []
    for R in R_fine:
        Tq = T_quarter(R)
        T = T_frac * Tq
        w = omega(R)
        c = np.cos(w*T); C = np.cosh(lam*T)
        if abs(C - c) > 1e-10:
            eps = c / (C - c)
            if eps > -1 and eps < 10:
                eps_curve.append((R, eps))
    if eps_curve:
        Rs, es = zip(*eps_curve)
        ax.plot(Rs, es, ls, color='gray', alpha=alpha, linewidth=1)
        ax.text(Rs[-1]+0.05, es[-1], f'T/Tq={T_frac}', fontsize=7, color='gray')

# LQR 궤적
eps_lqr_list = []; R_lqr_list = []
R_dense = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]
for R in R_dense:
    eps, t = measure_lqr_point(R, beta0_deg=3.0)
    if eps is not None:
        R_lqr_list.append(R)
        eps_lqr_list.append(eps)
        print(f"  R={R:.1f}: eps={eps:.3f}")

ax.plot(R_lqr_list, eps_lqr_list, 'o-', color='#e74c3c', linewidth=2.5,
        markersize=8, markeredgecolor='black', label='LQR operating point')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('R [m]', fontsize=13)
ax.set_ylabel('epsilon (LQR)', fontsize=13)
ax.set_title('LQR Operating Epsilon vs R\n'
             'Gray lines = constant T/Tq from main eq.',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 5)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'eps_T_curve_with_lqr.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.show()

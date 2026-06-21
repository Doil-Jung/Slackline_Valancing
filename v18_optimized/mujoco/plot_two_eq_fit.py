# -*- coding: utf-8 -*-
"""
두 식 모델에서 최적 접힘 가속도 a_eff 찾기
Eq1: (1+ε)cos(ωT) = εcosh(λT)
Eq2: T = √(2h(1+ε)/(c·a_eff))
a_eff를 피팅하여 LQR의 ε vs R을 재현
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.optimize import brentq, minimize_scalar
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

def T_fold(eps, a_eff):
    """접기 시간 모델"""
    delta = h*(1+eps)/c_foot
    return np.sqrt(2*delta/a_eff)

def solve_eps(R, a_eff):
    """두 식 연립 → ε"""
    w = omega(R); Tq = T_quarter(R)
    def f(eps):
        T = T_fold(eps, a_eff)
        if T >= Tq*0.99: return -1e6
        return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    # f(0)의 부호 확인
    f0 = f(0.001)
    if f0 < 0: return None  # T가 너무 커서 해 없음
    # 큰 ε에서 f < 0 되는 지점 찾기
    eps_max = 0.01
    for _ in range(50):
        eps_max *= 1.5
        if f(eps_max) < 0: break
        if eps_max > 100: return None
    try:
        return brentq(f, 0.001, eps_max)
    except:
        return None

# ============================================================
#  LQR 측정
# ============================================================
def measure_lqr(R, beta0_deg=3.0):
    xml = generate_mjcf(R); model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); K = compute_lqr(R)
    b0 = np.radians(beta0_deg)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = b0
    data.qpos[model.jnt_qposadr[model.joint('hip').id]] = 0
    mujoco.mj_forward(model, data)
    max_d = 0
    for i in range(int(5.0/DT)):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > 1.4: return None
        data.ctrl[0] = np.clip(-float(K@x), -TAU_MAX, TAU_MAX)
        d = x[2]-x[1]
        if abs(d) > abs(max_d): max_d=d
        mujoco.mj_step(model, data)
    return c_foot*abs(max_d)/(h*b0) - 1

R_data = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]
print("LQR 측정 중...")
eps_lqr = {}
for R in R_data:
    e = measure_lqr(R)
    if e is not None and e > -1:
        eps_lqr[R] = e
        print(f"  R={R:.1f}: eps={e:.4f}")

# ============================================================
#  최적 a_eff 탐색
# ============================================================
R_fit = sorted(eps_lqr.keys())
eps_fit = [eps_lqr[R] for R in R_fit]

def cost(log_a):
    a = np.exp(log_a)
    err = 0; n = 0
    for R, e_lqr in zip(R_fit, eps_fit):
        e_th = solve_eps(R, a)
        if e_th is not None:
            err += (e_th - e_lqr)**2
            n += 1
    return err/max(n,1) if n > 3 else 1e6

# 넓은 범위 탐색
print("\n최적 a_eff 탐색...")
best_a = 10; best_cost = 1e6
for log_a in np.linspace(np.log(1), np.log(500), 200):
    c = cost(log_a)
    if c < best_cost:
        best_cost = c; best_a = np.exp(log_a)

# 미세 조정
res = minimize_scalar(cost, bounds=(np.log(best_a*0.3), np.log(best_a*3)), method='bounded')
a_opt = np.exp(res.x)
print(f"최적 a_eff = {a_opt:.2f} rad/s²  (MSE = {res.fun:.6f})")

# ============================================================
#  그래프
# ============================================================
R_fine = np.linspace(0.3, 3.0, 150)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Plot 1: ε vs R ---
ax = axes[0]

# 최적 a_eff 곡선
eps_opt = []
for R in R_fine:
    e = solve_eps(R, a_opt)
    eps_opt.append(e if e else np.nan)
ax.plot(R_fine, eps_opt, '-', color='#3498db', linewidth=3,
        label=f'Theory (a_eff={a_opt:.1f} rad/s²)')

# 다른 a_eff 범위
for a, col, ls in [(a_opt*0.5, '#95a5a6', '--'), (a_opt*2, '#95a5a6', ':')]:
    eps_a = []
    for R in R_fine:
        e = solve_eps(R, a)
        eps_a.append(e if e else np.nan)
    ax.plot(R_fine, eps_a, ls, color=col, linewidth=1.5, alpha=0.6,
            label=f'a_eff={a:.1f}')

# LQR 데이터
ax.plot(R_fit, eps_fit, 'o', color='#e74c3c', markersize=12,
        markeredgecolor='black', markeredgewidth=2, label='LQR (MuJoCo)', zorder=5)

ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('R [m]', fontsize=14)
ax.set_ylabel('epsilon', fontsize=14)
ax.set_title('Two-Equation Model: epsilon vs R\n'
             'Eq1: (1+ε)cos(ωT)=εcosh(λT)  |  '
             f'Eq2: T=√(2h(1+ε)/(c·a))',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.3, 2.0)

# --- Plot 2: 잔차 분석 ---
ax = axes[1]
eps_theory_at_data = []
for R in R_fit:
    e = solve_eps(R, a_opt)
    eps_theory_at_data.append(e if e else np.nan)
residuals = [et - el for et, el in zip(eps_theory_at_data, eps_fit)]
ax.bar(R_fit, residuals, width=0.08, color='#2ecc71', edgecolor='black', alpha=0.8)
ax.axhline(y=0, color='red', linewidth=1)
ax.set_xlabel('R [m]', fontsize=14)
ax.set_ylabel('eps_theory - eps_LQR', fontsize=14)
ax.set_title(f'Residuals (Theory - LQR)\n'
             f'a_eff = {a_opt:.1f} rad/s²  |  RMSE = {np.sqrt(best_cost):.4f}',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# 수치 비교
print(f"\n{'R':>5} {'eps_LQR':>9} {'eps_theory':>11} {'diff':>8}")
print("-" * 40)
for R, el, et in zip(R_fit, eps_fit, eps_theory_at_data):
    d = (et-el) if et and el else float('nan')
    print(f"{R:5.1f} {el:9.4f} {et:11.4f} {d:8.4f}")

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'two_eq_fit.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.show()

# -*- coding: utf-8 -*-
"""
유한 접기 시간 모델: 두 번째 조건으로 ε 유일 결정
식 1: (1+ε)cos(ωT) = εcosh(λT)    (메인 식)
식 2: T = √(2·h·(1+ε)/(c·a_eff))  (접기 시간 = 대기 시간)
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
                               M1, M2_EFF, LC1, LC2_EFF, L1, I2_EFF, DT, TAU_MAX, GRAV)

g = GRAV; Mt = M1 + M2_EFF
p1 = M1*LC1 + M2_EFF*L1; p2 = M2_EFF*LC2_EFF
h = (p1 + p2) / Mt
c_foot = 0.126
I_eff = M1*(LC1**2 + L1**2/12) + M2_EFF*((L1+LC2_EFF)**2)
lam = np.sqrt(Mt*g*h / I_eff)
I_hip = I2_EFF + M2_EFF*LC2_EFF**2  # 상체 관성모멘트 (허리 기준)

print(f"h = {h:.3f}m, lambda = {lam:.3f}, c = {c_foot:.3f}")
print(f"I_hip = {I_hip:.3f} kg·m²")

def omega(R): return np.sqrt(g/(R+h))
def T_quarter(R): return np.pi/(2*omega(R))

def solve_T_main(R, eps):
    """메인 식에서 T 구하기"""
    w = omega(R); Tq = T_quarter(R)
    def eq(T): return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    try: return brentq(eq, 1e-6, Tq*0.999)
    except: return None

def T_fold_model(eps, a_eff):
    """접기 시간: 등가속도로 δ_fold만큼 접는 데 걸리는 시간
       δ_fold = h·(1+ε)/c (β₀ 소거됨)
       T = √(2·δ_fold/a_eff) = √(2·h·(1+ε)/(c·a_eff))
    """
    delta_fold = h * (1 + eps) / c_foot
    return np.sqrt(2 * delta_fold / a_eff)

def solve_eps_two_eq(R, a_eff):
    """두 식 연립: 메인 식 + 접기 시간 = T → ε 유일 결정"""
    w = omega(R)
    def residual(eps):
        if eps <= 0: return 1e6
        T = T_fold_model(eps, a_eff)
        Tq = T_quarter(R)
        if T >= Tq: return 1e6
        return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    try:
        return brentq(residual, 0.001, 20.0)
    except:
        return None

# ============================================================
#  LQR 작동점 측정 (MuJoCo)
# ============================================================
def measure_lqr(R, beta0_deg=3.0):
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    K = compute_lqr(R)
    b0 = np.radians(beta0_deg)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = b0
    data.qpos[model.jnt_qposadr[model.joint('hip').id]] = 0
    mujoco.mj_forward(model, data)
    max_d = 0; t_p = 0
    for i in range(int(5.0/DT)):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > 1.4: return None, None, None
        tau = np.clip(-float(K@x), -TAU_MAX, TAU_MAX)
        data.ctrl[0] = tau
        d = x[2]-x[1]
        if abs(d) > abs(max_d): max_d=d; t_p=data.time
        mujoco.mj_step(model, data)
    eps = c_foot*abs(max_d)/(h*b0) - 1
    # K에서 유효 가속도
    K_delta = abs(K[2]-K[1])  # δ에 대한 유효 이득
    a_eff_K = K_delta / I_hip
    return eps, t_p, a_eff_K

# ============================================================
#  계산 및 그래프
# ============================================================
R_list = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5])

print("\n=== LQR 측정 ===")
print(f"{'R':>5} {'eps_LQR':>9} {'T_peak':>8} {'a_eff':>8}")
eps_lqr = []; t_lqr = []; a_effs = []; R_ok = []
for R in R_list:
    e, t, a = measure_lqr(R)
    if e is not None and e > -1:
        print(f"{R:5.1f} {e:9.3f} {t:8.3f} {a:8.1f}")
        eps_lqr.append(e); t_lqr.append(t); a_effs.append(a); R_ok.append(R)
R_ok = np.array(R_ok); eps_lqr = np.array(eps_lqr)
t_lqr = np.array(t_lqr); a_effs = np.array(a_effs)

# a_eff 중앙값으로 이론 곡선 생성
a_median = np.median(a_effs)
print(f"\na_eff 중앙값: {a_median:.1f} rad/s²")

# 이론 예측: 각 R에서 두 식 연립으로 eps 계산
R_fine = np.linspace(0.3, 3.0, 100)

# a_eff 고정 모델
eps_theory_fixed = []
for R in R_fine:
    e = solve_eps_two_eq(R, a_median)
    eps_theory_fixed.append(e if e else np.nan)

# R별 실제 a_eff 사용 모델
eps_theory_actual = []
for R in R_fine:
    K = compute_lqr(R)
    K_delta = abs(K[2]-K[1])
    a = K_delta / I_hip
    e = solve_eps_two_eq(R, a)
    eps_theory_actual.append(e if e else np.nan)

# ============================================================
#  그래프
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Plot 1: ε vs R 비교 ---
ax = axes[0]
ax.plot(R_ok, eps_lqr, 'o', color='#e74c3c', markersize=10, 
        markeredgecolor='black', markeredgewidth=1.5, label='LQR (MuJoCo)', zorder=5)
ax.plot(R_fine, eps_theory_fixed, '--', color='#3498db', linewidth=2.5,
        label=f'Theory (a_eff={a_median:.0f} fixed)')
ax.plot(R_fine, eps_theory_actual, '-', color='#2ecc71', linewidth=2.5,
        label='Theory (a_eff from K)')
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('R [m]', fontsize=13)
ax.set_ylabel('epsilon', fontsize=13)
ax.set_title('Two-Equation Model vs LQR\n'
             'Eq1: Main eq.  Eq2: Fold time = T',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.5, 3)

# --- Plot 2: T vs R 비교 ---
ax = axes[1]
ax.plot(R_ok, t_lqr, 'o', color='#e74c3c', markersize=10,
        markeredgecolor='black', markeredgewidth=1.5, label='LQR T_peak', zorder=5)
# 이론 T
T_theory = []
for R in R_fine:
    K = compute_lqr(R)
    a = abs(K[2]-K[1])/I_hip
    e = solve_eps_two_eq(R, a)
    if e:
        T_theory.append(T_fold_model(e, a))
    else:
        T_theory.append(np.nan)
ax.plot(R_fine, T_theory, '-', color='#2ecc71', linewidth=2.5, label='Theory T_fold')
Tq_line = [T_quarter(R) for R in R_fine]
ax.plot(R_fine, Tq_line, ':', color='gray', linewidth=1.5, label='T_quarter')
ax.set_xlabel('R [m]', fontsize=13)
ax.set_ylabel('T [s]', fontsize=13)
ax.set_title('Fold Time vs R', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'two_equation_model.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.show()

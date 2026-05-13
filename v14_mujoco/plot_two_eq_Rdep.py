# -*- coding: utf-8 -*-
"""R-의존 접힘 가속도 모델 피팅 (LQR 캐싱 후 순수 수학만)"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.optimize import brentq, minimize_scalar
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (generate_mjcf, compute_lqr, get_state,
                               M1, M2_EFF, LC1, LC2_EFF, L1, DT, TAU_MAX, GRAV)

g = GRAV; Mt = M1 + M2_EFF
p1 = M1*LC1 + M2_EFF*L1; p2 = M2_EFF*LC2_EFF
h = (p1 + p2) / Mt; c_foot = 0.126
I_eff = M1*(LC1**2 + L1**2/12) + M2_EFF*((L1+LC2_EFF)**2)
lam = np.sqrt(Mt*g*h / I_eff)

def omega(R): return np.sqrt(g/(R+h))
def T_quarter(R): return np.pi/(2*omega(R))

def solve_eps(R, a_eff):
    w = omega(R); Tq = T_quarter(R)
    def f(eps):
        T = np.sqrt(2*h*(1+eps)/(c_foot*a_eff))
        if T >= Tq*0.99: return -1e6
        return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    f0 = f(0.001)
    if f0 < 0: return None
    eps_max = 0.01
    for _ in range(60):
        eps_max *= 1.5
        if f(eps_max) < 0: break
        if eps_max > 200: return None
    try: return brentq(f, 0.001, eps_max)
    except: return None

# ============================================================
#  LQR 한 번만 측정 (캐싱)
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
        if abs(x[2]-x[1]) > abs(max_d): max_d = x[2]-x[1]
        mujoco.mj_step(model, data)
    return c_foot*abs(max_d)/(h*b0) - 1

R_data = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]
print("LQR 측정 (1회)...")
R_fit = []; eps_fit = []
for R in R_data:
    e = measure_lqr(R)
    if e is not None and e > -1:
        R_fit.append(R); eps_fit.append(e)
        print(f"  R={R:.1f}: eps={e:.4f}")
R_fit = np.array(R_fit); eps_fit = np.array(eps_fit)

# ============================================================
#  피팅 (MuJoCo 없이 순수 수학만)
# ============================================================
models = {
    'a₀·ω²':     lambda R, a0: a0 * g/(R+h),
    'a₀·ω':      lambda R, a0: a0 * np.sqrt(g/(R+h)),
    'a₀/R':      lambda R, a0: a0 / R,
    'a₀/(R+h)²': lambda R, a0: a0 / (R+h)**2,
    'a₀·ω³':     lambda R, a0: a0 * (g/(R+h))**1.5,
    'a₀·ω⁴':     lambda R, a0: a0 * (g/(R+h))**2,
}

print("\n피팅 중...")
best_all = None
for name, func in models.items():
    def cost(log_a0, _func=func):
        a0 = np.exp(log_a0); err = 0; n = 0
        for R, e_lqr in zip(R_fit, eps_fit):
            e_th = solve_eps(R, _func(R, a0))
            if e_th is not None: err += (e_th - e_lqr)**2; n += 1
        return err/max(n,1) if n >= 5 else 1e6
    
    best_c = 1e6; best_a0 = 1
    for log_a in np.linspace(np.log(0.01), np.log(50000), 500):
        c = cost(log_a)
        if c < best_c: best_c = c; best_a0 = np.exp(log_a)
    
    rmse = np.sqrt(best_c)
    print(f"  {name:>15}: a₀={best_a0:10.2f}, RMSE={rmse:.4f}")
    
    if best_all is None or rmse < best_all['rmse']:
        best_all = {'name': name, 'a0': best_a0, 'rmse': rmse, 'func': func}

print(f"\n★ 최적: {best_all['name']}, a₀={best_all['a0']:.2f}, RMSE={best_all['rmse']:.4f}")

# ============================================================
#  그래프
# ============================================================
R_fine = np.linspace(0.3, 3.0, 150)
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
# 상위 모델들
sorted_m = sorted(models.items(), key=lambda x: x[0])
colors_m = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6', '#1abc9c', '#34495e']
for i, (name, func) in enumerate(sorted_m):
    def _cost(log_a0, _f=func):
        a0 = np.exp(log_a0); err = 0; n = 0
        for R, e in zip(R_fit, eps_fit):
            e_th = solve_eps(R, _f(R, a0))
            if e_th is not None: err += (e_th - e)**2; n += 1
        return err/max(n,1) if n>=5 else 1e6
    best_c = 1e6; best_a0 = 1
    for la in np.linspace(np.log(0.01), np.log(50000), 200):
        c = _cost(la)
        if c < best_c: best_c = c; best_a0 = np.exp(la)
    eps_th = []
    for R in R_fine:
        e = solve_eps(R, func(R, best_a0))
        eps_th.append(e if e else np.nan)
    lw = 3 if name == best_all['name'] else 1.5
    al = 1.0 if name == best_all['name'] else 0.5
    ax.plot(R_fine, eps_th, '-', color=colors_m[i], linewidth=lw, alpha=al,
            label=f'{name} ({np.sqrt(best_c):.3f})')

ax.plot(R_fit, eps_fit, 'o', color='#e74c3c', markersize=12,
        markeredgecolor='black', markeredgewidth=2, label='LQR', zorder=5)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel('R [m]', fontsize=14); ax.set_ylabel('epsilon', fontsize=14)
ax.set_title('R-dependent a_eff Models vs LQR', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3); ax.set_ylim(-0.3, 2.5)

ax = axes[1]
bf = best_all['func']; ba = best_all['a0']
eps_best = [solve_eps(R, bf(R, ba)) or np.nan for R in R_fit]
resid = [et-el for et, el in zip(eps_best, eps_fit)]
ax.bar(R_fit, resid, width=0.08, color='#3498db', edgecolor='black', alpha=0.8)
ax.axhline(y=0, color='red', linewidth=1)
ax.set_xlabel('R [m]', fontsize=14); ax.set_ylabel('Theory - LQR', fontsize=14)
ax.set_title(f'Best: {best_all["name"]}, a₀={ba:.1f}\nRMSE={best_all["rmse"]:.4f}',
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'two_eq_Rdep_fit.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

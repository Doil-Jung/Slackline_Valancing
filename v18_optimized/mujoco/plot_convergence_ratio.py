# -*- coding: utf-8 -*-
"""
수렴비 r 계산: R × ε 조합별 + 관절한계 최적 ε
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

g = 9.81
M1 = 30.0; L1 = 0.85; LC1 = L1/2
M2 = 40.0; L2 = 0.80; M_HEAD = 5.0; HEAD_R = L2/4
m2 = M2 + M_HEAD
LC2 = (M2*L2/2 + M_HEAD*(L2+HEAD_R)) / m2
Mt = M1 + m2
h = (M1*LC1 + m2*(L1+LC2)) / Mt
p1 = M1*LC1 + m2*L1; p2 = m2*LC2
c_foot = 0.126
I_foot = M1*(LC1**2 + L1**2/12) + m2*((L1+LC2)**2)
lam = np.sqrt(Mt*g*h / I_foot)

print(f"h = {h:.3f} m")
print(f"lambda = {lam:.3f} rad/s")
print(f"c_foot = {c_foot:.3f} m/rad")
print()

def omega(R):
    return np.sqrt(g/(R+h))

def T_quarter(R):
    return np.pi / (2*omega(R))

def solve_T_for_eps(R, eps):
    """메인 식에서 eps가 주어지면 T를 구함."""
    w = omega(R)
    Tq = T_quarter(R)
    def eq(T):
        return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    # T in (0, Tq) 범위에서 탐색
    try:
        T = brentq(eq, 1e-6, Tq*0.999)
        return T
    except:
        return None

def conv_ratio(T, R):
    """수렴비 r = C(1-c)/(C-c)"""
    w = omega(R)
    C = np.cosh(lam*T)
    c = np.cos(w*T)
    if abs(C-c) < 1e-12:
        return 1.0
    return C*(1-c)/(C-c)

def cycles_to_1pct(r):
    """1%까지 수렴하는 데 필요한 사이클 수"""
    if r <= 0 or r >= 1:
        return float('inf')
    return -4.605 / np.log(r)

# ============================================================
#  테이블 1: R × beta0 → 최적 ε(관절한계) → r → n
# ============================================================
print("=" * 85)
print(f"{'R':>5} {'beta0':>6} {'eps_max':>8} {'T [s]':>7} {'Tq [s]':>7} {'T/Tq':>6} {'r':>6} {'n(1%)':>6}")
print("=" * 85)

R_list = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
beta_list = [3, 5, 10]
delta_max = np.pi/2  # 90도

for R in R_list:
    w = omega(R)
    Tq = T_quarter(R)
    for b0_deg in beta_list:
        b0 = np.radians(b0_deg)
        eps_max = c_foot*delta_max/(h*b0) - 1
        if eps_max <= 0:
            print(f"{R:5.1f} {b0_deg:6d} {'불가':>8}")
            continue
        T = solve_T_for_eps(R, eps_max)
        if T is None:
            print(f"{R:5.1f} {b0_deg:6d} {eps_max:8.2f} {'해없음':>7}")
            continue
        r = conv_ratio(T, R)
        n = cycles_to_1pct(r)
        print(f"{R:5.1f} {b0_deg:6d} {eps_max:8.2f} {T:7.3f} {Tq:7.3f} {T/Tq:6.2f} {r:6.3f} {n:6.1f}")
    print()

# ============================================================
#  테이블 2: 고정 ε에서 R별 수렴비
# ============================================================
print("\n" + "=" * 70)
print("고정 epsilon에서 R별 수렴비")
print("=" * 70)
for eps in [0.2, 0.5, 1.0, 2.0]:
    print(f"\n--- eps = {eps:.1f} ---")
    print(f"{'R':>5} {'T [s]':>7} {'Tq [s]':>7} {'T/Tq':>6} {'r':>6} {'n(1%)':>6}")
    for R in R_list:
        Tq = T_quarter(R)
        T = solve_T_for_eps(R, eps)
        if T:
            r = conv_ratio(T, R)
            n = cycles_to_1pct(r)
            print(f"{R:5.1f} {T:7.3f} {Tq:7.3f} {T/Tq:6.2f} {r:6.3f} {n:6.1f}")
        else:
            print(f"{R:5.1f} {'해없음':>7}")

# ============================================================
#  그래프: r vs R (여러 epsilon)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# --- Plot 1: r vs R (고정 epsilon) ---
ax = axes[0]
R_fine = np.linspace(0.2, 3.0, 100)
for eps, color, ls in [(0.1, '#e74c3c', '-'), (0.3, '#e67e22', '-'),
                        (0.5, '#2ecc71', '-'), (1.0, '#3498db', '-'),
                        (2.0, '#9b59b6', '-')]:
    rs = []
    for R in R_fine:
        T = solve_T_for_eps(R, eps)
        if T:
            rs.append(conv_ratio(T, R))
        else:
            rs.append(np.nan)
    ax.plot(R_fine, rs, color=color, linewidth=2, label=f'eps={eps}')

ax.set_xlabel('R [m]', fontsize=12)
ax.set_ylabel('r (convergence ratio)', fontsize=12)
ax.set_title('r vs R (fixed eps)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
ax.axhline(y=1, color='black', linewidth=0.5, linestyle='--')

# --- Plot 2: r vs R (관절한계 최적 eps, beta0별) ---
ax = axes[1]
for b0_deg, color in [(3, '#e74c3c'), (5, '#2ecc71'), (10, '#3498db')]:
    b0 = np.radians(b0_deg)
    eps_max = c_foot*delta_max/(h*b0) - 1
    rs = []
    for R in R_fine:
        if eps_max <= 0:
            rs.append(np.nan); continue
        T = solve_T_for_eps(R, eps_max)
        if T:
            rs.append(conv_ratio(T, R))
        else:
            rs.append(np.nan)
    ax.plot(R_fine, rs, color=color, linewidth=2.5, label=f'b0={b0_deg} deg (eps={eps_max:.2f})')

ax.set_xlabel('R [m]', fontsize=12)
ax.set_ylabel('r (convergence ratio)', fontsize=12)
ax.set_title('r vs R (joint-limited eps)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
ax.axhline(y=1, color='black', linewidth=0.5, linestyle='--')

# --- Plot 3: Cycles to 1% vs R ---
ax = axes[2]
for b0_deg, color in [(3, '#e74c3c'), (5, '#2ecc71'), (10, '#3498db')]:
    b0 = np.radians(b0_deg)
    eps_max = c_foot*delta_max/(h*b0) - 1
    ns = []
    for R in R_fine:
        if eps_max <= 0:
            ns.append(np.nan); continue
        T = solve_T_for_eps(R, eps_max)
        if T:
            r = conv_ratio(T, R)
            ns.append(cycles_to_1pct(r))
        else:
            ns.append(np.nan)
    ax.plot(R_fine, ns, color=color, linewidth=2.5, label=f'b0={b0_deg} deg')

ax.set_xlabel('R [m]', fontsize=12)
ax.set_ylabel('Cycles to 1%', fontsize=12)
ax.set_title('Convergence speed\n(fewer = better)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 50)

plt.suptitle('Convergence Ratio Analysis', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()

save_path = r"c:\Users\user\OneDrive - 충북과학고등학교\원드라이브 동기화 폴더\코딩 작업 폴더\slackline_valancing\docs\convergence_ratio.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.show()

# -*- coding: utf-8 -*-
"""
Graph 2 수정: 메인 식의 영역 + 관절 한계 + LQR 실측
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

g = 9.81
M1 = 30.0; L1 = 0.85; LC1 = L1/2
M2 = 40.0; L2 = 0.80; LC2_raw = L2/2
M_HEAD = 5.0; HEAD_R = L2/4
m2 = M2 + M_HEAD
LC2 = (M2*LC2_raw + M_HEAD*(L2 + HEAD_R)) / m2
Mt = M1 + m2
h = (M1*LC1 + m2*(L1+LC2)) / Mt
p1 = M1*LC1 + m2*L1; p2 = m2*LC2
c_foot = 0.126
I_foot = M1*(LC1**2 + L1**2/12) + m2*((L1+LC2)**2)
lam = np.sqrt(Mt*g*h / I_foot)

def compute_system(R):
    M_mat = np.array([
        [Mt*R**2, R*p1, R*p2],
        [R*p1, M1*LC1**2+M1*L1**2/12+m2*L1**2, m2*L1*LC2],
        [R*p2, m2*L1*LC2, m2*LC2**2+m2*LC2**2]
    ])
    G_mat = np.diag([-Mt*g*R, p1*g, p2*g])
    E = np.array([0,-1,1])
    Minv = np.linalg.inv(M_mat)
    A = np.zeros((6,6)); A[0:3,3:6] = np.eye(3); A[3:6,0:3] = Minv@G_mat
    B = np.zeros((6,1)); B[3:6,0] = Minv@E
    return A, B

Q = np.diag([100,100,100,1,1,1]); R_cost = 0.01

# === 계산 ===
R_range = np.linspace(0.15, 3.0, 50)

# 1. LQR의 실제 epsilon
eps_lqr = []
for R in R_range:
    A, B = compute_system(R)
    P = solve_continuous_are(A, B, Q, R_cost)
    K = (1/R_cost)*B.T@P; A_cl = A-B@K
    x0 = np.array([0,0.1,0.1,0,0,0])
    sol = solve_ivp(lambda t,x: A_cl@x, [0,5], x0, max_step=0.002, dense_output=True)
    t_ev = np.linspace(0,3,3000); x_ev = sol.sol(t_ev)
    dp = np.max(np.abs(x_ev[2]-x_ev[1]))
    eps_lqr.append(c_foot*dp/(h*0.1) - 1)

# 2. 메인 식의 유효 영역: 각 R에서 가능한 epsilon 범위
eps_min_curve = []  # 최소 ε (T → T_quarter)
eps_max_theory = [] # 메인 식에서 합리적 최대 ε

for R in R_range:
    omega = np.sqrt(g/(R+h))
    T_q = np.pi/(2*omega)
    # T를 스캔하여 ε 범위
    eps_list = []
    for T_frac in np.linspace(0.3, 0.999, 200):
        T = T_q * T_frac
        denom = np.cosh(lam*T) - np.cos(omega*T)
        if denom > 0:
            e = np.cos(omega*T) / denom
            if e > 0: eps_list.append(e)
    eps_min_curve.append(min(eps_list) if eps_list else 0)
    eps_max_theory.append(max(eps_list) if eps_list else 0)

# 3. 관절 한계에 의한 ε_max (각 β₀에서)
delta_max = np.pi/2  # 90도

fig, ax = plt.subplots(figsize=(10, 7))

# 메인 식 유효 영역 (ε > 0이면 제어 가능)
ax.fill_between(R_range, 0, eps_max_theory, alpha=0.15, color='#3498db',
                label='Main Eq: valid region')

# 관절 한계 (β₀에 따라)
for beta0_deg, ls, alpha in [(3, '--', 0.8), (5, '-', 1.0), (10, '-.', 0.7), (15, ':', 0.5)]:
    beta0 = np.radians(beta0_deg)
    eps_joint = c_foot*delta_max/(h*beta0) - 1
    if eps_joint > 0:
        ax.axhline(y=eps_joint, color='#e74c3c', linestyle=ls, alpha=alpha, linewidth=1.5,
                   label=f'Joint limit (beta0={beta0_deg} deg): eps<{eps_joint:.2f}')

# LQR 실측
ax.plot(R_range, eps_lqr, 'o-', color='#2c3e50', markersize=3, linewidth=2.5,
        label='LQR (Q/R=10000)', zorder=5)

# 주석
ax.set_xlabel('R [m]', fontsize=14)
ax.set_ylabel('epsilon', fontsize=14)
ax.set_title('epsilon vs R: Main Eq Region + Joint Limit + LQR', fontsize=15, fontweight='bold')
ax.set_ylim(-0.1, 3.0)
ax.set_xlim(0.1, 3.0)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# 영역 설명
ax.annotate('joint limit impossible\n(can\'t overshoot enough)',
            xy=(2.5, 0.3), fontsize=9, color='#e74c3c', style='italic',
            ha='center')
ax.annotate('LQR chooses\nspecific epsilon\nbased on Q/R cost',
            xy=(1.0, eps_lqr[np.argmin(np.abs(R_range-1.0))]+0.15),
            fontsize=9, color='#2c3e50', style='italic', ha='center')

plt.tight_layout()
save_path = r"c:\Users\user\OneDrive - 충북과학고등학교\원드라이브 동기화 폴더\코딩 작업 폴더\slackline_valancing\docs\epsilon_region_graph.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.show()

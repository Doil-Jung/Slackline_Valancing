# -*- coding: utf-8 -*-
"""
두 그래프:
  Graph 1: δ_peak vs β₀ (비례 제어 확인) — 여러 R에서
  Graph 2: ε_LQR vs R (R→ε 결정 구조 확인)
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

def compute_system(M1, m2_eff, L1, LC1, LC2, R):
    Mt = M1 + m2_eff
    p1 = M1*LC1 + m2_eff*L1
    p2 = m2_eff*LC2
    p3 = m2_eff*L1*LC2
    I_alpha = M1*LC1**2 + M1*L1**2/12 + m2_eff*L1**2
    I_theta = m2_eff*LC2**2 + m2_eff*LC2**2
    M_mat = np.array([
        [Mt*R**2, R*p1, R*p2],
        [R*p1, I_alpha, p3],
        [R*p2, p3, I_theta]
    ])
    G_mat = np.diag([-Mt*g*R, p1*g, p2*g])
    E = np.array([0, -1, 1])
    Minv = np.linalg.inv(M_mat)
    A = np.zeros((6,6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = Minv @ G_mat
    B = np.zeros((6,1))
    B[3:6, 0] = Minv @ E
    h_CoM = (p1 + p2) / Mt
    return A, B, Mt, h_CoM, p1, p2

# 파라미터
M1 = 30.0; L1 = 0.85; LC1 = L1/2
M2 = 40.0; L2 = 0.80; LC2_raw = L2/2
M_HEAD = 5.0; HEAD_R = L2/4
m2 = M2 + M_HEAD
LC2 = (M2*LC2_raw + M_HEAD*(L2 + HEAD_R)) / m2
c_foot = 0.126  # m/rad

Q = np.diag([100, 100, 100, 1, 1, 1])
R_cost = 0.01

def simulate_lqr(R_rope, beta0):
    A, B, Mt, h_CoM, p1, p2 = compute_system(M1, m2, L1, LC1, LC2, R_rope)
    P = solve_continuous_are(A, B, Q, R_cost)
    K = (1/R_cost) * B.T @ P
    A_cl = A - B @ K
    x0 = np.array([0, beta0, beta0, 0, 0, 0])
    sol = solve_ivp(lambda t,x: A_cl@x, [0,5], x0, max_step=0.002, dense_output=True)
    t_ev = np.linspace(0, 3, 3000)
    x_ev = sol.sol(t_ev)
    delta_t = x_ev[2] - x_ev[1]
    idx_peak = np.argmax(np.abs(delta_t))
    return abs(delta_t[idx_peak]), t_ev[idx_peak], h_CoM

# ===== 그래프 생성 =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Graph 1: δ_peak vs β₀ (여러 R에서) ---
R_list = [0.3, 0.5, 0.7, 1.0, 1.5]
colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']
beta_range = np.linspace(0.01, 0.20, 15)  # 0.6° ~ 11.5°

for R_rope, color in zip(R_list, colors):
    delta_peaks = []
    for b0 in beta_range:
        dp, _, _ = simulate_lqr(R_rope, b0)
        delta_peaks.append(dp)
    delta_peaks = np.array(delta_peaks)
    
    # 선형 피팅
    slope = np.polyfit(beta_range, delta_peaks, 1)[0]
    
    ax1.plot(np.degrees(beta_range), np.degrees(delta_peaks), 'o-', 
             color=color, markersize=4, linewidth=2,
             label=f'R={R_rope}m (기울기={slope:.1f})')

ax1.set_xlabel('초기 기울기 β₀ [도]', fontsize=13)
ax1.set_ylabel('최대 접힘 각도 δ_peak [도]', fontsize=13)
ax1.set_title('Graph 1: δ_peak vs β₀ — 비례 관계 확인', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.text(0.95, 0.05, '직선 = 비례 제어\n(β₀에 비례하여 접음)', 
         transform=ax1.transAxes, ha='right', va='bottom',
         fontsize=10, style='italic', color='gray',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Graph 2: ε_LQR vs R ---
R_range = np.linspace(0.15, 3.0, 30)
epsilon_list = []
T_peak_list = []
T_quarter_list = []

for R_rope in R_range:
    dp, t_pk, h_CoM = simulate_lqr(R_rope, 0.1)
    eps = c_foot * dp / (h_CoM * 0.1) - 1
    epsilon_list.append(eps)
    T_peak_list.append(t_pk)
    T_quarter_list.append(np.pi / (2 * np.sqrt(g/(R_rope + h_CoM))))

epsilon_list = np.array(epsilon_list)
T_peak_list = np.array(T_peak_list)
T_quarter_list = np.array(T_quarter_list)

# 이론적 ε (메인 식에서)
lambda_inv = np.sqrt(M1+m2) * np.sqrt(g * h_CoM / 89.05)  # 대략적
eps_theory = []
for R_rope in R_range:
    omega = np.sqrt(g / (R_rope + h_CoM))
    T_q = np.pi / (2 * omega)
    best_eps = None
    for T_frac in np.linspace(0.5, 0.99, 200):
        T_try = T_q * T_frac
        denom = np.cosh(lambda_inv * T_try) - np.cos(omega * T_try)
        if denom > 0:
            e = np.cos(omega * T_try) / denom
            if 0.001 < e < 5.0:
                if best_eps is None or e < best_eps:
                    best_eps = e
    eps_theory.append(best_eps if best_eps else np.nan)

ax2.plot(R_range, epsilon_list, 'o-', color='#3498db', markersize=4, linewidth=2.5,
         label='ε (LQR 시뮬레이션)')
ax2.plot(R_range, eps_theory, '--', color='#e74c3c', linewidth=2,
         label='ε (이론: 메인 식, 최소 ε)')
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')

# R 참고선
for R_ref, label in [(0.7, 'cond 최소'), (1.0, 'R=h')]:
    ax2.axvline(x=R_ref, color='gray', linewidth=1, linestyle=':', alpha=0.5)
    ax2.text(R_ref+0.03, max(epsilon_list)*0.9, label, fontsize=8, color='gray')

ax2.set_xlabel('줄 곡률 반경 R [m]', fontsize=13)
ax2.set_ylabel('오버슈트 비율 ε', fontsize=13)
ax2.set_title('Graph 2: ε vs R — R이 ε를 결정', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.text(0.95, 0.95, 'R 작음 → ε 큼 (급히 많이 접음)\nR 큼 → ε 작음 (천천히 조금)', 
         transform=ax2.transAxes, ha='right', va='top',
         fontsize=10, style='italic', color='gray',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
save_path = r"c:\Users\user\OneDrive - 충북과학고등학교\원드라이브 동기화 폴더\코딩 작업 폴더\slackline_valancing\docs\epsilon_analysis_graphs.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"그래프 저장: {save_path}")
plt.show()

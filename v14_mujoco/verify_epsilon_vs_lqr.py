# -*- coding: utf-8 -*-
"""
LQR 이득과 이산 모델의 ε 비교

방법:
1. LQR K 행렬로 폐루프 시스템 시뮬레이션
2. 초기 기울기 β₀에서 최대 접힘각 δ_peak 측정
3. ε_LQR = (c·δ_peak)/(h·β₀) - 1 역산
4. 이산 모델의 예측과 비교
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp

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
    return A, B, {'Mt':Mt, 'p1':p1, 'p2':p2, 'h_CoM':h_CoM, 'c_foot': (p1+p2)/Mt * 0.131}

# 파라미터
M1 = 30.0; L1 = 0.85; LC1 = L1/2
M2 = 40.0; L2 = 0.80; LC2_raw = L2/2
M_HEAD = 5.0; HEAD_R = L2/4
m2 = M2 + M_HEAD
LC2 = (M2*LC2_raw + M_HEAD*(L2 + HEAD_R)) / m2
Mt = M1 + m2
h_CoM = (M1*LC1 + m2*(L1+LC2)) / Mt  # 수정: 정확한 h_CoM
p1 = M1*LC1 + m2*L1
p2 = m2*LC2
c_foot = 0.126  # Δx_foot/δ (수치 검증 결과)

print("=" * 70)
print("  LQR 이득 ↔ 이산 모델의 ε 비교")
print("=" * 70)
print(f"  h_CoM = {h_CoM:.4f} m,  c_foot = {c_foot} m/rad")
print(f"  이산 모델: δ = h·(1+ε)/c · β₀ = {h_CoM/c_foot:.2f}·(1+ε)·β₀")

Q = np.diag([100, 100, 100, 1, 1, 1])
R_cost = 0.01

print(f"\n  {'R':>6} {'β₀':>8} {'δ_peak':>10} {'δ/β₀':>8} {'ε_LQR':>8} {'T_peak':>8} {'T_q/4':>8} {'T_peak/T_q':>10}")
print(f"  {'-'*75}")

for R_rope in [0.3, 0.5, 0.7, 1.0, 1.5]:
    A, B, params = compute_system(M1, m2, L1, LC1, LC2, R_rope)
    
    # LQR 계산
    P = solve_continuous_are(A, B, Q, R_cost)
    K = (1/R_cost) * B.T @ P
    K = K[0]
    
    # 폐루프: ẋ = (A - B·K)·x
    A_cl = A - B @ K.reshape(1,-1)
    
    # 시뮬레이션: 초기 기울기 β₀
    for beta0 in [0.05, 0.1]:
        x0 = np.array([0, beta0, beta0, 0, 0, 0])  # φ=0, α=θ=β₀
        
        def dynamics(t, x):
            return A_cl @ x
        
        sol = solve_ivp(dynamics, [0, 5], x0, max_step=0.001, dense_output=True)
        
        t_eval = np.linspace(0, 3, 3000)
        x_eval = sol.sol(t_eval)
        
        # δ(t) = θ(t) - α(t)
        delta_t = x_eval[2] - x_eval[1]
        
        # δ의 최대값 (절대값)
        idx_peak = np.argmax(np.abs(delta_t))
        delta_peak = delta_t[idx_peak]
        t_peak = t_eval[idx_peak]
        
        # ε 역산: c·δ_peak = h·β₀·(1+ε) → ε = c·|δ_peak|/(h·β₀) - 1
        foot_displacement = c_foot * abs(delta_peak)
        epsilon_lqr = foot_displacement / (h_CoM * beta0) - 1
        
        # T_rope/4
        T_rope = 2 * np.pi * np.sqrt((R_rope + h_CoM) / g)
        T_quarter = T_rope / 4
        
        # δ/β₀ 비율
        ratio_db = abs(delta_peak) / beta0
        
        print(f"  {R_rope:6.1f} {beta0:8.3f} {delta_peak:10.4f} {ratio_db:8.2f} {epsilon_lqr:8.3f} {t_peak:8.4f} {T_quarter:8.4f} {t_peak/T_quarter:10.3f}")

# === 이산 모델의 예측과 비교 ===
print(f"\n{'='*70}")
print("  이산 모델의 예측 vs LQR 실측")
print("=" * 70)
print(f"  이산 모델: δ/β₀ = h·(1+ε)/c = {h_CoM/c_foot:.2f}·(1+ε)")
print()

# 메인 식에서 ε를 구하기 (각 R에서)
from scipy.optimize import brentq

print(f"  {'R':>6} {'ω':>8} {'λ':>8} {'T_q/4':>8} {'ε_theory':>10} {'δ/β₀ (이론)':>12} {'δ/β₀ (LQR)':>12} {'차이':>8}")
print(f"  {'-'*75}")

# λ (역진자) 계산
I_foot = M1*(LC1**2 + L1**2/12) + m2*((L1+LC2)**2)  # 발목 기준
lambda_inv = np.sqrt(Mt * g * h_CoM / I_foot)

for R_rope in [0.3, 0.5, 0.7, 1.0, 1.5]:
    omega = np.sqrt(g / (R_rope + h_CoM))
    T_quarter = np.pi / (2 * omega)
    
    # 메인 식: (1+ε)·cos(ωT) = ε·cosh(λT)
    # T를 T_quarter보다 약간 작게 설정하고 ε 구하기
    # ε = (1+ε)·cos(ωT)/cosh(λT)
    # → ε·cosh(λT) = cos(ωT) + ε·cos(ωT)
    # → ε·(cosh(λT) - cos(ωT)) = cos(ωT)
    # → ε = cos(ωT) / (cosh(λT) - cos(ωT))
    
    # T를 여러 값으로 시도하여 합리적 ε 찾기
    best_T = None
    best_eps = None
    
    for T_frac in np.linspace(0.5, 0.99, 100):
        T_try = T_quarter * T_frac
        denom = np.cosh(lambda_inv * T_try) - np.cos(omega * T_try)
        if denom > 0:
            eps_try = np.cos(omega * T_try) / denom
            if 0.01 < eps_try < 1.0:
                if best_eps is None or abs(eps_try - 0.2) < abs(best_eps - 0.2):
                    best_eps = eps_try
                    best_T = T_try
    
    if best_eps is not None:
        delta_beta_theory = h_CoM * (1 + best_eps) / c_foot
        
        # LQR 시뮬에서 실측한 δ/β₀ (β₀=0.1)
        A_sys, B_sys, _ = compute_system(M1, m2, L1, LC1, LC2, R_rope)
        P = solve_continuous_are(A_sys, B_sys, Q, R_cost)
        K = (1/R_cost) * B_sys.T @ P
        A_cl = A_sys - B_sys @ K
        x0 = np.array([0, 0.1, 0.1, 0, 0, 0])
        sol = solve_ivp(lambda t,x: A_cl@x, [0,5], x0, max_step=0.001, dense_output=True)
        t_ev = np.linspace(0, 3, 3000)
        x_ev = sol.sol(t_ev)
        delta_lqr = np.max(np.abs(x_ev[2] - x_ev[1])) / 0.1
        
        diff_pct = (delta_beta_theory - delta_lqr) / delta_lqr * 100
        
        print(f"  {R_rope:6.1f} {omega:8.3f} {lambda_inv:8.3f} {T_quarter:8.4f} {best_eps:10.4f} {delta_beta_theory:12.2f} {delta_lqr:12.2f} {diff_pct:7.1f}%")

# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
H2 정교화 검증: 결합 모드의 유효 주파수가 MuJoCo T_cross를 설명하는가?

비교 모델:
  1. 분리 모델: (1+ε)cos(ωT) = εcosh(λT)  [원래 가설]
  2. 결합 2×2:  (phi,beta) 커플링 포함, delta 고정 가정
  3. 결합 3×3:  delta 자유진화 포함 (가장 정확한 선형 모델)

MuJoCo 참조 데이터: verify_pure_mujoco.py (ε=0.5, β₀=2°)

2026-06-08
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

# ============================================================
# 1. 시스템 파라미터 (사람 기준, modal_analysis와 동일)
# ============================================================
g = 9.81
M1 = 30.0;  m2 = 45.0;  L1 = 0.85
LC1 = 0.30; LC2 = 0.40;  L_upper = 0.85
Mt = M1 + m2  # 75 kg
I1 = M1 * L1**2 / 12
I2 = m2 * L_upper**2 / 12

p1 = M1 * LC1 + m2 * L1      # 47.25
p2 = m2 * LC2                 # 18.0
p3 = m2 * L1 * LC2            # 15.3
ps = p1 + p2                  # 65.25
h_com = ps / Mt               # 0.87 m
c_beta = Mt * h_com / ps      # 좌표 변환 스케일

J_aa = M1 * LC1**2 + I1 + m2 * L1**2
J_tt = m2 * LC2**2 + I2
J_at = p3

# 단일 강체 관성모멘트 (분리 모델용)
I_rigid = M1*LC1**2 + I1 + m2*(L1+LC2)**2 + I2  # 발목 기준 총 I

B_theta = 0.5  # 힙 감쇠

print("=" * 80)
print("  H2 정교화 검증: 분리 vs 결합 모드의 T_cross 비교")
print("=" * 80)
print(f"  파라미터: Mt={Mt}kg, h_com={h_com:.4f}m, I_rigid={I_rigid:.2f}")
print(f"  분리 모델: λ_orig = √(Mt·g·h/{I_rigid:.1f}) = {np.sqrt(Mt*g*h_com/I_rigid):.4f} rad/s")

# ============================================================
# 2. MuJoCo 참조 데이터
# ============================================================
mujoco_Tcross = {0.5: 0.092, 0.7: 0.103, 1.0: 0.124, 1.5: 0.148, 2.0: 0.167}

# ============================================================
# 3. 접기 파라미터
# ============================================================
eps = 0.5
beta0 = np.radians(2)

# 분리 모델 주파수
lambda_orig = np.sqrt(Mt * g * h_com / I_rigid)

print(f"\n  접기 파라미터: ε={eps}, β₀={np.degrees(beta0):.0f}°")
print(f"  분리 λ = {lambda_orig:.4f} rad/s (모든 R에서 동일)")

# ============================================================
# 4. 좌표 변환 행렬
# ============================================================
def get_T_inv():
    """(phi,alpha,theta) -> (phi,beta,delta) 변환"""
    return np.array([
        [1.0, 0.0,      0.0],
        [0.0, c_beta, -p2/ps],
        [0.0, c_beta,  p1/ps]])

T_inv_mat = get_T_inv()
T_mat = np.linalg.inv(T_inv_mat)

# ============================================================
# 5. 각 R에서 세 모델 비교
# ============================================================
print(f"\n{'='*80}")
print("  STEP 1: 각 R에서 행렬과 주파수 계산")
print(f"{'='*80}")

R_values = [0.5, 0.7, 1.0, 1.5, 2.0]
results = []

for R in R_values:
    print(f"\n--- R = {R} m ---")
    
    # (phi,alpha,theta) 좌표에서 M, G
    M = np.array([
        [Mt*R**2, R*p1,  R*p2],
        [R*p1,    J_aa,  J_at],
        [R*p2,    J_at,  J_tt]])
    G = np.diag([-Mt*g*R, p1*g, p2*g])
    D = np.diag([0.0, 0.0, -B_theta])
    M_inv = np.linalg.inv(M)
    
    # (phi,beta,delta) 좌표로 변환
    M_pbd = T_inv_mat.T @ M @ T_inv_mat
    G_pbd = T_inv_mat.T @ G @ T_inv_mat
    
    # --- 분리 모델 주파수 ---
    omega_orig = np.sqrt(g / (R + h_com))
    print(f"  분리 모델:  ω = {omega_orig:.4f},  λ = {lambda_orig:.4f}")
    
    # --- 결합 2×2 (M_22^{-1} * G_22, delta=상수 가정) ---
    M_22 = M_pbd[:2, :2]
    G_22 = G_pbd[:2, :2]
    A2_exact = np.linalg.solve(M_22, G_22)
    
    evals_2x2, evecs_2x2 = np.linalg.eig(A2_exact)
    idx = np.argsort(evals_2x2.real)
    evals_2x2 = evals_2x2[idx].real
    evecs_2x2 = evecs_2x2[:, idx].real
    
    omega_2x2 = np.sqrt(abs(evals_2x2[0]))
    lambda_2x2 = np.sqrt(abs(evals_2x2[1]))
    print(f"  결합 2×2:   ω_eff = {omega_2x2:.4f},  λ_eff = {lambda_2x2:.4f}")
    print(f"    비율: ω_eff/ω_orig = {omega_2x2/omega_orig:.3f},  λ_eff/λ_orig = {lambda_2x2/lambda_orig:.3f}")
    
    # --- 결합 3×3 (전체 동역학) ---
    A_full = np.linalg.solve(M_pbd, G_pbd)
    D_pbd = T_inv_mat.T @ D @ T_inv_mat  # 감쇠도 변환
    D_full = np.linalg.solve(M_pbd, D_pbd)
    
    evals_3x3 = np.linalg.eigvals(A_full).real
    evals_3x3_sorted = np.sort(evals_3x3)
    freqs_3x3 = [np.sqrt(abs(e)) for e in evals_3x3_sorted]
    print(f"  결합 3×3:   고유값 = [{evals_3x3_sorted[0]:.2f}, {evals_3x3_sorted[1]:.2f}, {evals_3x3_sorted[2]:.2f}]")
    print(f"              주파수 = [{freqs_3x3[0]:.4f}, {freqs_3x3[1]:.4f}, {freqs_3x3[2]:.4f}]")
    
    results.append({
        'R': R,
        'omega_orig': omega_orig, 'lambda_orig': lambda_orig,
        'omega_2x2': omega_2x2, 'lambda_2x2': lambda_2x2,
        'A2_exact': A2_exact, 'evecs_2x2': evecs_2x2, 'evals_2x2': evals_2x2,
        'A_full': A_full, 'D_full': D_full,
        'M_pbd': M_pbd, 'G_pbd': G_pbd,
    })

# ============================================================
# 6. T_cross 계산
# ============================================================
print(f"\n{'='*80}")
print("  STEP 2: 초기조건 설정 (해석적 모델의 순간접기)")
print(f"{'='*80}")
print(f"  접기 전: φ=0, β=β₀")
print(f"  접기 후: φ₀ = h*(1+ε)*β₀/R,  β₀_new = -ε*β₀")
print(f"  x_CoM = R*φ + h*β  (접기 전후 불변 확인)")
print()

for r in results:
    R = r['R']
    phi0 = h_com * (1+eps) * beta0 / R
    beta0_new = -eps * beta0
    
    x_before = R*0 + h_com*beta0
    x_after  = R*phi0 + h_com*beta0_new
    print(f"  R={R}: φ₀={np.degrees(phi0):.3f}°, β_new={np.degrees(beta0_new):.3f}°, "
          f"x_CoM: {x_before*1000:.2f}mm → {x_after*1000:.2f}mm (차이={abs(x_after-x_before)*1e6:.1f}μm)")

print(f"\n{'='*80}")
print("  STEP 3: T_cross 계산 — 세 모델 비교")
print(f"{'='*80}")

# x_CoM = R*phi + h_com*beta = 0 인 시점 찾기
# c_xcom = [R, h_com] (x_CoM = c_xcom · q)

for r in results:
    R = r['R']
    
    # 초기조건 (phi, beta)
    phi0 = h_com * (1+eps) * beta0 / R
    beta0_new = -eps * beta0
    q0_2 = np.array([phi0, beta0_new])
    
    # x_CoM 계수 벡터
    c_xcom_2 = np.array([R, h_com])
    
    # (a) 분리 모델
    omega = r['omega_orig']; lam = r['lambda_orig']
    def xcom_sep(t):
        return R * phi0 * np.cos(omega*t) + h_com * beta0_new * np.cosh(lam*t)
    try:
        T_sep = brentq(xcom_sep, 1e-4, 3.0)
    except:
        T_sep = np.nan
    
    # (b) 결합 2×2 (delta 고정)
    V = r['evecs_2x2']
    V_inv = np.linalg.inv(V)
    ev = r['evals_2x2']
    q0_mode = V_inv @ q0_2  # 초기조건의 모드 좌표 투영
    
    def xcom_coupled2(t):
        # 모드 좌표에서 시간 진화
        q_mode_t = np.array([
            q0_mode[0] * np.cos(np.sqrt(abs(ev[0])) * t),   # 안정 모드
            q0_mode[1] * np.cosh(np.sqrt(abs(ev[1])) * t),  # 불안정 모드
        ])
        q_t = V @ q_mode_t  # 물리 좌표로 복원
        return c_xcom_2 @ q_t
    try:
        T_c2 = brentq(xcom_coupled2, 1e-4, 3.0)
    except:
        T_c2 = np.nan
    
    # (c) 결합 3×3 (delta 자유진화, 감쇠 포함)
    # 6차 상태공간: [phi, beta, delta, phi_d, beta_d, delta_d]
    Ac = np.zeros((6, 6))
    Ac[:3, 3:] = np.eye(3)
    Ac[3:, :3] = r['A_full']
    Ac[3:, 3:] = r['D_full']
    
    # 초기조건: delta_fold 계산
    # fold 전: alpha=theta=c_beta*beta0, delta=0
    # fold 후: phi 변화, beta 변화, delta 변화
    # delta_fold = theta_new - alpha_new
    # beta_new에서 alpha_new, theta_new 역산:
    #   alpha_new = c_beta*beta_new - (p2/ps)*delta_new
    #   theta_new = c_beta*beta_new + (p1/ps)*delta_new
    # 또한 phi_new에서 줄 위 발의 위치가 변해야 하므로...
    # 일단 delta_fold는 "접힘에 필요한 각도"
    # 해석적으로: x_foot 이동 = c_foot * delta, c_foot ≈ p2/Mt
    c_foot = p2 / Mt  # 0.24
    x_foot_move = h_com * beta0 * (1 + eps)  # 발 이동량
    delta_fold = x_foot_move / c_foot
    
    x0_3 = np.zeros(6)
    x0_3[0] = phi0
    x0_3[1] = beta0_new
    x0_3[2] = delta_fold
    # 속도 = 0 (순간접기)
    
    c_xcom_3 = np.zeros(6)
    c_xcom_3[0] = R
    c_xcom_3[1] = h_com
    
    def xcom_full3(t):
        state_t = expm(Ac * t) @ x0_3
        return c_xcom_3 @ state_t
    try:
        T_c3 = brentq(xcom_full3, 1e-4, 3.0)
    except:
        T_c3 = np.nan
    
    # MuJoCo 참조
    T_mj = mujoco_Tcross.get(R, np.nan)
    
    # 결과 저장
    r['T_sep'] = T_sep
    r['T_c2'] = T_c2
    r['T_c3'] = T_c3
    r['T_mj'] = T_mj
    r['q0_mode'] = q0_mode

# ============================================================
# 7. 결과 출력
# ============================================================
print(f"\n{'='*80}")
print("  최종 결과: T_cross 비교")
print(f"{'='*80}")
print(f"\n  {'R':>4} | {'T_sep':>7} | {'T_2x2':>7} | {'T_3x3':>7} | {'T_MuJoCo':>8} | {'sep/MJ':>6} | {'2x2/MJ':>7} | {'3x3/MJ':>7}")
print(f"  {'-'*75}")

for r in results:
    T_s, T_2, T_3, T_m = r['T_sep'], r['T_c2'], r['T_c3'], r['T_mj']
    r_sm = T_s/T_m if T_m else 0
    r_2m = T_2/T_m if T_m else 0
    r_3m = T_3/T_m if T_m else 0
    print(f"  {r['R']:4.1f} | {T_s:7.4f} | {T_2:7.4f} | {T_3:7.4f} | {T_m:8.4f} | {r_sm:6.2f}x | {r_2m:7.2f}x | {r_3m:7.2f}x")

print(f"\n{'='*80}")
print("  분석: 모드 투영 (초기조건이 각 모드에 얼마나 들어가는지)")
print(f"{'='*80}")

for r in results:
    q0m = r['q0_mode']
    print(f"  R={r['R']}: 안정모드(cos) = {q0m[0]:.6f},  불안정모드(cosh) = {q0m[1]:.6f},  "
          f"비율 = {abs(q0m[0]/q0m[1]):.3f}")

print(f"\n{'='*80}")
print("  유효 주파수 요약")
print(f"{'='*80}")
print(f"\n  {'R':>4} | {'ω_orig':>7} {'ω_2x2':>7} {'비율':>5} | {'λ_orig':>7} {'λ_2x2':>7} {'비율':>5}")
print(f"  {'-'*55}")
for r in results:
    wr = r['omega_2x2']/r['omega_orig']
    lr = r['lambda_2x2']/r['lambda_orig']
    print(f"  {r['R']:4.1f} | {r['omega_orig']:7.4f} {r['omega_2x2']:7.4f} {wr:5.2f}x | "
          f"{r['lambda_orig']:7.4f} {r['lambda_2x2']:7.4f} {lr:5.2f}x")

print(f"\n  Done!")

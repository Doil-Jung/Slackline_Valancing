# -*- coding: utf-8 -*-
"""
가설 검증: H1이 LQR 이득 행렬 K에 남기는 흔적

예측:
  H1(p1=p2) 성립 시 → 토크가 δ=θ-α를 제어 → k2≈-k3, k5≈-k6
  H1 불만족 시 → k2와 k3가 독립적

검증 방법:
  1. p1/p2 비율을 변화시키며 K 행렬 계산
  2. k2/k3 비율과 k5/k6 비율 관찰
  3. R을 변화시키며 k1의 변화 관찰 (H3 연결)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.linalg import solve_continuous_are

g = 9.81

def compute_system_matrices(M1, m2_eff, L1, LC1, LC2, R):
    """선형화된 상태공간 모델 (A, B) 계산"""
    Mt = M1 + m2_eff
    p1 = M1*LC1 + m2_eff*L1
    p2 = m2_eff*LC2
    p3 = m2_eff*L1*LC2
    I_alpha = M1*LC1**2 + M1*L1**2/12 + m2_eff*L1**2
    I_theta = m2_eff*LC2**2 + m2_eff*LC2**2  # simplified
    
    # 질량 행렬
    M_mat = np.array([
        [Mt*R**2, R*p1, R*p2],
        [R*p1, I_alpha, p3],
        [R*p2, p3, I_theta]
    ])
    
    # 중력 강성 행렬 (선형화)
    G_mat = np.array([
        [-Mt*g*R, -p1*g, -p2*g],
        [0, p1*g, p2*g],
        [0, 0, p2*g]
    ])
    # 수정: 정확한 선형화
    G_mat = np.diag([-Mt*g*R, p1*g, p2*g])
    
    E = np.array([0, -1, 1])
    
    Minv = np.linalg.inv(M_mat)
    
    # 상태공간: x = [φ,α,θ,φ̇,α̇,θ̇]
    A = np.zeros((6,6))
    A[0:3, 3:6] = np.eye(3)
    A[3:6, 0:3] = Minv @ G_mat
    
    B = np.zeros((6,1))
    B[3:6, 0] = Minv @ E
    
    return A, B, M_mat, Minv, E, {'Mt':Mt, 'p1':p1, 'p2':p2, 'p3':p3}

def compute_lqr(A, B, Q, R_cost):
    """LQR 이득 행렬 K 계산"""
    P = solve_continuous_are(A, B, Q, R_cost)
    K = (1/R_cost) * B.T @ P
    return K[0]  # 1×6 → 6

# === 기본 파라미터 ===
L1 = 0.85; LC1 = L1/2
L2 = 0.80; LC2_raw = L2/2
M_HEAD = 5.0; HEAD_R = L2/4

# LQR 비용 함수
Q = np.diag([100, 100, 100, 1, 1, 1])
R_cost = 0.01

print("=" * 70)
print("  검증 1: p1/p2 비율에 따른 K 행렬 구조")
print("=" * 70)
print(f"  H1 예측: p1=p2일 때 k2≈-k3, k5≈-k6")
print()

# p1/p2를 변화시키기 위해 M1을 조정
# p1 = M1*LC1 + m2*L1,  p2 = m2*LC2
# p1 = p2 → M1*LC1 + m2*L1 = m2*LC2 → M1 = m2*(LC2 - L1)/LC1

M2_base = 40.0
m2_base = M2_base + M_HEAD
LC2_base = (M2_base*LC2_raw + M_HEAD*(L2 + HEAD_R)) / m2_base

# p1=p2 조건의 M1
M1_equal = m2_base * (LC2_base - L1) / LC1
print(f"  p1=p2 조건의 M1: {M1_equal:.2f} kg")
print(f"  (LC2={LC2_base:.4f}, L1={L1}, LC1={LC1})")
if M1_equal < 0:
    print(f"  ※ M1 < 0 → p1=p2는 현재 체형에서 물리적으로 불가능")
    print(f"     (L1={L1} > LC2={LC2_base:.4f}이므로 p1 > p2 항상 성립)")
    # 대신 m2를 조정하여 p1/p2를 변화
    print(f"\n  → m2를 조정하여 p1/p2 비율 변화시킴")

R_rope = 1.0  # 고정

print(f"\n  R = {R_rope} m 고정")
print(f"\n  {'p1/p2':>8} {'k1(φ)':>10} {'k2(α)':>10} {'k3(θ)':>10} {'k2/k3':>8} {'k5(α̇)':>10} {'k6(θ̇)':>10} {'k5/k6':>8}")
print(f"  {'-'*80}")

# M1을 변화시키며 관찰
for M1_test in [10, 15, 20, 25, 30, 35, 40, 50, 60]:
    try:
        A, B, M_mat, Minv, E, params = compute_system_matrices(
            M1_test, m2_base, L1, LC1, LC2_base, R_rope)
        K = compute_lqr(A, B, Q, R_cost)
        p1 = params['p1']
        p2 = params['p2']
        ratio_p = p1/p2
        ratio_k23 = K[1]/K[2] if abs(K[2]) > 1e-10 else float('inf')
        ratio_k56 = K[4]/K[5] if abs(K[5]) > 1e-10 else float('inf')
        marker = " ◀ p1≈p2" if abs(ratio_p - 1.0) < 0.1 else ""
        print(f"  {ratio_p:8.3f} {K[0]:10.4f} {K[1]:10.4f} {K[2]:10.4f} {ratio_k23:8.3f} {K[4]:10.4f} {K[5]:10.4f} {ratio_k56:8.3f}{marker}")
    except Exception as e:
        print(f"  M1={M1_test}: 실패 ({e})")

# === 검증 2: (φ,σ,δ) 좌표에서 K 변환 ===
print(f"\n{'='*70}")
print("  검증 2: (φ, σ, δ) 좌표에서의 K 행렬")
print("=" * 70)
print("  σ = p1·α + p2·θ (CoM),  δ = θ - α (접힘)")
print("  H1 예측: p1=p2일 때 K_σ ≈ 0, K_σ̇ ≈ 0")
print()

M1_default = 30.0
for M1_test in [15, 20, 25, 30, 40]:
    try:
        A, B, M_mat, Minv, E, params = compute_system_matrices(
            M1_test, m2_base, L1, LC1, LC2_base, R_rope)
        K = compute_lqr(A, B, Q, R_cost)
        p1 = params['p1']
        p2 = params['p2']
        
        # K = [k1,k2,k3,k4,k5,k6] in (φ,α,θ,φ̇,α̇,θ̇)
        # τ = -(k1φ + k2α + k3θ + ...)
        # 좌표 변환: α = (p2·σ + ... ) 역변환은 복잡하므로
        # 직접 변환: σ = p1α + p2θ, δ = θ - α
        # 역변환: α = (p2·σ - σ·... ) 대신 간단히
        # τ = k1φ + k2α + k3θ = k1φ + (k2+k3)·(중간값) + (k3-k2)·(δ관련)
        # 정확한 역변환: α = (1/(p1+p2))·(p2·σ - p2·δ·(뭔가))
        
        # 더 간단: δ = θ-α, 그리고 α = θ-δ
        # τ = k1φ + k2α + k3θ = k1φ + k2(θ-δ) + k3θ 
        #   = k1φ + (k2+k3)θ - k2δ
        # 이건 아직 (φ,θ,δ)이지 (φ,σ,δ)가 아님
        
        # 정확한 변환:
        # σ = p1α + p2θ, δ = θ - α
        # → α = (p2·... ) 풀면:
        # α = (σ - p2·δ·... ) 이건 아니고
        # σ = p1α + p2(α+δ) = (p1+p2)α + p2δ
        # → α = (σ - p2δ)/(p1+p2)
        # → θ = α + δ = (σ - p2δ)/(p1+p2) + δ = (σ + p1δ)/(p1+p2)
        
        # τ = k1φ + k2·(σ-p2δ)/(p1+p2) + k3·(σ+p1δ)/(p1+p2)
        #   = k1φ + (k2+k3)/(p1+p2)·σ + (-k2p2+k3p1)/(p1+p2)·δ
        
        K_phi = K[0]
        K_sigma = (K[1] + K[2]) / (p1 + p2)
        K_delta = (-K[1]*p2 + K[2]*p1) / (p1 + p2)
        
        # 속도 항도 동일 변환
        K_phi_dot = K[3]
        K_sigma_dot = (K[4] + K[5]) / (p1 + p2)
        K_delta_dot = (-K[4]*p2 + K[5]*p1) / (p1 + p2)
        
        ratio_p = p1/p2
        print(f"  p1/p2={ratio_p:.2f}: K_φ={K_phi:8.3f}  K_σ={K_sigma:8.4f}  K_δ={K_delta:8.3f}  "
              f"K_φ̇={K_phi_dot:8.3f}  K_σ̇={K_sigma_dot:8.4f}  K_δ̇={K_delta_dot:8.3f}")
    except Exception as e:
        print(f"  M1={M1_test}: 실패 ({e})")

# === 검증 3: R에 따른 K 변화 ===
print(f"\n{'='*70}")
print("  검증 3: R에 따른 K 행렬 변화 (H3 연결)")
print("=" * 70)
print("  H3 예측: 특정 R에서 K_φ가 최소화 (접힘이 줄을 안 흔들면 φ 제어 부담 감소)")
print()

M1_test = 30.0
print(f"  M1 = {M1_test} kg 고정")
print(f"\n  {'R [m]':>8} {'K_φ':>10} {'K_σ':>10} {'K_δ':>10} {'|K_σ|/|K_δ|':>12} {'K_φ̇':>10}")
print(f"  {'-'*60}")

for R_test in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
    try:
        A, B, M_mat, Minv, E, params = compute_system_matrices(
            M1_test, m2_base, L1, LC1, LC2_base, R_test)
        K = compute_lqr(A, B, Q, R_cost)
        p1 = params['p1']
        p2 = params['p2']
        
        K_phi = K[0]
        K_sigma = (K[1] + K[2]) / (p1 + p2)
        K_delta = (-K[1]*p2 + K[2]*p1) / (p1 + p2)
        K_phi_dot = K[3]
        
        ratio_sd = abs(K_sigma)/abs(K_delta) if abs(K_delta) > 1e-10 else float('inf')
        print(f"  {R_test:8.2f} {K_phi:10.4f} {K_sigma:10.4f} {K_delta:10.4f} {ratio_sd:12.4f} {K_phi_dot:10.4f}")
    except Exception as e:
        print(f"  R={R_test}: 실패 ({e})")

# === 검증 4: 가제어성 행렬의 조건수 ===
print(f"\n{'='*70}")
print("  검증 4: R에 따른 가제어성 행렬 조건수")
print("=" * 70)
print("  조건수가 작을수록 제어가 '쉬움' — H3가 성립하는 R에서 최소 예상")
print()

print(f"  {'R [m]':>8} {'rank(C)':>8} {'cond(C)':>15} {'det(C·Cᵀ)':>15}")
print(f"  {'-'*50}")

for R_test in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
    try:
        A, B, M_mat, Minv, E, params = compute_system_matrices(
            M1_test, m2_base, L1, LC1, LC2_base, R_test)
        
        # 가제어성 행렬: C = [B, AB, A²B, ..., A⁵B]
        C = B.copy()
        AB = B.copy()
        for i in range(5):
            AB = A @ AB
            C = np.hstack([C, AB])
        
        rank = np.linalg.matrix_rank(C)
        cond = np.linalg.cond(C)
        det_CCt = np.linalg.det(C @ C.T)
        
        print(f"  {R_test:8.2f} {rank:8d} {cond:15.2f} {det_CCt:15.4e}")
    except Exception as e:
        print(f"  R={R_test}: 실패 ({e})")

print(f"\n{'='*70}")
print("  요약")
print(f"{'='*70}")
print("  1. K 행렬의 (φ,σ,δ) 변환에서 K_σ가 작을수록 H1에 가까움")
print("  2. K_φ가 작아지는 R이 있다면 → H3 조건의 후보")  
print("  3. 가제어성 조건수가 최소인 R → 제어가 가장 '쉬운' R")

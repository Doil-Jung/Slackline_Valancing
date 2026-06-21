"""
LQR K행렬 계산 — v17 부호규약 + v11_small 파라미터
======================================================
좌표계 (v17 규약):
  φ : 원호 위 발 각도
  α : 하체 기울기 (연직 기준, 시계방향 +)
  θ : 상체 기울기 (연직 기준, 시계방향 +)
  δ = θ − α : 힙 관절각 (앞으로 접기 = +)
  τ : 힙 토크 (α에 −τ, θ에 +τ)

3-DOF 라그랑주 모델 (v17 model.js 동일):
  M(q)q̈ = f(q, q̇, τ)
  
질량행렬 M (평형점 q=0 근방 선형화):
  M₁₁ = (m₁+m₂)R²
  M₁₂ = R·p₁     (cos(φ+α)→1)
  M₁₃ = R·p₂     (cos(φ+θ)→1)
  M₂₂ = m₁ℓ₁²+I₁+m₂L₁²
  M₂₃ = p₃        (cos(α−θ)→1)
  M₃₃ = m₂ℓ₂²+I₂
"""
import numpy as np
from scipy.linalg import solve_discrete_are, expm

# ============================================================
#  v11_small 물리 파라미터 (소형 로봇)
# ============================================================
m1 = 0.10       # 하체 질량 (kg)
m2 = 0.30       # 상체 질량 (kg)
L1 = 0.20       # 하체 길이 (m) — 발→힙
L2 = 0.25       # 상체 길이 (m) — 힙→머리
R  = 0.30       # 원호 반경 = 줄 처짐 깊이 (m)
g  = 9.81       # 중력 가속도 (m/s²)
B_THETA = 0.01  # 상체 점성 감쇠 (N·m·s/rad)
TAU_MAX = 4.0   # XM430 최대 토크 (N·m)
DT = 0.001      # 이산화 시간 간격 (s)

# ============================================================
#  파생 상수 (v17 model.js 동일)
# ============================================================
ell1 = L1 / 2           # 하체 CoM 거리
ell2 = L2 / 2           # 상체 CoM 거리
I1 = m1 * L1**2 / 12    # 하체 관성모멘트
I2 = m2 * L2**2 / 12    # 상체 관성모멘트
Mt = m1 + m2             # 전체 질량
p1 = m1 * ell1 + m2 * L1  # m₁ℓ₁ + m₂L₁
p2 = m2 * ell2             # m₂ℓ₂
p3 = m2 * L1 * ell2        # m₂L₁ℓ₂

print("=" * 60)
print("  v17 규약 LQR 계산 (v11_small 파라미터)")
print("=" * 60)
print(f"m1={m1} kg, m2={m2} kg, L1={L1} m, L2={L2} m")
print(f"R={R} m, g={g} m/s², TAU_MAX={TAU_MAX} N·m")
print(f"ell1={ell1:.4f}, ell2={ell2:.4f}")
print(f"I1={I1:.6f}, I2={I2:.6f}")
print(f"Mt={Mt:.4f}, p1={p1:.4f}, p2={p2:.4f}, p3={p3:.4f}")

# ============================================================
#  평형점 선형화: M_eq * q̈ = G_eq * q + D_eq * q̇ + E * τ
# ============================================================
# 질량행렬 (q=0에서)
M_eq = np.array([
    [Mt * R**2,                R * p1,         R * p2],
    [R * p1,    m1*ell1**2 + I1 + m2*L1**2,    p3],
    [R * p2,                   p3,    m2*ell2**2 + I2]
])

# 중력 행렬 (sin → 선형화)
# f₁에서: −Mt·g·R·sinφ → −Mt·g·R·φ
# f₂에서: +p₁·g·sinα → +p₁·g·α  (불안정: 도립진자!)
# f₃에서: +p₂·g·sinθ → +p₂·g·θ  (불안정: 도립진자!)
G_mat = np.array([
    [-Mt * g * R,    0,       0],
    [0,         p1 * g,       0],
    [0,              0,  p2 * g]
])

# 감쇠 행렬
D_mat = np.diag([0, 0, -B_THETA])

# 토크 입력 벡터: τ → [0, −τ, +τ] (v17 규약)
E_mat = np.array([[0], [-1], [1]])

print(f"\nM_eq =\n{M_eq}")
print(f"G_mat diag = [{-Mt*g*R:.4f}, {p1*g:.4f}, {p2*g:.4f}]")

# ============================================================
#  연속 상태공간: ẋ = Ac·x + Bc·u
#  x = [φ, α, θ, φ̇, α̇, θ̇],  u = τ
# ============================================================
Mi = np.linalg.inv(M_eq)
n = 6

Ac = np.zeros((n, n))
Bc = np.zeros((n, 1))

Ac[:3, 3:] = np.eye(3)          # ẋ₁₋₃ = q̇
Ac[3:, :3] = Mi @ G_mat         # q̈ (중력 선형화)
Ac[3:, 3:] = Mi @ D_mat         # q̈ (감쇠)
Bc[3:, :]  = Mi @ E_mat         # q̈ (토크 입력)

print(f"\n연속 시스템 고유치:")
eigs_cont = np.linalg.eigvals(Ac)
for e in sorted(eigs_cont, key=lambda x: x.real):
    print(f"  {e:.4f}")

# ============================================================
#  이산화 (ZOH: matrix exponential)
# ============================================================
Z = np.zeros((n + 1, n + 1))
Z[:n, :n] = Ac * DT
Z[:n, n:] = Bc * DT
eZ = expm(Z)
Ad = eZ[:n, :n]
Bd = eZ[:n, n:n + 1]

# ============================================================
#  LQR 설계
# ============================================================
# Q: 상태 가중치 [φ, α, θ, φ̇, α̇, θ̇]
Q_DIAG = [1, 50, 50, 0.1, 5, 5]
R_COST = 0.001

Q = np.diag(Q_DIAG)
Rc = np.array([[R_COST]])

# 이산 리카티 방정식
P = solve_discrete_are(Ad, Bd, Q, Rc)
K = np.linalg.inv(Rc + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

# 폐루프 안정성
eigs_cl = sorted(abs(e) for e in np.linalg.eigvals(Ad - Bd @ K))

print(f"\n{'=' * 60}")
print(f"  LQR 결과")
print(f"{'=' * 60}")
print(f"Q = diag({Q_DIAG})")
print(f"R = {R_COST}")
print(f"\nK = [{', '.join(f'{k:.6f}' for k in K[0])}]")
print(f"    [K_phi, K_alpha, K_theta, K_phiDot, K_alphaDot, K_thetaDot]")
print(f"\n폐루프 |고유치|: {[f'{e:.4f}' for e in eigs_cl]}")
print(f"Max |eig| = {max(eigs_cl):.6f} → {'✅ STABLE' if max(eigs_cl) < 1.0 else '❌ UNSTABLE!'}")

# ============================================================
#  Arduino 출력 (v17 부호규약 그대로)
# ============================================================
print(f"\n{'=' * 60}")
print(f"  Arduino 하드코딩용 (v17 부호규약)")
print(f"{'=' * 60}")

# v17 규약에서는 phi를 직접 관측할 수 없으므로,
# 운동학 추정: φ ≈ −(C₁·α + C₂·θ)/R  (CoM ≈ 발 위 구속)
C1 = (m1 * ell1 + m2 * L1) / Mt  # = p1/Mt
C2 = (m2 * ell2) / Mt             # = p2/Mt
print(f"\nφ 운동학 추정 계수:")
print(f"  C1 = p1/Mt = {C1:.6f}")
print(f"  C2 = p2/Mt = {C2:.6f}")
print(f"  φ ≈ −(C1·α + C2·θ) / R")

print(f"\n// ===== LQR 게인 (v17 규약, v11_small 파라미터) =====")
print(f"// τ = −K·x = −(K₀φ + K₁α + K₂θ + K₃φ̇ + K₄α̇ + K₅θ̇)")
print(f"// v17 부호: τ → α에 −τ, θ에 +τ")
print(f"const float K_LQR[6] = {{{', '.join(f'{k:.4f}f' for k in K[0])}}};")
print(f"const float TAU_MAX = {TAU_MAX}f;")
print(f"")
print(f"// φ 운동학 추정 (IMU로 직접 관측 불가)")
print(f"const float C1_phi = {C1:.6f}f;  // (m1*ell1 + m2*L1) / Mt")
print(f"const float C2_phi = {C2:.6f}f;  // m2*ell2 / Mt")
print(f"const float R_arc  = {R:.2f}f;")
print(f"// phi_est = -(C1_phi * alpha + C2_phi * theta) / R_arc;")
print(f"// phiDot_est = -(C1_phi * alphaDot + C2_phi * thetaDot) / R_arc;")

# ============================================================
#  칼만 필터 게인 L (phi 추정 보강)
# ============================================================
print(f"\n{'=' * 60}")
print(f"  칼만 필터 관측기 게인 L")
print(f"{'=' * 60}")

m = 4  # 관측: [α, θ, α̇, θ̇]
C_obs = np.zeros((m, n))
C_obs[0, 1] = 1  # α
C_obs[1, 2] = 1  # θ
C_obs[2, 4] = 1  # α̇
C_obs[3, 5] = 1  # θ̇

# 관측가능성
O = np.vstack([C_obs @ np.linalg.matrix_power(Ad, k) for k in range(n)])
rank_O = np.linalg.matrix_rank(O)
print(f"관측가능성 rank = {rank_O} / {n} → {'✅ 관측가능' if rank_O == n else '❌ NOT Observable'}")

# 노이즈 공분산
Q_kf = np.diag([1e-4, 1e-6, 1e-6, 1e-3, 1e-5, 1e-5])
R_kf = np.diag([1e-4, 1e-4, 1e-3, 1e-3])

P_kf = solve_discrete_are(Ad.T, C_obs.T, Q_kf, R_kf)
L = P_kf @ C_obs.T @ np.linalg.inv(C_obs @ P_kf @ C_obs.T + R_kf)

eigs_obs = sorted(abs(e) for e in np.linalg.eigvals(Ad - L @ C_obs))
print(f"관측기 |고유치|: {[f'{e:.4f}' for e in eigs_obs]}")
print(f"Max |eig| = {max(eigs_obs):.6f} → {'✅ STABLE' if max(eigs_obs) < 1.0 else '❌ UNSTABLE!'}")

print(f"\n// ===== 칼만 게인 L (6×4) =====")
print(f"// L[i][j]: 행=상태(φ,α,θ,φ̇,α̇,θ̇), 열=관측(α,θ,α̇,θ̇)")
print(f"const float L_KF[6][4] = {{")
for i in range(n):
    vals = ', '.join(f'{L[i,j]:.6f}f' for j in range(m))
    comma = ',' if i < n - 1 else ''
    names = ['phi', 'alpha', 'theta', 'phiDot', 'alphaDot', 'thetaDot']
    print(f"  {{{vals}}}{comma}  // {names[i]}")
print(f"}};")

# Ad, Bd 출력
print(f"\n// ===== 선형 모델 행렬 (칼만 예측용) =====")
print(f"const float Ad_mat[6][6] = {{")
for i in range(n):
    vals = ', '.join(f'{Ad[i,j]:.8f}f' for j in range(n))
    comma = ',' if i < n - 1 else ''
    print(f"  {{{vals}}}{comma}")
print(f"}};")

print(f"\nconst float Bd_vec[6] = {{{', '.join(f'{Bd[i,0]:.8f}f' for i in range(n))}}};")

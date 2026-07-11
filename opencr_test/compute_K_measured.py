"""
LQR K행렬 계산 — 실측 파라미터 버전 (전류제어 대응)
======================================================
compute_K.py 개선판:
  1. I, ℓ 을 균일막대 가정(I=mL²/12, ℓ=L/2)에서 분리 → 실측값 직접 입력
  2. 복합진자 측정값(m, d, T) → I_cm 자동 계산 보조함수 제공
  3. TAU_MAX = 3.0 (XM430-W210 스톨, 기존 4.0은 과대)
  4. 토크 τ → Goal Current(unit) 변환 상수 출력 (전류제어 펌웨어용)

좌표계 (v17 규약):
  φ : 원호 위 발 각도
  α : 하체 기울기 (연직 기준, 시계방향 +)
  θ : 상체 기울기 (연직 기준, 시계방향 +)
  δ = θ − α : 힙 관절각 (앞으로 접기 = +)
  τ : 힙 토크 (α에 −τ, θ에 +τ),  τ = −K·x

CoM 기준점 주의:
  ℓ₁ = 발 → 하체 CoM 거리
  ℓ₂ = 힙 → 상체 CoM 거리
  I₁ = 하체 자기 CoM 기준 관성모멘트
  I₂ = 상체 자기 CoM 기준 관성모멘트
"""
import numpy as np
from scipy.linalg import solve_discrete_are, expm

# ============================================================
#  복합진자 측정 → I_cm 보조 함수
# ============================================================
def I_cm_from_pendulum(m, d, T, g=9.81):
    """복합진자 측정값으로 CoM 기준 관성모멘트 계산.
    m: 질량[kg], d: 피벗→CoM 거리[m], T: 미소진동 주기[s]
    I_pivot = m·g·d·(T/2π)² ,  I_cm = I_pivot − m·d²
    반환: (I_cm, I_pivot)
    """
    I_pivot = m * g * d * (T / (2 * np.pi)) ** 2
    I_cm = I_pivot - m * d ** 2
    return I_cm, I_pivot


# ============================================================
#  ★★★ MEASURED 블록 — 여기에 실측값을 입력 ★★★
#  (아래는 예시값: 측정 전 동작확인용. 실측 후 교체할 것)
# ============================================================
USE_PENDULUM = False   # True 면 아래 진자 측정값으로 I₁,I₂ 자동계산

# --- 질량 [kg] ---
m1 = 0.10       # 하체
m2 = 0.30       # 상체 (모터·보드·배터리 포함)

# --- 길이 [m] ---
L1 = 0.20       # 발 → 힙
L2 = 0.25       # 힙 → 상체끝

# --- CoM 거리 [m] (★기준점 다름: ℓ₁은 발, ℓ₂는 힙) ---
ell1 = 0.10     # 발 → 하체 CoM   (균일막대면 L1/2=0.10)
ell2 = 0.125    # 힙 → 상체 CoM   (균일막대면 L2/2=0.125; 배터리 상단이면 더 큼)

# --- 관성모멘트 [kg·m²] (자기 CoM 기준) ---
if USE_PENDULUM:
    # 복합진자 측정값 입력: (질량, 피벗→CoM거리 d, 주기 T)
    I1, _ = I_cm_from_pendulum(m=m1, d=0.10, T=0.0)   # ← 하체 측정값 입력
    I2, _ = I_cm_from_pendulum(m=m2, d=0.125, T=0.0)  # ← 상체 측정값 입력
else:
    I1 = m1 * L1**2 / 12    # 균일막대 근사 (실측 전 임시)
    I2 = m2 * L2**2 / 12

# --- 환경/줄 ---
R  = 0.30       # 원호 반경 = 줄 처짐 깊이 [m]
g  = 9.81
B_THETA = 0.01  # 상체 점성 감쇠 [N·m·s/rad]

# --- 모터 (XM430-W210-R 전류제어) ---
TAU_MAX  = 3.0      # 스톨 토크 [N·m] @12V (연속운전은 2.0~2.5 권장)
KT       = 1.304    # 토크상수 [N·m/A] (공칭; 토크보정으로 실측 교체)
CUR_UNIT = 2.69     # Goal/Present Current 단위 [mA/unit]
DT = 0.002          # 제어 주기 [s] (lqr_balance.ino = 500Hz)

# ============================================================
#  LQR 가중치 (튜닝 파라미터)
# ============================================================
Q_DIAG = [1, 50, 50, 0.1, 5, 5]   # [φ, α, θ, φ̇, α̇, θ̇]
R_COST = 0.001

# ============================================================
#  파생 상수
# ============================================================
Mt = m1 + m2
p1 = m1 * ell1 + m2 * L1   # 발 기준 1차 모멘트
p2 = m2 * ell2             # 힙 기준 상체 1차 모멘트
p3 = m2 * L1 * ell2        # 커플링

print("=" * 60)
print("  실측 LQR 계산 (v17 규약, 전류제어 대응)")
print("=" * 60)
print(f"m1={m1} kg, m2={m2} kg, L1={L1} m, L2={L2} m")
print(f"ell1={ell1:.4f} m (발기준), ell2={ell2:.4f} m (힙기준)")
print(f"I1={I1:.6e}, I2={I2:.6e}  (균일막대 참고: {m1*L1**2/12:.3e}, {m2*L2**2/12:.3e})")
print(f"R={R} m, TAU_MAX={TAU_MAX} N·m, Kt={KT} N·m/A, DT={DT*1000:.0f}ms")
print(f"Mt={Mt:.4f}, p1={p1:.4f}, p2={p2:.4f}, p3={p3:.4f}")

# ============================================================
#  평형점 선형화: M_eq·q̈ = G·q + D·q̇ + E·τ
# ============================================================
M_eq = np.array([
    [Mt * R**2,                   R * p1,            R * p2],
    [R * p1,    m1*ell1**2 + I1 + m2*L1**2,             p3],
    [R * p2,                         p3,    m2*ell2**2 + I2]
])

G_mat = np.array([
    [-Mt * g * R,    0,       0],
    [0,         p1 * g,       0],
    [0,              0,  p2 * g]
])

D_mat = np.diag([0, 0, -B_THETA])
E_mat = np.array([[0], [-1], [1]])   # τ → α에 −τ, θ에 +τ

# 질량행렬 양정치성 검사 (물리적 타당성)
eig_M = np.linalg.eigvalsh(M_eq)
print(f"\nM_eq 고유치 = {eig_M}  → {'✅ 양정치(OK)' if np.all(eig_M > 0) else '❌ 비물리! 파라미터 점검'}")

# ============================================================
#  연속 상태공간
# ============================================================
Mi = np.linalg.inv(M_eq)
n = 6
Ac = np.zeros((n, n))
Bc = np.zeros((n, 1))
Ac[:3, 3:] = np.eye(3)
Ac[3:, :3] = Mi @ G_mat
Ac[3:, 3:] = Mi @ D_mat
Bc[3:, :]  = Mi @ E_mat

print("\n연속 시스템 고유치 (불안정극 존재 = 도립진자 정상):")
for e in sorted(np.linalg.eigvals(Ac), key=lambda x: x.real):
    print(f"  {e:.4f}")

# ============================================================
#  이산화 (ZOH)
# ============================================================
Z = np.zeros((n + 1, n + 1))
Z[:n, :n] = Ac * DT
Z[:n, n:] = Bc * DT
eZ = expm(Z)
Ad = eZ[:n, :n]
Bd = eZ[:n, n:n + 1]

# ============================================================
#  가제어성 / LQR
# ============================================================
Ctrb = np.hstack([np.linalg.matrix_power(Ad, k) @ Bd for k in range(n)])
print(f"\n가제어성 rank = {np.linalg.matrix_rank(Ctrb)}/{n} "
      f"→ {'✅' if np.linalg.matrix_rank(Ctrb)==n else '❌ 비가제어'}")

Q = np.diag(Q_DIAG)
Rc = np.array([[R_COST]])
P = solve_discrete_are(Ad, Bd, Q, Rc)
K = np.linalg.inv(Rc + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)

eigs_cl = sorted(abs(e) for e in np.linalg.eigvals(Ad - Bd @ K))
print("\n" + "=" * 60)
print("  LQR 결과")
print("=" * 60)
print(f"Q = diag({Q_DIAG}),  R = {R_COST}")
print(f"K = [{', '.join(f'{k:.6f}' for k in K[0])}]")
print(f"    [K_phi, K_alpha, K_theta, K_phiDot, K_alphaDot, K_thetaDot]")
print(f"폐루프 max|eig| = {max(eigs_cl):.6f} → "
      f"{'✅ STABLE' if max(eigs_cl) < 1.0 else '❌ UNSTABLE'}")

# ============================================================
#  칼만 관측기 게인 L — ★단일 모델기반 칼만 (물리센서 직접 측정)
# ------------------------------------------------------------
#  옛 버전은 {α, θ, α̇, θ̇} 를 독립 측정으로 받았으나, 실물에서 α·α̇는
#  센서로 직접 안 나온다(α=θ_cf−δ, α̇=θ̇−δ̇로 합성 → θ 오차가 α로 새고,
#  CF와 칼만이 직렬 중첩). 여기서는 IMU·엔코더의 *물리 측정*을 그대로 쓴다:
#    z0 = θ_acc   (가속도계 중력기울기, 절대 θ)        → C row e3
#    z1 = θ̇_gyro  (자이로 Y축)                          → C row e6
#    z2 = δ        (엔코더 위치 = θ − α)                 → C row (e3−e2)
#    z3 = δ̇        (PRESENT_VELOCITY = θ̇ − α̇)           → C row (e6−e5)
#  α, φ 는 측정하지 않고 모델 커플링(Ad)+엔코더 구속으로 칼만이 복원.
#  → CF 제거, 단일 모델기반 칼만으로 통합 (입력=전류제어의 진짜 τ).
#  게이팅: 가속도 크기가 1g에서 벗어나면(모터 구동 선형가속) 펌웨어가
#         z0(θ_acc) innovation을 0으로 막는다. 정상상태 L은 z0 포함 가정.
# ============================================================
m_obs = 4
C_obs = np.zeros((m_obs, n))
C_obs[0, 2] = 1.0                      # z0: θ_acc      = θ
C_obs[1, 5] = 1.0                      # z1: θ̇_gyro     = θ̇
C_obs[2, 2] = 1.0; C_obs[2, 1] = -1.0  # z2: δ          = θ − α
C_obs[3, 5] = 1.0; C_obs[3, 4] = -1.0  # z3: δ̇          = θ̇ − α̇

O = np.vstack([C_obs @ np.linalg.matrix_power(Ad, k) for k in range(n)])
print(f"\n[단일칼만] 측정 = [θ_acc, θ̇_gyro, δ, δ̇]")
print(f"관측가능성 rank = {np.linalg.matrix_rank(O)}/{n} "
      f"→ {'✅' if np.linalg.matrix_rank(O)==n else '❌ 비관측'}")

# 프로세스 잡음 Q_kf: 모델이 좋다(전류제어 진짜 τ). φ는 모델로만 보이므로
#   φ·φ̇에 약간 더 신뢰. 측정 잡음 R_kf: 센서 특성(rad, rad/s) 기반.
Q_kf = np.diag([1e-4, 1e-6, 1e-6, 1e-3, 1e-5, 1e-5])
#                θ_acc   θ̇_gyro   δ(enc)   δ̇(vel)
R_kf = np.diag([1.2e-3,  1.0e-4,  1.0e-5,  1.0e-3])
P_kf = solve_discrete_are(Ad.T, C_obs.T, Q_kf, R_kf)
L = P_kf @ C_obs.T @ np.linalg.inv(C_obs @ P_kf @ C_obs.T + R_kf)
eigs_obs = sorted(abs(e) for e in np.linalg.eigvals(Ad - L @ C_obs))
print(f"관측기 max|eig| = {max(eigs_obs):.6f} → "
      f"{'✅ STABLE' if max(eigs_obs) < 1.0 else '❌ UNSTABLE'}")

# ============================================================
#  φ 운동학 추정 계수
# ============================================================
C1 = p1 / Mt
C2 = p2 / Mt

# ============================================================
#  토크 → Goal Current 변환 (전류제어)
# ============================================================
# I[A] = τ/Kt ,  raw_unit = I·1000/CUR_UNIT
tau2unit = 1000.0 / (KT * CUR_UNIT)   # unit per N·m
TAU_MAX_UNIT = int(TAU_MAX * tau2unit)
print("\n" + "=" * 60)
print("  전류제어 변환")
print("=" * 60)
print(f"τ→GoalCurrent: raw_unit = τ[N·m] × {tau2unit:.3f}")
print(f"  예) τ=1.0 → {tau2unit:.1f} unit ({tau2unit*CUR_UNIT/1000:.3f} A)")
print(f"  TAU_MAX={TAU_MAX} → {TAU_MAX_UNIT} unit "
      f"({TAU_MAX_UNIT*CUR_UNIT/1000:.2f} A; CurrentLimit 1193 이내 "
      f"{'OK' if TAU_MAX_UNIT<=1193 else '초과!'})")

# ============================================================
#  Arduino 출력
# ============================================================
print("\n" + "=" * 60)
print("  Arduino 하드코딩용 (전류제어 펌웨어)")
print("=" * 60)
print(f"// ===== 실측 LQR 게인 (v17 규약) =====")
print(f"const float K_lqr[6] = {{{', '.join(f'{k:.4f}f' for k in K[0])}}};")
print(f"const float TAU_MAX = {TAU_MAX}f;        // N·m")
print(f"const float KT       = {KT}f;     // N·m/A (토크보정 실측값으로 교체)")
print(f"const float CUR_UNIT = {CUR_UNIT}f;     // mA/unit")
print(f"const float TAU2UNIT = {tau2unit:.4f}f; // τ[N·m] → GoalCurrent[unit]")
print(f"// GoalCurrent_unit = (int)(tau * TAU2UNIT);")
print(f"\n// φ 운동학 추정 (IMU 직접관측 불가)")
print(f"const float C1_PHI = {C1:.6f}f;")
print(f"const float C2_PHI = {C2:.6f}f;")
print(f"const float R_ARC  = {R:.2f}f;")
print(f"// phi_est = -(C1_PHI*alpha + C2_PHI*theta) / R_ARC;")

print(f"const int   TAU_MAX_UNIT = {TAU_MAX_UNIT};   // = TAU_MAX*TAU2UNIT (CurrentLimit 1193 이내)")
print(f"// 점진검증 권장: CUR_SAFE 600(≈1.6A)부터 시작해 단계적으로 올릴 것")

print(f"\n// ===== 단일 칼만 게인 L (6×4) =====")
print(f"// 측정 순서 z = [theta_acc, thetaDot_gyro, delta(enc), deltaDot(vel)]")
print(f"//   C row0: theta      C row1: thetaDot")
print(f"//   C row2: delta=theta-alpha   C row3: deltaDot=thetaDot-alphaDot")
names = ['phi', 'alpha', 'theta', 'phiDot', 'alphaDot', 'thetaDot']
print(f"const float L_kf[6][4] = {{")
for i in range(n):
    vals = ', '.join(f'{L[i,j]:.6f}f' for j in range(m_obs))
    comma = ',' if i < n - 1 else ''
    print(f"  {{{vals}}}{comma}  // {names[i]}")
print(f"}};")

print(f"\n// ===== 이산 모델 Ad (6×6) =====")
print(f"const float Ad_mat[6][6] = {{")
for i in range(n):
    vals = ', '.join(f'{Ad[i,j]:.8f}f' for j in range(n))
    comma = ',' if i < n - 1 else ''
    print(f"  {{{vals}}}{comma}")
print(f"}};")
print(f"\nconst float Bd_vec[6] = {{{', '.join(f'{Bd[i,0]:.8f}f' for i in range(n))}}};")

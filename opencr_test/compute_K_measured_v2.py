"""
LQR K + 칼만 L 계산 — v2 (2026-07-02)
======================================================
compute_K_measured.py 대비 변경 (★lqr_balance_current_v2.ino 전용):

  1. ★가속도계 측정모델 확장 (게이팅 의존 축소):
       구버전:  z0 = θ_acc ≈ θ                (C row = e3)
       v2:      z0 = θ_acc ≈ θ − ẍ_imu/g      (IMU 선형가속 보상)
     이 로봇은 정상 동작 중에도 발이 원호 위에서 흔들려 IMU에 상시
     선형가속이 걸린다. ẍ_imu = [R, L1, ℓ_imu]·q̈, q̈ = M⁻¹(Gq + Dq̇ + Eτ)
     는 상태의 '선형'함수이므로 EKF 없이 C행 확장 + τ 직결항(D0)으로
     정확히 반영된다:
       C0 = e3 − (1/g)·w·M⁻¹[G | D],   D0 = −(1/g)·w·M⁻¹E,
       w = [R, L1, ℓ_imu]
     펌웨어 innovation:  innov0 = z_thAcc − (C0·xPred + D0·τ)
     → 흔들림·구동 중에도 가속도계 정보를 버리지 않고 사용.
       (1g 게이팅은 모델 밖 충격(베어링 스틱션 등) 대비 안전장치로 유지)

  2. ℓ_imu(힙→IMU 거리) 를 MEASURED 파라미터로 추가. ★실측할 것.

  3. 자이로 단위 버그 문서화: OpenCR 라이브러리의 imu.gyroData[]는
     deg/s가 아니라 raw LSB (gRes=2000/32768≈0.061 dps/LSB).
     펌웨어 v2는 라이브러리가 dps로 변환해 둔 imu.gy 를 사용한다.

  나머지(플랜트, Q/R, K, 이산화)는 v1과 동일 → K·Ad·Bd 값 불변.

좌표계 (v17 규약):
  φ, α, θ 연직기준 시계+, δ=θ−α(앞접기+), τ→[0,−1,+1], τ=−K·x
"""
import numpy as np
from scipy.linalg import solve_discrete_are, expm

# ============================================================
#  복합진자 측정 → I_cm 보조 함수
# ============================================================
def I_cm_from_pendulum(m, d, T, g=9.81):
    """m[kg], d=피벗→CoM[m], T=미소진동주기[s] → (I_cm, I_pivot)"""
    I_pivot = m * g * d * (T / (2 * np.pi)) ** 2
    I_cm = I_pivot - m * d ** 2
    return I_cm, I_pivot


# ============================================================
#  ★★★ MEASURED 블록 — 실측값 입력 (아래는 v11_small 예시값) ★★★
# ============================================================
USE_PENDULUM = False

m1 = 0.10       # 하체 [kg]
m2 = 0.30       # 상체 [kg] (모터·보드·배터리 포함)
L1 = 0.20       # 발 → 힙 [m]
L2 = 0.25       # 힙 → 상체끝 [m]
ell1 = 0.10     # 발 → 하체 CoM [m]
ell2 = 0.125    # 힙 → 상체 CoM [m]

# ★신규: 힙 → IMU(OpenCR 보드 ICM-20648) 거리 [m]. 자로 직접 잴 것.
ELL_IMU = 0.13

if USE_PENDULUM:
    I1, _ = I_cm_from_pendulum(m=m1, d=0.10, T=0.0)
    I2, _ = I_cm_from_pendulum(m=m2, d=0.125, T=0.0)
else:
    I1 = m1 * L1**2 / 12
    I2 = m2 * L2**2 / 12

R  = 0.30       # 원호 반경 = 줄 처짐 [m] (실물: 베어링축~발 거리)
g  = 9.81
B_THETA = 0.01  # 상체 점성 감쇠 [N·m·s/rad]

TAU_MAX  = 3.0      # [N·m]
KT       = 1.304    # [N·m/A]
CUR_UNIT = 2.69     # [mA/unit]
DT = 0.002          # 500Hz

Q_DIAG = [1, 50, 50, 0.1, 5, 5]
R_COST = 0.001

# ============================================================
#  파생 상수 / 플랜트 (v1과 동일)
# ============================================================
Mt = m1 + m2
p1 = m1 * ell1 + m2 * L1
p2 = m2 * ell2
p3 = m2 * L1 * ell2

print("=" * 60)
print("  실측 LQR 계산 v2 (가속도계 측정모델 확장)")
print("=" * 60)
print(f"m1={m1}, m2={m2}, L1={L1}, L2={L2}, ell1={ell1}, ell2={ell2}")
print(f"ELL_IMU={ELL_IMU} m (힙→IMU ★실측), R={R}, DT={DT*1000:.0f}ms")

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
E_mat = np.array([[0], [-1], [1]])

eig_M = np.linalg.eigvalsh(M_eq)
assert np.all(eig_M > 0), "M_eq 비양정치 — 파라미터 점검!"

Mi = np.linalg.inv(M_eq)
n = 6
Ac = np.zeros((n, n)); Bc = np.zeros((n, 1))
Ac[:3, 3:] = np.eye(3)
Ac[3:, :3] = Mi @ G_mat
Ac[3:, 3:] = Mi @ D_mat
Bc[3:, :]  = Mi @ E_mat

# 이산화 (ZOH)
Z = np.zeros((n + 1, n + 1))
Z[:n, :n] = Ac * DT
Z[:n, n:] = Bc * DT
eZ = expm(Z)
Ad = eZ[:n, :n]; Bd = eZ[:n, n:n + 1]

# LQR (v1과 동일 → K 불변)
Ctrb = np.hstack([np.linalg.matrix_power(Ad, k) @ Bd for k in range(n)])
assert np.linalg.matrix_rank(Ctrb) == n, "비가제어!"
P = solve_discrete_are(Ad, Bd, np.diag(Q_DIAG), np.array([[R_COST]]))
K = np.linalg.inv(np.array([[R_COST]]) + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
eigs_cl = max(abs(e) for e in np.linalg.eigvals(Ad - Bd @ K))
print(f"K = [{', '.join(f'{k:.4f}' for k in K[0])}]  폐루프 max|eig|={eigs_cl:.6f} "
      f"{'✅' if eigs_cl < 1 else '❌'}")

# ============================================================
#  ★ 칼만 관측기 v2 — 가속도계 선형가속 보상 측정모델
#    z = [θ_acc, θ̇_gyro, δ, δ̇]
#    z0 = θ − ẍ_imu/g  (ẍ_imu = w·q̈,  w=[R, L1, ℓ_imu])
# ============================================================
w_imu = np.array([R, L1, ELL_IMU])          # IMU 위치의 x ≈ Rφ + L1α + ℓ_imu θ
wMiG = w_imu @ (Mi @ G_mat)                 # q 계수
wMiD = w_imu @ (Mi @ D_mat)                 # q̇ 계수
wMiE = float(w_imu @ (Mi @ E_mat).ravel())  # τ 계수

m_obs = 4
C_obs = np.zeros((m_obs, n))
C_obs[0, 2] = 1.0                           # θ
C_obs[0, 0:3] -= wMiG / g                   # − ẍ_imu(q)/g
C_obs[0, 3:6] -= wMiD / g                   # − ẍ_imu(q̇)/g
D0_feed = -wMiE / g                         # innovation에서 D0·τ 로 사용
C_obs[1, 5] = 1.0                           # θ̇_gyro
C_obs[2, 2] = 1.0; C_obs[2, 1] = -1.0       # δ = θ − α
C_obs[3, 5] = 1.0; C_obs[3, 4] = -1.0       # δ̇ = θ̇ − α̇

O = np.vstack([C_obs @ np.linalg.matrix_power(Ad, k) for k in range(n)])
rank_O = np.linalg.matrix_rank(O)
print(f"\n[칼만 v2] z=[θ_acc(선형가속 보상), θ̇_gyro, δ, δ̇]")
print(f"C0 (θ_acc 행) = {np.array2string(C_obs[0], precision=6)}")
print(f"D0 (τ 직결)   = {D0_feed:.6f}")
print(f"관측가능성 rank = {rank_O}/{n} {'✅' if rank_O == n else '❌'}")

Q_kf = np.diag([1e-4, 1e-6, 1e-6, 1e-3, 1e-5, 1e-5])
R_kf = np.diag([1.2e-3, 1.0e-4, 1.0e-5, 1.0e-3])
P_kf = solve_discrete_are(Ad.T, C_obs.T, Q_kf, R_kf)
L = P_kf @ C_obs.T @ np.linalg.inv(C_obs @ P_kf @ C_obs.T + R_kf)
eigs_obs = max(abs(e) for e in np.linalg.eigvals(Ad - L @ C_obs))
print(f"관측기 max|eig| = {eigs_obs:.6f} {'✅ STABLE' if eigs_obs < 1 else '❌'}")

# φ 보조추정 / 전류 변환
C1 = p1 / Mt; C2 = p2 / Mt
tau2unit = 1000.0 / (KT * CUR_UNIT)
TAU_MAX_UNIT = int(TAU_MAX * tau2unit)

# ============================================================
#  Arduino 출력 (lqr_balance_current_v2.ino 용)
# ============================================================
print("\n" + "=" * 60)
print("  Arduino 하드코딩용 (lqr_balance_current_v2)")
print("=" * 60)
print(f"const float K_lqr[6] = {{{', '.join(f'{k:.4f}f' for k in K[0])}}};")
print(f"const float TAU2UNIT = {tau2unit:.4f}f;  // TAU_MAX={TAU_MAX} → {TAU_MAX_UNIT} unit")
print(f"const float C1_PHI = {C1:.6f}f;")
print(f"const float C2_PHI = {C2:.6f}f;")
print(f"const float R_ARC  = {R:.2f}f;")
print(f"\n// θ_acc 측정행 C0 (6) — innov0 = z_thAcc − (C0·xPred + D0_FEED·τ)")
print(f"const float C0_acc[6] = {{{', '.join(f'{c:.6f}f' for c in C_obs[0])}}};")
print(f"const float D0_FEED = {D0_feed:.6f}f;")
names = ['phi', 'alpha', 'theta', 'phiDot', 'alphaDot', 'thetaDot']
print(f"\nconst float L_kf[6][4] = {{")
for i in range(n):
    vals = ', '.join(f'{L[i, j]:.6f}f' for j in range(m_obs))
    print(f"  {{{vals}}}{',' if i < n-1 else ''}  // {names[i]}")
print(f"}};")
print(f"\nconst float Ad_mat[6][6] = {{")
for i in range(n):
    vals = ', '.join(f'{Ad[i, j]:.8f}f' for j in range(n))
    print(f"  {{{vals}}}{',' if i < n-1 else ''}")
print(f"}};")
print(f"\nconst float Bd_vec[6] = {{{', '.join(f'{Bd[i,0]:.8f}f' for i in range(n))}}};")

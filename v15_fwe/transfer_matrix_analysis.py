"""확장 모델 전달 행렬 계산 및 수렴비 분석
유한 시간 접기-기다리기-펴기 이론 검증
"""
import numpy as np

# === 시스템 파라미터 (사람 크기 모델) ===
g = 9.81
R = 1.0        # 줄 곡률 반경 [m]
h = 0.85       # CoM 높이 [m]
M_total = 75   # 총 질량 [kg]
I_inv = 20     # 역진자 관성모멘트 [kg·m²]
c_foot = 0.280 # 발 변위 계수

omega = np.sqrt(g / (R + h))
lam = np.sqrt(M_total * g * h / I_inv)

print(f"omega = {omega:.4f} rad/s")
print(f"lambda = {lam:.4f} rad/s")
print(f"T_quarter = {np.pi/(2*omega):.4f} s")

# === 제어 파라미터 ===
tau_fold = 100    # 접기 토크 [Nm]
I_delta = 5.0     # 접기 관성모멘트 [kg·m²]
a_f = tau_fold / I_delta

f_phi = (c_foot / R) * a_f
f_beta = (c_foot / h) * a_f

print(f"\na_f = {a_f:.1f} rad/s^2")
print(f"f_phi = {f_phi:.4f} rad/s^2")
print(f"f_beta = {f_beta:.4f} rad/s^2")

# === 행렬 생성 함수 ===
def make_A(om, la, T):
    """전체 4x4 전달 행렬 (블록 대각)"""
    A = np.zeros((4, 4))
    c, s = np.cos(om*T), np.sin(om*T)
    A[0,0], A[0,1] = c, s/om
    A[1,0], A[1,1] = -om*s, c
    ch, sh = np.cosh(la*T), np.sinh(la*T)
    A[2,2], A[2,3] = ch, sh/la
    A[3,2], A[3,3] = la*sh, ch
    return A

def make_b_fold(om, la, T_f, fp, fb):
    """FOLD 변위 벡터"""
    C = np.cos(om * T_f)
    S = np.sin(om * T_f)
    Ch = np.cosh(la * T_f)
    Sh = np.sinh(la * T_f)
    return np.array([
        fp/om**2 * (1 - C),
        fp/om * S,
        -fb/la**2 * (Ch - 1),
        -fb/la * Sh
    ])

def one_cycle(T_f, T_w):
    """1사이클 전달 행렬 M과 변위 벡터 d"""
    A_F = make_A(omega, lam, T_f)
    A_W = make_A(omega, lam, T_w)
    A_E = A_F  # T_e = T_f (대칭)
    b_F = make_b_fold(omega, lam, T_f, f_phi, f_beta)
    b_E = -b_F
    M = A_E @ A_W @ A_F
    d = A_E @ A_W @ b_F + b_E
    return M, d

# === 수렴비 스윕 ===
beta0 = np.radians(5)
T_quarter = np.pi / (2 * omega)

print(f"\nbeta0 = {np.degrees(beta0):.1f} deg")
print(f"T_quarter = {T_quarter:.4f} s")

# --- 스윕 1: T_f x T_w 탐색 ---
print("\n" + "="*60)
print("Sweep 1: T_f x T_w (beta0 = 5 deg, tau = 100 Nm)")
print("="*60)

header = f"{'T_f':>6} {'T_w':>6} {'T_tot':>6} {'beta_f(deg)':>11} {'phi_f(deg)':>11} {'r':>7}"
print(header)
print("-" * len(header))

best_r = 999
best_params = None

T_f_vals = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
T_w_ratios = np.arange(0.05, 0.55, 0.05)

for T_f in T_f_vals:
    for T_w_ratio in T_w_ratios:
        T_w = T_w_ratio * T_quarter
        M, d = one_cycle(T_f, T_w)
        v0 = np.array([0, 0, beta0, 0])
        v_final = M @ v0 + d
        beta_final = v_final[2]
        phi_final = v_final[0]
        r = abs(beta_final / beta0)
        if r < best_r:
            best_r = r
            best_params = (T_f, T_w, 2*T_f+T_w, beta_final, phi_final)

print(f"\n*** BEST: T_f={best_params[0]:.3f}s, T_w={best_params[1]:.3f}s, "
      f"T_total={best_params[2]:.3f}s")
print(f"    beta_final={np.degrees(best_params[3]):.4f} deg, "
      f"phi_final={np.degrees(best_params[4]):.4f} deg, r={best_r:.4f}")

# --- 상세 출력: T_f별 최적 T_w ---
print("\n" + "="*60)
print("T_f별 최적 T_w 탐색 (세밀)")
print("="*60)

for T_f in T_f_vals:
    best_r_local = 999
    best_T_w = 0
    best_beta = 0
    best_phi = 0
    for T_w in np.linspace(0.01, 0.8*T_quarter, 200):
        M, d = one_cycle(T_f, T_w)
        v0 = np.array([0, 0, beta0, 0])
        v_final = M @ v0 + d
        r = abs(v_final[2] / beta0)
        if r < best_r_local:
            best_r_local = r
            best_T_w = T_w
            best_beta = v_final[2]
            best_phi = v_final[0]
    
    T_total = 2*T_f + best_T_w
    print(f"T_f={T_f:.3f}  best_T_w={best_T_w:.4f}  T_total={T_total:.4f}  "
          f"r={best_r_local:.4f}  beta_f={np.degrees(best_beta):.3f}deg  "
          f"phi_f={np.degrees(best_phi):.3f}deg")

# --- Ψ 함수 검증 ---
print("\n" + "="*60)
print("Psi 함수 검증 (해석적 vs 수치적)")
print("="*60)

T_f_test = 0.10
T_w_test = 0.15 * T_quarter
T_total_test = 2*T_f_test + T_w_test
T_fw = T_f_test + T_w_test

Psi = (np.cosh(lam*T_total_test) - np.cosh(lam*T_fw) 
       - np.cosh(lam*T_f_test) + 1)
beta_analytical = np.cosh(lam*T_total_test)*beta0 - (f_beta/lam**2)*Psi

M, d = one_cycle(T_f_test, T_w_test)
v0 = np.array([0, 0, beta0, 0])
v_final = M @ v0 + d
beta_numerical = v_final[2]

print(f"T_f={T_f_test}, T_w={T_w_test:.4f}")
print(f"Psi = {Psi:.6f}")
print(f"beta_analytical = {np.degrees(beta_analytical):.6f} deg")
print(f"beta_numerical  = {np.degrees(beta_numerical):.6f} deg")
print(f"Match: {np.isclose(beta_analytical, beta_numerical)}")

# --- 순간 모델 극한 검증 ---
print("\n" + "="*60)
print("순간 모델 극한 검증 (T_f -> 0)")
print("="*60)

T_w_fixed = 0.17 * T_quarter
for T_f in [0.001, 0.0001, 0.00001]:
    # 유한 모델
    tau_scaled = tau_fold * (0.10 / T_f)**2 * (I_delta)  # a_f 조정해서 delta_fold 유지
    a_f_scaled = tau_scaled / I_delta
    delta_fold = 0.5 * a_f * 0.10**2  # T_f=0.10에서의 delta_fold 고정
    a_f_needed = 2 * delta_fold / T_f**2
    
    fp_s = (c_foot / R) * a_f_needed
    fb_s = (c_foot / h) * a_f_needed
    
    A_F = make_A(omega, lam, T_f)
    A_W = make_A(omega, lam, T_w_fixed)
    b_F = make_b_fold(omega, lam, T_f, fp_s, fb_s)
    b_E = -b_F
    
    M = A_F @ A_W @ A_F
    d = A_F @ A_W @ b_F + b_E
    
    v0 = np.array([0, 0, beta0, 0])
    v_final = M @ v0 + d
    r = abs(v_final[2] / beta0)
    print(f"T_f={T_f:.5f}  r={r:.6f}  beta_f={np.degrees(v_final[2]):.4f}deg")

print("\n완료!")

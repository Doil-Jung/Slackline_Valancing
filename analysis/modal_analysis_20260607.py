"""
모드 분석: 유효 phi / 유효 beta 고유모드와 LQR 게인의 관계 검증

좌표: (phi, beta, delta)
  phi  = 줄 흔들림 각도
  beta = (p1*alpha + p2*theta) / (Mt*h_com)  -- CoM 기울기 각도
  delta = theta - alpha                       -- 접힘 각도

2026-06-07 대화에서 작성
"""
import numpy as np
from scipy.linalg import solve_discrete_are

g = 9.81

# ============================================
# 시스템 파라미터
# ============================================
M1 = 30.0;  m2 = 45.0;  L1 = 0.85
LC1 = 0.30; LC2 = 0.40; L_upper = 0.85

Mt = M1 + m2
I1 = M1 * L1**2 / 12
I2 = m2 * L_upper**2 / 12

p1 = M1 * LC1 + m2 * L1
p2 = m2 * LC2
p3 = m2 * L1 * LC2
ps = p1 + p2
h_com = (M1 * LC1 + m2 * (L1 + LC2)) / Mt
c_beta = Mt * h_com / ps  # beta 스케일 인자

J_aa = M1 * LC1**2 + I1 + m2 * L1**2
J_tt = m2 * LC2**2 + I2
J_at = p3
B_theta = 0.5

print("=" * 75)
print("  slackline modal analysis: effective phi/beta vs LQR")
print("=" * 75)
print(f"  M1={M1}, m2={m2}, Mt={Mt} kg")
print(f"  p1={p1:.3f}, p2={p2:.3f}, ps={ps:.3f}")
print(f"  h_com={h_com:.4f} m, c_beta=Mt*h/ps={c_beta:.4f}")


def analyze(R):
    # --- (phi, alpha, theta) 좌표 ---
    M = np.array([
        [Mt*R**2,  R*p1,   R*p2],
        [R*p1,     J_aa,   J_at],
        [R*p2,     J_at,   J_tt]])
    G = np.diag([-Mt*g*R, p1*g, p2*g])
    D = np.diag([0.0, 0.0, -B_theta])
    E_vec = np.array([[0.0], [-1.0], [1.0]])
    M_inv = np.linalg.inv(M)

    # --- (phi,alpha,theta) -> (phi, beta, delta) 변환 ---
    #   alpha = c_beta * beta - (p2/ps) * delta
    #   theta = c_beta * beta + (p1/ps) * delta
    T_inv_mat = np.array([
        [1.0, 0.0,         0.0],
        [0.0, c_beta, -p2/ps],
        [0.0, c_beta,  p1/ps]])
    T_mat = np.linalg.inv(T_inv_mat)

    M_pbd = T_inv_mat.T @ M @ T_inv_mat
    G_pbd = T_inv_mat.T @ G @ T_inv_mat

    # (phi, beta) 2x2 부분계: q_dd = M_inv * G * q
    A_pos = np.linalg.solve(M_pbd, G_pbd)
    A2 = A_pos[:2, :2]

    evals, evecs = np.linalg.eig(A2)
    idx = np.argsort(evals.real)
    evals = evals[idx].real
    evecs = evecs[:, idx].real

    omega_eff = np.sqrt(abs(evals[0]))
    lambda_eff = np.sqrt(abs(evals[1]))
    omega_diag = np.sqrt(abs(A2[0, 0]))
    lambda_diag = np.sqrt(abs(A2[1, 1]))

    # 고유벡터 정규화
    v_pend = evecs[:, 0] / evecs[0, 0]   # phi=1
    v_inv  = evecs[:, 1] / evecs[1, 1]   # beta=1

    # --- LQR in (phi, alpha, theta) ---
    Ac = np.zeros((6, 6))
    Ac[:3, 3:] = np.eye(3)
    Ac[3:, :3] = M_inv @ G
    Ac[3:, 3:] = M_inv @ D
    Bc = np.zeros((6, 1))
    Bc[3:, :] = M_inv @ E_vec

    dt = 0.001
    A2c = Ac@Ac; A3c = A2c@Ac; A4c = A3c@Ac
    Ad = np.eye(6) + Ac*dt + A2c*dt**2/2 + A3c*dt**3/6 + A4c*dt**4/24
    Bd = (np.eye(6)*dt + Ac*dt**2/2 + A2c*dt**3/6 + A3c*dt**4/24) @ Bc

    Q_lqr = np.diag([1.0, 50.0, 50.0, 0.1, 5.0, 5.0])
    R_cost = np.array([[0.001]])
    P = solve_discrete_are(Ad, Bd, Q_lqr, R_cost)
    K = (np.linalg.inv(R_cost + Bd.T@P@Bd) @ (Bd.T@P@Ad)).flatten()
    K[0] *= -1; K[3] *= -1

    # --- K -> (phi, beta, delta) ---
    T6 = np.zeros((6, 6))
    T6[:3,:3] = T_inv_mat; T6[3:,3:] = T_inv_mat
    K_pbd = K @ T6

    # --- K -> (Phi_eff, Beta_eff, delta) 모드 좌표 ---
    V_m = evecs
    T6m = np.eye(6)
    T6m[:2,:2] = V_m; T6m[3:5,3:5] = V_m
    K_mode = K_pbd @ T6m

    # --- 토크 입력 in (phi,beta,delta) ---
    E_pbd = T_mat @ E_vec.flatten()

    return {
        'R': R, 'evals': evals,
        'v_pend': v_pend, 'v_inv': v_inv,
        'w_eff': omega_eff, 'l_eff': lambda_eff,
        'w_diag': omega_diag, 'l_diag': lambda_diag,
        'K': K, 'K_pbd': K_pbd, 'K_mode': K_mode,
        'A2': A2, 'E_pbd': E_pbd,
    }


R_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
results = [analyze(R) for R in R_values]

# ========================================================
print(f"\n{'='*75}")
print("  1. (phi, beta) 2x2 부분계의 고유모드")
print(f"{'='*75}")
print(f"  w_diag = 커플링 무시한 분리 주파수")
print(f"  w_eff  = 커플링 포함 유효 주파수 (고유값)\n")
print(f"{'R':>5} | {'w_diag':>7} {'w_eff':>7} {'ratio':>5} | {'l_diag':>7} {'l_eff':>7} {'ratio':>5} | Phi_eff           | Beta_eff")
print("-" * 100)
for r in results:
    wr = r['w_eff']/r['w_diag']
    lr = r['l_eff']/r['l_diag']
    print(f"{r['R']:5.1f} | {r['w_diag']:7.3f} {r['w_eff']:7.3f} {wr:5.2f}x | "
          f"{r['l_diag']:7.3f} {r['l_eff']:7.3f} {lr:5.2f}x | "
          f"phi + ({r['v_pend'][1]:+.4f})*beta | ({r['v_inv'][0]:+.4f})*phi + beta")

# ========================================================
print(f"\n{'='*75}")
print("  2. LQR gain in 3 coordinate systems")
print(f"{'='*75}")
for r in results:
    K, Kp, Km = r['K'], r['K_pbd'], r['K_mode']
    print(f"\n  R = {r['R']:.1f} m")
    print(f"    (phi, alpha, theta):      pos=[{K[0]:8.1f} {K[1]:8.1f} {K[2]:8.1f}]  vel=[{K[3]:8.1f} {K[4]:8.1f} {K[5]:8.1f}]")
    print(f"    (phi, beta,  delta):      pos=[{Kp[0]:8.1f} {Kp[1]:8.1f} {Kp[2]:8.1f}]  vel=[{Kp[3]:8.1f} {Kp[4]:8.1f} {Kp[5]:8.1f}]")
    print(f"    (Phi_eff, Beta_eff, d):   pos=[{Km[0]:8.1f} {Km[1]:8.1f} {Km[2]:8.1f}]  vel=[{Km[3]:8.1f} {Km[4]:8.1f} {Km[5]:8.1f}]")

# ========================================================
print(f"\n{'='*75}")
print("  3. Modal gain magnitude comparison")
print(f"{'='*75}")
print(f"{'R':>5} | {'|K_Phi|':>9} {'|K_Beta|':>9} {'|K_d|':>9} | {'|K_dPhi|':>9} {'|K_dBeta|':>9} {'|K_dd|':>9} | Beta/Phi")
print("-" * 90)
for r in results:
    km = np.abs(r['K_mode'])
    ratio = km[1]/km[0] if km[0] > 1e-10 else float('inf')
    print(f"{r['R']:5.1f} | {km[0]:9.1f} {km[1]:9.1f} {km[2]:9.1f} | "
          f"{km[3]:9.1f} {km[4]:9.1f} {km[5]:9.1f} | {ratio:5.2f}x")

# ========================================================
print(f"\n{'='*75}")
print("  4. K_delta R-independence")
print(f"{'='*75}")
print(f"  {'R':>5} | {'K_delta(pos)':>12} | {'K_delta_dot(vel)':>16}")
print(f"  {'-'*40}")
kd_vals = []
for r in results:
    kd_vals.append(r['K_mode'][2])
    print(f"  {r['R']:5.1f} | {r['K_mode'][2]:12.4f} | {r['K_mode'][5]:16.4f}")
print(f"\n  K_delta variation: {np.std(kd_vals)/abs(np.mean(kd_vals))*100:.3f}%")

# ========================================================
print(f"\n{'='*75}")
print("  5. Torque input E in (phi, beta, delta)")
print(f"{'='*75}")
r0 = results[0]
print(f"  E = [{r0['E_pbd'][0]:+.4f}, {r0['E_pbd'][1]:+.4f}, {r0['E_pbd'][2]:+.4f}]  (same for all R)")
print(f"  -> phi:   0       (under-actuated)")
print(f"  -> beta:  {r0['E_pbd'][1]:+.4f}")
print(f"  -> delta: {r0['E_pbd'][2]:+.4f}  (strongest)")

# ========================================================
print(f"\n{'='*75}")
print("  6. Coupling strength")
print(f"{'='*75}")
for r in results:
    mu_phi = r['A2'][0,1] / r['A2'][0,0]
    mu_beta = r['A2'][1,0] / r['A2'][1,1]
    print(f"  R={r['R']:.1f}: "
          f"phi_dd beta dependency={abs(mu_phi)*100:.1f}%  |  "
          f"beta_dd phi dependency={abs(mu_beta)*100:.1f}%")

print(f"\n  Done!")

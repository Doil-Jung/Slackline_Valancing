# -*- coding: utf-8 -*-
"""4단계 FWE 전달행렬 분석 (진단 포함)
핵심 발견: lambda >> omega 이면 beta가 한 사이클 내에 폭발할 수 있음
"""
import numpy as np
from scipy.optimize import brentq

# ============================================================
# 시스템 파라미터
# ============================================================
g = 9.81
R = 1.0
h = 0.85
M_total = 75.0
I_inv = 20.0
c_foot = 0.280

omega = np.sqrt(g / (R + h))
lam = np.sqrt(M_total * g * h / I_inv)
T_quarter = np.pi / (2 * omega)

# 제어 파라미터
tau_fold = 100.0
I_delta = 5.0
a_f = tau_fold / I_delta
f_phi = (c_foot / R) * a_f
f_beta = (c_foot / h) * a_f

beta0 = np.radians(5.0)

# ============================================================
# 기본 함수
# ============================================================
def make_A(om, la, T):
    A = np.zeros((4, 4))
    c, s = np.cos(om*T), np.sin(om*T)
    A[0,0], A[0,1] = c, s/om
    A[1,0], A[1,1] = -om*s, c
    ch, sh = np.cosh(la*T), np.sinh(la*T)
    A[2,2], A[2,3] = ch, sh/la
    A[3,2], A[3,3] = la*sh, ch
    return A

def make_b_fold(om, la, T_f, fp, fb):
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

def one_cycle_4phase(T_f, T_w1, T_e, T_w2, fp, fb):
    A_F  = make_A(omega, lam, T_f)
    A_W1 = make_A(omega, lam, T_w1)
    A_E  = make_A(omega, lam, T_e)
    A_W2 = make_A(omega, lam, T_w2)
    b_F = make_b_fold(omega, lam, T_f, fp, fb)
    b_E = make_b_fold(omega, lam, T_e, -fp, -fb)
    M = A_W2 @ A_E @ A_W1 @ A_F
    d = A_W2 @ A_E @ A_W1 @ b_F + A_W2 @ b_E
    return M, d

def find_T_w2_for_phi_zero(T_f, T_w1, T_e, fp, fb, v_start,
                            T_w2_min=0.001, T_w2_max=2.0):
    def objective(T_w2):
        M, d = one_cycle_4phase(T_f, T_w1, T_e, T_w2, fp, fb)
        v_end = M @ v_start + d
        return v_end[0]
    
    N_search = 500
    T_w2_vals = np.linspace(T_w2_min, T_w2_max, N_search)
    f_vals = [objective(tw2) for tw2 in T_w2_vals]
    
    roots = []
    for i in range(len(f_vals)-1):
        if f_vals[i] * f_vals[i+1] < 0:
            root = brentq(objective, T_w2_vals[i], T_w2_vals[i+1])
            roots.append(root)
    return roots

# ============================================================
# 진단: 왜 beta가 폭발하는가?
# ============================================================
print("=" * 70)
print("[진단] 시스템 시간 스케일 분석")
print("=" * 70)
print(f"  omega  = {omega:.4f} rad/s  (진자)")
print(f"  lambda = {lam:.4f} rad/s  (역진자)")
print(f"  lambda/omega = {lam/omega:.2f}")
print(f"  T_quarter (진자) = {T_quarter:.4f} s")
print(f"  1/lambda (역진자 시간스케일) = {1/lam:.4f} s")
print()

# 사이클 동안 beta가 얼마나 성장하는가?
T_cycle_typical = 1.0  # 약 1초
growth = np.cosh(lam * T_cycle_typical)
print(f"  T_cycle ~ {T_cycle_typical:.1f}s 동안:")
print(f"  cosh(lambda * T_cycle) = {growth:.1f}")
print(f"  --> beta가 약 {growth:.0f}배 성장!")
print()
print(f"  결론: lambda/omega = {lam/omega:.1f} >> 1 이므로")
print(f"  발이 돌아오기 전에 beta가 이미 폭발한다.")
print(f"  이 메커니즘이 작동하려면 lambda * T_cycle < ~1 이어야 한다.")

# 필요한 조건 계산
T_cycle_max = 1.0 / lam  # lambda * T_cycle = 1
print(f"\n  lambda * T_cycle = 1 이 되려면:")
print(f"  T_cycle < {T_cycle_max:.4f} s")
print(f"  그런데 T_quarter = {T_quarter:.4f} s 이므로")
print(f"  T_cycle > T_quarter 일 수밖에 없고 (4단계이므로)")
print(f"  lambda * T_quarter = {lam * T_quarter:.2f}")

# ============================================================
# 그러면 어떤 파라미터에서 작동할까?
# ============================================================
print(f"\n{'='*70}")
print("[탐색] lambda * T_quarter < 1 이 되는 파라미터 조건")
print(f"{'='*70}")
print()
print("조건: lambda * pi/(2*omega) < 1")
print("즉:  sqrt(M*g*h/I) * pi/(2*sqrt(g/(R+h))) < 1")
print()

# R을 바꿔가며 탐색
print(f"{'R (m)':>8} {'omega':>8} {'lambda':>8} {'lam/om':>8} "
      f"{'T_q (s)':>8} {'lam*T_q':>8} {'가능?':>6}")
print("-" * 60)

for R_test in [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
    om_t = np.sqrt(g / (R_test + h))
    la_t = lam  # lambda는 R에 의존하지 않음!
    Tq_t = np.pi / (2 * om_t)
    ratio = la_t * Tq_t
    ok = "OK" if ratio < 1.5 else "X"
    print(f"{R_test:8.1f} {om_t:8.4f} {la_t:8.4f} {la_t/om_t:8.2f} "
          f"{Tq_t:8.4f} {ratio:8.2f} {ok:>6}")

print()
print("lambda는 R에 의존하지 않는다! (M, g, h, I_inv 에만 의존)")
print("따라서 R을 바꿔도 lambda는 같고, T_quarter만 길어진다.")
print("R이 커지면 T_quarter가 길어져서 오히려 더 불리하다.")

# h를 바꿔가며 탐색
print(f"\n{'h (m)':>8} {'omega':>8} {'lambda':>8} {'lam/om':>8} "
      f"{'T_q (s)':>8} {'lam*T_q':>8} {'가능?':>6}")
print("-" * 60)

for h_test in [0.05, 0.10, 0.20, 0.30, 0.50, 0.85]:
    om_t = np.sqrt(g / (R + h_test))
    la_t = np.sqrt(M_total * g * h_test / I_inv)
    Tq_t = np.pi / (2 * om_t)
    ratio = la_t * Tq_t
    ok = "OK" if ratio < 1.5 else "X"
    print(f"{h_test:8.2f} {om_t:8.4f} {la_t:8.4f} {la_t/om_t:8.2f} "
          f"{Tq_t:8.4f} {ratio:8.2f} {ok:>6}")

# I_inv를 바꿔가며 탐색
print(f"\n{'I_inv':>8} {'omega':>8} {'lambda':>8} {'lam/om':>8} "
      f"{'T_q (s)':>8} {'lam*T_q':>8} {'가능?':>6}")
print("-" * 60)

for I_test in [5, 10, 20, 50, 100, 200, 500, 1000]:
    om_t = omega  # omega는 I_inv에 무관
    la_t = np.sqrt(M_total * g * h / I_test)
    Tq_t = T_quarter
    ratio = la_t * Tq_t
    ok = "OK" if ratio < 1.5 else "X"
    print(f"{I_test:8.0f} {om_t:8.4f} {la_t:8.4f} {la_t/om_t:8.2f} "
          f"{Tq_t:8.4f} {ratio:8.2f} {ok:>6}")

# ============================================================
# 작동 가능한 파라미터로 4단계 검증
# ============================================================
print(f"\n{'='*70}")
print("[검증] 작동 가능한 파라미터로 4단계 beta 수렴 확인")
print(f"{'='*70}")

# I_inv를 크게 (= lambda를 작게) 해서 시도
I_inv_new = 500.0
lam_new = np.sqrt(M_total * g * h / I_inv_new)
omega_new = omega  # 동일

print(f"\n  변경: I_inv = {I_inv} -> {I_inv_new}")
print(f"  omega = {omega_new:.4f} (변화 없음)")
print(f"  lambda = {lam:.4f} -> {lam_new:.4f}")
print(f"  lam*T_quarter = {lam_new * T_quarter:.3f}")

# 새 파라미터로 함수 재정의
def make_A_new(T):
    return make_A(omega_new, lam_new, T)

def make_b_fold_new(T_f, fp, fb):
    return make_b_fold(omega_new, lam_new, T_f, fp, fb)

def one_cycle_4phase_new(T_f, T_w1, T_e, T_w2, fp, fb):
    A_F  = make_A_new(T_f)
    A_W1 = make_A_new(T_w1)
    A_E  = make_A_new(T_e)
    A_W2 = make_A_new(T_w2)
    b_F = make_b_fold_new(T_f, fp, fb)
    b_E = make_b_fold_new(T_e, -fp, -fb)
    M = A_W2 @ A_E @ A_W1 @ A_F
    d = A_W2 @ A_E @ A_W1 @ b_F + A_W2 @ b_E
    return M, d

def find_T_w2_new(T_f, T_w1, T_e, fp, fb, v_start):
    def obj(T_w2):
        M, d = one_cycle_4phase_new(T_f, T_w1, T_e, T_w2, fp, fb)
        return (M @ v_start + d)[0]
    
    T_w2_vals = np.linspace(0.001, 2.0, 500)
    f_vals = [obj(tw2) for tw2 in T_w2_vals]
    roots = []
    for i in range(len(f_vals)-1):
        if f_vals[i] * f_vals[i+1] < 0:
            roots.append(brentq(obj, T_w2_vals[i], T_w2_vals[i+1]))
    return roots

# 스윕
T_f_test = 0.10
T_e_test = 0.10
v0 = np.array([0.0, 0.0, beta0, 0.0])

print(f"\nT_f = T_e = {T_f_test:.3f} s")
print(f"\n{'T_w1':>7} {'T_w2':>7} {'T_cyc':>7} {'beta_end':>10} "
      f"{'r':>8} {'rho(M_b)':>10}")
print("-" * 60)

best_r = 999
best_params = None

for T_w1 in np.arange(0.05, 0.80, 0.05):
    roots = find_T_w2_new(T_f_test, T_w1, T_e_test, f_phi, f_beta, v0)
    for T_w2 in roots:
        T_cyc = T_f_test + T_w1 + T_e_test + T_w2
        M, d_v = one_cycle_4phase_new(T_f_test, T_w1, T_e_test, T_w2,
                                       f_phi, f_beta)
        v_end = M @ v0 + d_v
        beta_end = np.degrees(v_end[2])
        r = abs(v_end[2] / beta0)
        
        M_b = M[2:4, 2:4]
        rho = max(abs(np.linalg.eigvals(M_b)))
        
        if r < best_r:
            best_r = r
            best_params = (T_w1, T_w2, T_cyc, beta_end, r, rho, M, d_v)
        
        print(f"{T_w1:7.3f} {T_w2:7.3f} {T_cyc:7.3f} {beta_end:10.4f} "
              f"{r:8.4f} {rho:10.4f}")

# 최적으로 다중 사이클
if best_params:
    T_w1_b, T_w2_b, T_cyc_b, _, _, rho_b, M_b_mat, d_b_vec = best_params
    
    print(f"\n--- 최적 조합 ---")
    print(f"  T_w1={T_w1_b:.3f}, T_w2={T_w2_b:.3f}, T_cycle={T_cyc_b:.3f}")
    print(f"  r={best_r:.4f}, rho(M_beta)={rho_b:.4f}")
    
    eigvals_b = np.linalg.eigvals(M_b_mat[2:4, 2:4])
    print(f"  M_beta eigenvalues: {eigvals_b}")
    
    if rho_b < 1.0:
        print(f"\n  rho < 1 -> beta 수렴! 다중 사이클 시뮬레이션:")
        print(f"\n{'Cyc':>4} {'beta(deg)':>10} {'bdot(deg/s)':>12} "
              f"{'phi(deg)':>10} {'|b/b0|':>8}")
        print("-" * 50)
        
        v = np.array([0.0, 0.0, beta0, 0.0])
        print(f"{'0':>4} {np.degrees(v[2]):10.4f} {np.degrees(v[3]):12.4f} "
              f"{np.degrees(v[0]):10.4f} {1.0:8.4f}")
        
        for k in range(1, 11):
            fb_k = f_beta * ((-1)**(k-1))
            fp_k = f_phi * ((-1)**(k-1))
            
            roots_k = find_T_w2_new(T_f_test, T_w1_b, T_e_test,
                                     fp_k, fb_k, v)
            if roots_k:
                T_w2_k = min(roots_k)
            else:
                T_w2_k = T_w2_b
            
            M_k, d_k = one_cycle_4phase_new(
                T_f_test, T_w1_b, T_e_test, T_w2_k, fp_k, fb_k)
            v = M_k @ v + d_k
            ratio = abs(v[2] / beta0)
            print(f"{k:4d} {np.degrees(v[2]):10.4f} "
                  f"{np.degrees(v[3]):12.4f} "
                  f"{np.degrees(v[0]):10.6f} {ratio:8.4f}")
    else:
        print(f"\n  rho = {rho_b:.2f} >= 1 -> beta 발산. 이 파라미터로는 수렴 불가.")

print("\n완료!")

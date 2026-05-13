# -*- coding: utf-8 -*-
"""
H2 검증: 가제어성 그래미안 W(T)의 조건수가 T=T_rope/4에서 최소인가?

W(T) = ∫₀ᵀ e^{Aτ} B Bᵀ e^{Aᵀτ} dτ

W(T)의 조건수가 작은 T = "이 시간 안에 모든 방향의 제어가 균등하게 가능"
→ 이것이 T_rope/4와 일치하면 H2 성립
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

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
    T_rope = 2 * np.pi * np.sqrt((R + h_CoM) / g)
    
    return A, B, {'Mt':Mt, 'p1':p1, 'p2':p2, 'h_CoM':h_CoM, 'T_rope':T_rope}

def controllability_gramian(A, B, T, n_steps=500):
    """수치 적분으로 W(T) 계산"""
    n = A.shape[0]
    W = np.zeros((n, n))
    dt = T / n_steps
    for i in range(n_steps):
        tau = i * dt
        eAt = expm(A * tau)
        integrand = eAt @ B @ B.T @ eAt.T
        W += integrand * dt
    return W

# === 파라미터 ===
M1 = 30.0; L1 = 0.85; LC1 = L1/2
M2 = 40.0; L2 = 0.80; LC2_raw = L2/2
M_HEAD = 5.0; HEAD_R = L2/4
m2 = M2 + M_HEAD
LC2 = (M2*LC2_raw + M_HEAD*(L2 + HEAD_R)) / m2

# === 여러 R에서 검증 ===
print("=" * 70)
print("  H2 검증: W(T) 조건수의 T 의존성")
print("=" * 70)

for R_test in [0.3, 0.5, 0.7, 1.0, 1.5]:
    A, B, params = compute_system(M1, m2, L1, LC1, LC2, R_test)
    T_rope = params['T_rope']
    T_quarter = T_rope / 4
    h_CoM = params['h_CoM']
    
    print(f"\n  R = {R_test:.1f} m  (R/h_CoM = {R_test/h_CoM:.2f})")
    print(f"  T_rope = {T_rope:.4f} s,  T_rope/4 = {T_quarter:.4f} s")
    print(f"  {'T [s]':>8} {'T/T_q':>6} {'cond(W)':>15} {'λ_min':>12} {'λ_max':>12}")
    print(f"  {'-'*60}")
    
    # T를 스캔
    cond_min = float('inf')
    T_best = 0
    
    T_values = np.linspace(0.05, T_rope * 1.5, 60)
    results = []
    
    for T_val in T_values:
        try:
            W = controllability_gramian(A, B, T_val, n_steps=300)
            eigvals = np.linalg.eigvalsh(W)
            eigvals_pos = eigvals[eigvals > 1e-30]  # 양수만
            if len(eigvals_pos) >= 2:
                cond = eigvals_pos[-1] / eigvals_pos[0]
                results.append((T_val, cond, eigvals_pos[0], eigvals_pos[-1]))
                if cond < cond_min:
                    cond_min = cond
                    T_best = T_val
        except:
            pass
    
    # 주요 지점 출력
    for T_val, cond, lmin, lmax in results[::4]:  # 매 4번째
        marker = ""
        if abs(T_val - T_quarter) < (T_values[1] - T_values[0]):
            marker = " ◀ T_rope/4"
        if abs(T_val - T_best) < (T_values[1] - T_values[0]):
            marker += " ★ 최소"
        print(f"  {T_val:8.3f} {T_val/T_quarter:6.2f} {cond:15.2f} {lmin:12.4e} {lmax:12.4e}{marker}")
    
    print(f"\n  → 조건수 최소: T = {T_best:.4f} s (T/T_quarter = {T_best/T_quarter:.3f})")
    print(f"     T_rope/4   : T = {T_quarter:.4f} s")
    print(f"     차이        : {abs(T_best - T_quarter)/T_quarter * 100:.1f}%")

# === 특정 R에서 더 세밀하게 ===
print(f"\n{'='*70}")
print("  상세 분석: R = 0.7 m (조건수 최소 후보)")
print("=" * 70)

R_detail = 0.7
A, B, params = compute_system(M1, m2, L1, LC1, LC2, R_detail)
T_rope = params['T_rope']
T_quarter = T_rope / 4

T_fine = np.linspace(0.02, T_rope, 200)
conds = []
for T_val in T_fine:
    try:
        W = controllability_gramian(A, B, T_val, n_steps=300)
        eigvals = np.linalg.eigvalsh(W)
        eigvals_pos = eigvals[eigvals > 1e-30]
        if len(eigvals_pos) >= 2:
            conds.append(eigvals_pos[-1] / eigvals_pos[0])
        else:
            conds.append(float('inf'))
    except:
        conds.append(float('inf'))

conds = np.array(conds)
idx_min = np.argmin(conds)
T_opt = T_fine[idx_min]

print(f"  T_rope/4 = {T_quarter:.4f} s")
print(f"  W(T) 조건수 최소: T = {T_opt:.4f} s (T/T_quarter = {T_opt/T_quarter:.3f})")
print(f"  조건수 at T_rope/4: {conds[np.argmin(np.abs(T_fine - T_quarter))]:.2f}")
print(f"  조건수 at T_opt:    {conds[idx_min]:.2f}")

# === A행렬 고유값 분석 ===
print(f"\n{'='*70}")
print("  A 행렬 고유값 (시스템의 자연 시간 스케일)")
print("=" * 70)

for R_test in [0.3, 0.5, 0.7, 1.0, 1.5]:
    A, B, params = compute_system(M1, m2, L1, LC1, LC2, R_test)
    eigs = np.linalg.eigvals(A)
    T_rope = params['T_rope']
    
    print(f"\n  R = {R_test:.1f} m (T_rope/4 = {T_rope/4:.3f} s):")
    for e in sorted(eigs, key=lambda x: abs(x)):
        if abs(e.imag) > 0.01:
            freq = abs(e.imag)
            period = 2*np.pi/freq
            print(f"    λ = {e.real:+.3f} ± {abs(e.imag):.3f}j  "
                  f"(T = {period:.3f} s, T/4 = {period/4:.3f} s)")
        elif abs(e) > 0.01:
            print(f"    λ = {e.real:+.3f}  (τ = {1/abs(e.real):.3f} s)")

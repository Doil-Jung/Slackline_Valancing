# -*- coding: utf-8 -*-
"""
R* (CoM invariance condition) vs rho_foot (foot trajectory curvature)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.integrate import solve_ivp

# === V14 기본 파라미터 (human-scale) ===
M1 = 30.0;  L1 = 0.85;  LC1 = L1/2
M2 = 40.0;  L2 = 0.80;  LC2_raw = L2/2
M_HEAD = 5.0; HEAD_R = L2/4

# 유효 상체 (상체 + 머리)
m2 = M2 + M_HEAD
d2 = LC2_raw; d_h = L2 + HEAD_R
LC2 = (M2*d2 + M_HEAD*d_h) / m2
I1_cm = M1*L1**2/12
I2_body = M2*L2**2/12 + M2*(d2 - LC2)**2
I2_head = 2/5*M_HEAD*HEAD_R**2 + M_HEAD*(d_h - LC2)**2
I2_cm = I2_body + I2_head

Mt = M1 + m2
p1 = M1*LC1 + m2*L1
p2 = m2*LC2
p3 = m2*L1*LC2
I_alpha = M1*LC1**2 + I1_cm + m2*L1**2
I_theta = m2*LC2**2 + I2_cm
h_CoM = (p1 + p2) / Mt

print("="*55)
print("  파라미터")
print(f"  M1={M1} m2_eff={m2} L1={L1} LC1={LC1}")
print(f"  LC2_eff={LC2:.4f}  I1_cm={I1_cm:.4f}  I2_cm={I2_cm:.4f}")
print(f"  p1={p1:.2f}  p2={p2:.2f}  p1/p2={p1/p2:.3f}")
print(f"  h_CoM={h_CoM:.4f} m")
print("="*55)

# ============================================================
#  1. R* 계산: [0,p1,p2] · M^-1(R) · [0,-1,1]^T = 0
# ============================================================
def com_invariance(R):
    """CoM invariance condition as function of R. Returns the scalar value."""
    M = np.array([
        [Mt*R**2, R*p1, R*p2],
        [R*p1, I_alpha, p3],
        [R*p2, p3, I_theta]
    ])
    E = np.array([0, -1, 1])
    w = np.array([0, p1, p2])
    try:
        Minv_E = np.linalg.solve(M, E)
        return w @ Minv_E
    except np.linalg.LinAlgError:
        return float('inf')

# Scan R to find zero crossing
R_scan = np.linspace(0.01, 10.0, 10000)
vals = [com_invariance(R) for R in R_scan]
vals = np.array(vals)

# Find sign changes
R_star_list = []
for i in range(len(vals)-1):
    if vals[i]*vals[i+1] < 0:
        # Bisect
        a, b = R_scan[i], R_scan[i+1]
        for _ in range(100):
            mid = (a+b)/2
            if com_invariance(a)*com_invariance(mid) < 0:
                b = mid
            else:
                a = mid
        R_star_list.append((a+b)/2)

print(f"\n[R*] CoM invariance condition zeros:")
if R_star_list:
    for rs in R_star_list:
        print(f"  R* = {rs:.6f} m  (R*/h_CoM = {rs/h_CoM:.3f})")
else:
    print("  No zero found in [0.01, 10.0]")
    print(f"  Value at R=0.01: {com_invariance(0.01):.6f}")
    print(f"  Value at R=1.0:  {com_invariance(1.0):.6f}")
    print(f"  Value at R=10.0: {com_invariance(10.0):.6f}")
    print(f"  -> Condition never reaches zero: p1!=p2 cannot be compensated by R alone")
    print(f"  -> This means H1 (p1=p2) is an INDEPENDENT necessary condition")

# ============================================================
#  2. ρ_foot 계산: 자유공간에서 접힘 시 발 궤적의 곡률 반경
# ============================================================
def foot_pos(alpha, delta):
    """Foot (ankle) position with CoM at origin."""
    theta = alpha + delta
    x_a = -(p1*np.sin(alpha) + p2*np.sin(theta)) / Mt
    z_a = -(p1*np.cos(alpha) + p2*np.cos(theta)) / Mt
    return x_a, z_a

def angular_momentum_coeffs(alpha, delta):
    """Compute f1, f2 such that L_CoM = f1*alpha_dot + f2*delta_dot."""
    theta = alpha + delta
    # Positions relative to system CoM (at origin)
    x_a, z_a = foot_pos(alpha, delta)
    x1 = x_a + LC1*np.sin(alpha)
    z1 = z_a + LC1*np.cos(alpha)
    x2 = x_a + L1*np.sin(alpha) + LC2*np.sin(theta)
    z2 = z_a + L1*np.cos(alpha) + LC2*np.cos(theta)

    # d/dalpha derivatives (dtheta/dalpha = 1)
    dxa_da = -(p1*np.cos(alpha) + p2*np.cos(theta)) / Mt
    dza_da = (p1*np.sin(alpha) + p2*np.sin(theta)) / Mt
    dx1_da = dxa_da + LC1*np.cos(alpha)
    dz1_da = dza_da - LC1*np.sin(alpha)
    dx2_da = dxa_da + L1*np.cos(alpha) + LC2*np.cos(theta)
    dz2_da = dza_da - L1*np.sin(alpha) - LC2*np.sin(theta)

    # d/ddelta derivatives (dalpha/ddelta = 0, dtheta/ddelta = 1)
    dxa_dd = -p2*np.cos(theta) / Mt
    dza_dd = p2*np.sin(theta) / Mt
    dx1_dd = dxa_dd
    dz1_dd = dza_dd
    dx2_dd = dxa_dd + LC2*np.cos(theta)
    dz2_dd = dza_dd - LC2*np.sin(theta)

    # L = I1*alpha_dot + I2*theta_dot + orbital terms
    # theta_dot = alpha_dot + delta_dot
    f1 = (I1_cm + I2_cm
          + M1*(x1*dz1_da - z1*dx1_da)
          + m2*(x2*dz2_da - z2*dx2_da))
    f2 = (I2_cm
          + M1*(x1*dz1_dd - z1*dx1_dd)
          + m2*(x2*dz2_dd - z2*dx2_dd))
    return f1, f2

# Integrate dα/dδ = -f2/f1 with α(0)=0
def dalpha_ddelta(delta, alpha):
    f1, f2 = angular_momentum_coeffs(alpha[0], delta)
    return [-f2/f1]

sol = solve_ivp(dalpha_ddelta, [0, 0.5], [0.0], max_step=0.001, dense_output=True)

# Compute foot trajectory
deltas = np.linspace(0, 0.3, 1000)
alphas = sol.sol(deltas)[0]
xs, zs = [], []
for a, d in zip(alphas, deltas):
    x, z = foot_pos(a, d)
    xs.append(x); zs.append(z)
xs = np.array(xs); zs = np.array(zs)

# Curvature at δ=0: fit z = c*x^2 near origin
# Use first ~100 points
mask = np.abs(xs) < 0.05  # small x range
if mask.sum() > 10:
    coeffs = np.polyfit(xs[mask], zs[mask] - zs[0], 2)
    c = coeffs[0]  # coefficient of x^2
    rho_foot = 1.0 / (2*abs(c)) if abs(c) > 1e-10 else float('inf')
else:
    # fallback: use all points
    coeffs = np.polyfit(xs[:100], zs[:100] - zs[0], 2)
    c = coeffs[0]
    rho_foot = 1.0 / (2*abs(c)) if abs(c) > 1e-10 else float('inf')

print(f"\n[rho_foot] Foot trajectory curvature radius:")
print(f"  Parabola fit: z = {c:.6f} * x^2")
print(f"  rho_foot = {rho_foot:.6f} m  (rho/h_CoM = {rho_foot/h_CoM:.3f})")
print(f"  dAlpha/dDelta at 0: {-angular_momentum_coeffs(0,0)[1]/angular_momentum_coeffs(0,0)[0]:.6f}")

# ============================================================
#  3. 비교
# ============================================================
print(f"\n{'='*55}")
print(f"  비교 결과")
print(f"{'='*55}")
print(f"  h_CoM     = {h_CoM:.4f} m")
print(f"  p1 = {p1:.2f},  p2 = {p2:.2f},  p1-p2 = {p1-p2:.2f}")
if R_star_list:
    for rs in R_star_list:
        print(f"  R*        = {rs:.4f} m  ({rs/h_CoM:.3f} × h_CoM)")
print(f"  ρ_foot    = {rho_foot:.4f} m  ({rho_foot/h_CoM:.3f} × h_CoM)")
if R_star_list:
    ratio = R_star_list[0] / rho_foot
    print(f"  R*/ρ_foot = {ratio:.4f}")
    if abs(ratio - 1.0) < 0.05:
        print(f"  → 거의 일치! H1+H3 통합 가능")
    else:
        print(f"  → 불일치. H1(p1=p2)과 H3(R=ρ)는 독립 조건")
print(f"  Paoletti range: {1.0*h_CoM:.4f} ~ {1.5*h_CoM:.4f} m")

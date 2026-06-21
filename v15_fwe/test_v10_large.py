# -*- coding: utf-8 -*-
"""
Sweep FWE parameters for the v10 large-scale robot model (M=75kg)
to find convergent cases by exploring wider T_f, T_w1, and R_rope values.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Base parameters (large-scale)
g = 9.81
M1, m2 = 30.0, 45.0
L1 = 0.85
L_upper = 0.85
LC1 = L1 / 2.0
LC2 = L_upper / 2.0
Mt = M1 + m2
I1 = M1*L1**2/12
I2 = m2*L_upper**2/12
p1 = M1*LC1 + m2*L1
p2 = m2*LC2
ps = p1+p2
h_com = (M1*LC1 + m2*(L1+LC2))/Mt
c_foot = p2/ps
J_aa = M1*LC1**2 + I1 + m2*L1**2
J_tt = m2*LC2**2 + I2
J_at = m2*L1*LC2
J_tot = J_aa + J_tt + 2*J_at
C_sd = (-J_aa*p2 + J_tt*p1 + J_at*(p1-p2))/ps**2

def check_convergence(R_rope, eps, T_f, T_w1):
    M11 = Mt*R_rope**2
    M12 = R_rope * Mt  # Fixed coupling term (Mt scaling added to match dimensional consistency)
    M22 = J_tot/ps**2
    det_M = M11*M22 - M12**2
    if det_M <= 0:
        # If unphysical det_M, fallback to the original code's formula
        M12 = R_rope
        det_M = M11*M22 - M12**2
        
    iM11 = M22/det_M
    iM12 = -M12/det_M
    iM22 = M11/det_M
    gMR = g*Mt*R_rope
    g_ps = g/ps
    
    w_eff = np.sqrt(iM11*gMR)
    lam_eff = np.sqrt(iM22*g_ps)
    T_quarter = np.pi/(2*w_eff)
    
    beta0 = np.radians(5)
    sigma0 = Mt * h_com * beta0
    d_fold = h_com * (1 + eps) * abs(beta0) / c_foot
    a_fold = 4 * d_fold / T_f**2
    
    t1 = T_f
    t2 = T_f + T_w1
    t3 = T_f + T_w1 + T_f
    
    def rhs(t, y):
        phi, dp, sig, ds = y
        # delta acceleration profile
        if t < T_f/2:           dd = +a_fold
        elif t < T_f:           dd = -a_fold
        elif t < t2:            dd = 0.0
        elif t < t2 + T_f/2:    dd = -a_fold
        elif t < t3:            dd = +a_fold
        else:                   dd = 0.0
        
        r1 = -gMR * phi
        r2 = g_ps * sig - C_sd * dd
        return [dp, iM11*r1 + iM12*r2, ds, iM12*r1 + iM22*r2]
    
    T_w2_max = 4 * T_quarter
    T_total_max = t3 + T_w2_max
    
    sol = solve_ivp(rhs, (0, T_total_max), [0, 0, sigma0, 0],
                    method='RK45', rtol=1e-8, atol=1e-10,
                    max_step=T_f/20, dense_output=True)
    
    w2_margin = 0.005
    phi_arr = sol.y[0]
    
    for k in range(len(sol.t) - 1):
        if sol.t[k] < t3 + w2_margin:
            continue
        if phi_arr[k] * phi_arr[k+1] < 0:
            try:
                t_zero = brentq(lambda t: sol.sol(t)[0],
                                sol.t[k], sol.t[k+1], xtol=1e-6)
                yf = sol.sol(t_zero)
                T_w2 = t_zero - t3
                beta_f = yf[2] / (Mt * h_com)
                r = beta_f / beta0
                return T_w2, r, t_zero
            except Exception as e:
                continue
    return None

print("Starting sweep for large-scale v10 robot model...")
print(f"{'R_rope':>6} {'eps':>5} {'T_f':>5} {'T_w1':>6} | {'T_w2':>6} {'T_tot':>6} | {'r':>8} | Status")
print("-" * 70)

results = []
for R in [1.0, 0.8, 0.6, 0.4]:
    for eps in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]:
        for T_f in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
            for T_w1 in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
                res = check_convergence(R, eps, T_f, T_w1)
                if res is not None:
                    T_w2, r, T_tot = res
                    status = "CONV" if abs(r) < 1.0 else "DIV"
                    if abs(r) < 0.1:
                        status = "EXCELLENT"
                    if abs(r) < 1.0:
                        print(f"{R:6.2f} {eps:5.1f} {T_f:5.2f} {T_w1:6.2f} | {T_w2:6.2f} {T_tot:6.2f} | {r:8.4f} | {status}")
                        results.append((R, eps, T_f, T_w1, T_w2, T_tot, r))
                else:
                    pass

print(f"\nSweep completed. Found {len(results)} convergent configurations.")
if results:
    results.sort(key=lambda x: abs(x[6]))
    best = results[0]
    print(f"Best configuration: R={best[0]}, eps={best[1]}, T_f={best[2]}s, T_w1={best[3]}s, T_w2={best[4]:.2f}s -> r={best[6]:.4f}")
else:
    print("No convergent configurations found under checked ranges.")

"""
자기일관성 조건 탐색:
phi(T_total) = 0 이 되는 T_w를 찾아서 수렴비 계산
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === System parameters (same as lagrangian_coupling.py) ===
g = 9.81; R_rope = 1.0
M1, m2 = 30.0, 45.0
L1, LC1, LC2 = 0.85, 0.30, 0.40
L_upper = 0.85
Mt = M1 + m2
I1 = M1 * L1**2 / 12
I2 = m2 * L_upper**2 / 12

p1 = M1 * LC1 + m2 * L1
p2 = m2 * LC2
ps = p1 + p2
h_com = (M1 * LC1 + m2 * (L1 + LC2)) / Mt
c_foot = p2 / ps

J_aa = M1 * LC1**2 + I1 + m2 * L1**2
J_tt = m2 * LC2**2 + I2
J_at = m2 * L1 * LC2
J_tot = J_aa + J_tt + 2 * J_at
C_sd = (-J_aa * p2 + J_tt * p1 + J_at * (p1 - p2)) / ps**2

M11 = Mt * R_rope**2; M12 = R_rope; M22 = J_tot / ps**2
det_M = M11 * M22 - M12**2
iM11 = M22 / det_M; iM12 = -M12 / det_M; iM22 = M11 / det_M
gMR = g * Mt * R_rope; g_ps = g / ps

w2_eff = iM11 * gMR; lam2_eff = iM22 * g_ps
w_eff = np.sqrt(w2_eff); lam_eff = np.sqrt(lam2_eff)
K_phi = -iM12 * gMR; K_delta = -iM22 * C_sd; K_phidelta = -iM12 * C_sd

print(f"w_eff={w_eff:.4f}, lam_eff={lam_eff:.4f}")
print(f"K_phi={K_phi:.1f}, K_delta={K_delta:.4f}, K_phidelta={K_phidelta:.4f}")

def solve_full_coupled(beta0, eps, T_f, T_w):
    """Full coupled ODE, returns (phi_f, dphi_f, beta_f, dbeta_f)"""
    sigma0 = Mt * h_com * beta0
    delta_fold = h_com * (1 + eps) * abs(beta0) / c_foot * np.sign(beta0)
    a = 4 * delta_fold / T_f**2
    T_total = 2 * T_f + T_w

    def get_dd(t):
        if t < T_f/2: return +a
        elif t < T_f: return -a
        elif t < T_f + T_w: return 0.0
        elif t < T_f + T_w + T_f/2: return -a
        else: return +a

    def rhs(t, y):
        phi, dp, sig, ds = y
        dd = get_dd(t)
        r1 = -gMR * phi
        r2 = g_ps * sig - C_sd * dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

    sol = solve_ivp(rhs, (0, T_total), [0, 0, sigma0, 0],
                    method='RK45', rtol=1e-10, atol=1e-12, max_step=T_f/50)
    yf = sol.y[:, -1]
    return yf[0], yf[1], yf[2]/(Mt*h_com), yf[3]/(Mt*h_com)

# ==============================
# 1. phi=0 복귀하는 T_w 찾기
# ==============================
print(f"\n{'='*60}")
print("Self-consistency search: T_w that gives phi_final=0")
print(f"{'='*60}")

beta0 = np.radians(5)

results = []
for eps in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
    for T_f in [0.03, 0.05, 0.08, 0.10]:
        # Sweep T_w to find phi=0 crossing
        Tw_test = np.linspace(0.01, 0.5, 200)
        phi_vals = []
        for Tw in Tw_test:
            try:
                pf, _, bf, _ = solve_full_coupled(beta0, eps, T_f, Tw)
                phi_vals.append(pf)
            except:
                phi_vals.append(np.nan)
        phi_vals = np.array(phi_vals)

        # Find zero crossings
        for k in range(len(phi_vals)-1):
            if np.isnan(phi_vals[k]) or np.isnan(phi_vals[k+1]):
                continue
            if phi_vals[k] * phi_vals[k+1] < 0:
                # Brent's method
                try:
                    Tw_opt = brentq(
                        lambda tw: solve_full_coupled(beta0, eps, T_f, tw)[0],
                        Tw_test[k], Tw_test[k+1], xtol=1e-8)
                    pf, _, bf, _ = solve_full_coupled(beta0, eps, T_f, Tw_opt)
                    r = abs(bf / beta0)
                    T_total = 2*T_f + Tw_opt
                    results.append((eps, T_f, Tw_opt, T_total, r, np.degrees(bf), np.degrees(pf)))
                    print(f"eps={eps:.1f} T_f={T_f:.3f} T_w*={Tw_opt:.5f} "
                          f"T_tot={T_total:.4f} r={r:.4f} beta_f={np.degrees(bf):.3f}deg "
                          f"phi_f={np.degrees(pf):.2e}deg")
                except:
                    pass

# ==============================
# 2. r vs eps plot (self-consistent)
# ==============================
if results:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for T_f_val in [0.03, 0.05, 0.08, 0.10]:
        data = [(r[0], r[4]) for r in results if abs(r[1] - T_f_val) < 0.001]
        if data:
            eps_arr, r_arr = zip(*data)
            axes[0].plot(eps_arr, r_arr, 'o-', label=f'T_f={T_f_val}')

    axes[0].axhline(1.0, color='red', ls='--', label='r=1 (boundary)')
    axes[0].set_xlabel('epsilon (overshoot)')
    axes[0].set_ylabel('convergence ratio r')
    axes[0].set_title('Self-consistent: r vs epsilon\n(T_w chosen so phi_final=0)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for T_f_val in [0.03, 0.05, 0.08, 0.10]:
        data = [(r[0], r[2]) for r in results if abs(r[1] - T_f_val) < 0.001]
        if data:
            eps_arr, tw_arr = zip(*data)
            axes[1].plot(eps_arr, tw_arr, 'o-', label=f'T_f={T_f_val}')

    axes[1].set_xlabel('epsilon')
    axes[1].set_ylabel('T_w* [s]')
    axes[1].set_title('Self-consistent T_w* vs epsilon')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = __file__.replace('.py', '_selfconsistent.png')
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.close()

print("\nDone!")

"""phi_final vs T_w 곡선 진단 — 자기일관성 조건의 영점 위치 파악"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === System parameters ===
g = 9.81; R_rope = 1.0
M1, m2 = 30.0, 45.0; L1, LC1, LC2 = 0.85, 0.30, 0.40; L_upper = 0.85
Mt = M1 + m2; I1 = M1*L1**2/12; I2 = m2*L_upper**2/12
p1 = M1*LC1 + m2*L1; p2 = m2*LC2; ps = p1+p2
h_com = (M1*LC1 + m2*(L1+LC2))/Mt; c_foot = p2/ps
J_aa = M1*LC1**2 + I1 + m2*L1**2; J_tt = m2*LC2**2 + I2
J_at = m2*L1*LC2; J_tot = J_aa + J_tt + 2*J_at
C_sd = (-J_aa*p2 + J_tt*p1 + J_at*(p1-p2))/ps**2
M11 = Mt*R_rope**2; M12 = R_rope; M22 = J_tot/ps**2
det_M = M11*M22 - M12**2
iM11 = M22/det_M; iM12 = -M12/det_M; iM22 = M11/det_M
gMR = g*Mt*R_rope; g_ps = g/ps

def solve_cycle(beta0, eps, T_f, T_w):
    sigma0 = Mt * h_com * beta0
    d_fold = h_com*(1+eps)*abs(beta0)/c_foot * np.sign(beta0)
    a = 4*d_fold/T_f**2
    T_total = 2*T_f + T_w
    def get_dd(t):
        if t < T_f/2: return +a
        elif t < T_f: return -a
        elif t < T_f+T_w: return 0.0
        elif t < T_f+T_w+T_f/2: return -a
        else: return +a
    def rhs(t, y):
        phi, dp, sig, ds = y
        dd = get_dd(t)
        r1 = -gMR*phi; r2 = g_ps*sig - C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]
    sol = solve_ivp(rhs, (0, T_total), [0, 0, sigma0, 0],
                    method='RK45', rtol=1e-10, atol=1e-12, max_step=T_f/50)
    yf = sol.y[:, -1]
    return yf[0], yf[1], yf[2]/(Mt*h_com), yf[3]/(Mt*h_com)

beta0 = np.radians(5)
T_w_range = np.linspace(0.001, 1.0, 300)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax_idx, (eps, T_f) in enumerate([(0.5, 0.05), (1.0, 0.05), (1.0, 0.10), (2.0, 0.05)]):
    ax = axes.flat[ax_idx]
    phi_f_list = []
    beta_f_list = []
    for Tw in T_w_range:
        try:
            pf, _, bf, _ = solve_cycle(beta0, eps, T_f, Tw)
            phi_f_list.append(np.degrees(pf))
            beta_f_list.append(bf / beta0)  # r = beta_f/beta0
        except:
            phi_f_list.append(np.nan)
            beta_f_list.append(np.nan)

    ax.plot(T_w_range, phi_f_list, 'b-', lw=1.5, label='phi_final [deg]')
    ax.axhline(0, color='red', ls='--', alpha=0.7)
    ax.set_xlabel('T_w [s]')
    ax.set_ylabel('phi_final [deg]')
    ax.set_title(f'eps={eps}, T_f={T_f}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-20, 20)

    # secondary axis for r
    ax2 = ax.twinx()
    ax2.plot(T_w_range, beta_f_list, 'r-', lw=1, alpha=0.5, label='r=beta_f/beta0')
    ax2.axhline(1.0, color='orange', ls=':', alpha=0.5)
    ax2.axhline(-1.0, color='orange', ls=':', alpha=0.5)
    ax2.set_ylabel('r (convergence ratio)', color='red')
    ax2.set_ylim(-5, 5)

plt.suptitle('phi_final vs T_w  (looking for self-consistent T_w where phi=0)', y=1.02)
plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
plt.close()
print("Done!")

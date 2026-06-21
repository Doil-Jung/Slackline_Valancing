"""Fine-grained search: very small T_w range + negative beta (fold other way)"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
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
    return yf[0], yf[1], yf[2]/(Mt*h_com), yf[3]/(Mt*h_com), sol

beta0 = np.radians(5)

# Fine-grained T_w sweep (0 to 0.05s)
T_w_fine = np.linspace(0.0005, 0.05, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax_idx, (eps, T_f) in enumerate([(0.5, 0.05), (1.0, 0.03), (1.0, 0.05), (2.0, 0.03)]):
    ax = axes.flat[ax_idx]
    phi_f = []; beta_f = []
    for Tw in T_w_fine:
        pf, _, bf, _, _ = solve_cycle(beta0, eps, T_f, Tw)
        phi_f.append(np.degrees(pf))
        beta_f.append(bf/beta0)
    
    ax.plot(T_w_fine*1000, phi_f, 'b-', lw=2, label='phi_final')
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('T_w [ms]')
    ax.set_ylabel('phi_final [deg]')
    ax.set_title(f'eps={eps}, T_f={T_f}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    ax2 = ax.twinx()
    ax2.plot(T_w_fine*1000, beta_f, 'r-', lw=1, alpha=0.5, label='r')
    ax2.axhline(1, color='orange', ls=':', alpha=0.5)
    ax2.axhline(0, color='orange', ls=':', alpha=0.5)
    ax2.set_ylabel('r', color='red')
    ax2.legend(loc='upper right')
    
    # Find zero crossings
    phi_arr = np.array(phi_f)
    for k in range(len(phi_arr)-1):
        if phi_arr[k]*phi_arr[k+1] < 0:
            try:
                Tw_opt = brentq(lambda tw: solve_cycle(beta0, eps, T_f, tw)[0],
                                T_w_fine[k], T_w_fine[k+1], xtol=1e-10)
                _, _, bf_opt, _, _ = solve_cycle(beta0, eps, T_f, Tw_opt)
                r_opt = bf_opt/beta0
                ax.axvline(Tw_opt*1000, color='green', ls='-', lw=2, alpha=0.7)
                print(f"FOUND: eps={eps} T_f={T_f} T_w*={Tw_opt*1000:.3f}ms "
                      f"r={r_opt:.4f} beta_f={np.degrees(bf_opt):.3f}deg")
            except:
                pass

plt.suptitle('Fine-grained phi_final vs T_w (small T_w range)', y=1.02)
plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

# Also try: what if we look at phi_final across many PERIODS of T_w?
print("\n--- Also checking long T_w (multiple pendulum periods) ---")
T_quarter = np.pi / (2 * np.sqrt(iM11 * gMR))
print(f"T_quarter = {T_quarter*1000:.1f} ms")

T_w_long = np.linspace(0.001, 4*T_quarter, 300)
eps, T_f = 1.0, 0.05
phi_long = []
for Tw in T_w_long:
    try:
        pf, _, _, _, _ = solve_cycle(beta0, eps, T_f, Tw)
        phi_long.append(np.degrees(pf))
    except:
        phi_long.append(np.nan)

fig2, ax = plt.subplots(figsize=(12, 4))
ax.plot(T_w_long*1000, phi_long, 'b-')
ax.axhline(0, color='red', ls='--')
for n in range(1, 5):
    ax.axvline(n*T_quarter*1000, color='gray', ls=':', alpha=0.3, label=f'{n}*T_q' if n==1 else '')
ax.set_xlabel('T_w [ms]')
ax.set_ylabel('phi_final [deg]')
ax.set_title(f'phi_final over multiple pendulum periods (eps={eps}, T_f={T_f})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
out2 = __file__.replace('.py', '_long.png')
plt.savefig(out2, dpi=150)
print(f"Saved: {out2}")
print("Done!")

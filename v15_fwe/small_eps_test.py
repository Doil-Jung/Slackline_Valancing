"""Small epsilon 4-phase: eps = 0.01 ~ 0.3, show phi trajectory in W2"""
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
w_eff = np.sqrt(iM11*gMR); lam_eff = np.sqrt(iM22*g_ps)
T_quarter = np.pi/(2*w_eff)

print(f"w_eff={w_eff:.4f}, lam_eff={lam_eff:.4f}, T_q={T_quarter*1000:.1f}ms")

def run_4phase(beta0, eps, T_f, T_w1):
    """Run full 4-phase, return time arrays and state for plotting"""
    sigma0 = Mt * h_com * beta0
    d_fold = h_com * (1 + eps) * abs(beta0) / c_foot * np.sign(beta0)
    a_fold = 4 * d_fold / T_f**2
    t2 = T_f + T_w1; t3 = t2 + T_f
    T_w2_max = 4 * T_quarter
    T_end = t3 + T_w2_max

    def get_dd(t):
        if t < T_f/2: return +a_fold
        elif t < T_f: return -a_fold
        elif t < t2: return 0.0
        elif t < t2+T_f/2: return -a_fold
        elif t < t3: return +a_fold
        else: return 0.0

    def rhs(t, y):
        phi, dp, sig, ds = y
        dd = get_dd(t)
        r1 = -gMR*phi; r2 = g_ps*sig - C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

    sol = solve_ivp(rhs, (0, T_end), [0, 0, sigma0, 0],
                    method='RK45', rtol=1e-10, atol=1e-12,
                    max_step=T_f/30, dense_output=True)

    # Find phi=0 in W2
    t_dense = np.linspace(t3+0.002, T_end, 5000)
    phi_dense = np.array([sol.sol(t)[0] for t in t_dense])

    zero_crossings = []
    for k in range(len(phi_dense)-1):
        if phi_dense[k] * phi_dense[k+1] < 0:
            t_zero = brentq(lambda t: sol.sol(t)[0], t_dense[k], t_dense[k+1], xtol=1e-10)
            yf = sol.sol(t_zero)
            beta_f = yf[2]/(Mt*h_com)
            zero_crossings.append((t_zero - t3, t_zero, beta_f, beta_f/beta0))

    return sol, t3, zero_crossings

beta0 = np.radians(5)

# ==============================
# 1. Phi trajectory for small eps
# ==============================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
eps_test = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
T_f = 0.05; T_w1 = 0.005

for idx, eps in enumerate(eps_test):
    ax = axes.flat[idx]
    sol, t3, zeros = run_4phase(beta0, eps, T_f, T_w1)

    t_plot = np.linspace(0, t3 + 2*T_quarter, 3000)
    phi_plot = [np.degrees(sol.sol(t)[0]) for t in t_plot]
    beta_plot = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_plot]

    ax.plot(np.array(t_plot)*1000, phi_plot, 'b-', lw=1.5, label='phi')
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.axvline(t3*1000, color='gray', ls=':', alpha=0.5, label='W2 start')

    for zc in zeros:
        ax.axvline((zc[1])*1000, color='green', ls='-', lw=2, alpha=0.7)
        ax.text((zc[1])*1000, ax.get_ylim()[1]*0.8 if len(phi_plot)>0 else 1,
               f'r={zc[3]:.3f}', fontsize=7, color='green', rotation=90)

    ax2 = ax.twinx()
    ax2.plot(np.array(t_plot)*1000, beta_plot, 'r-', lw=0.8, alpha=0.4)
    ax2.set_ylabel('beta [deg]', color='red', fontsize=7)

    status = f"{len(zeros)} zeros" if zeros else "NO zero"
    if zeros:
        status += f", r={zeros[0][3]:.4f}"
    ax.set_title(f'eps={eps}, {status}', fontsize=9)
    ax.set_xlabel('time [ms]', fontsize=7)
    ax.set_ylabel('phi [deg]', fontsize=7)

plt.suptitle(f'Small eps scan: T_f={T_f*1000:.0f}ms, T_w1={T_w1*1000:.0f}ms, beta0=5deg', fontsize=12)
plt.tight_layout()
out1 = __file__.replace('.py', '_trajectories.png')
plt.savefig(out1, dpi=150)
print(f"Saved: {out1}")
plt.close()

# ==============================
# 2. Systematic sweep: eps x T_w1
# ==============================
print(f"\n{'='*70}")
print("4-Phase small-eps sweep")
print(f"{'='*70}")

eps_sweep = np.concatenate([np.arange(0.01, 0.10, 0.01), np.arange(0.10, 0.51, 0.02)])
tw1_sweep = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03]

print(f"{'eps':>5} {'T_w1':>6} | {'T_w2':>7} {'T_tot':>7} | {'r':>8} | note")
print("-" * 60)

found_any = False
for eps in eps_sweep:
    for tw1 in tw1_sweep:
        sol, t3, zeros = run_4phase(beta0, eps, T_f, tw1)
        if zeros:
            zc = zeros[0]
            found_any = True
            note = "CONV" if abs(zc[3]) < 1 else "DIV"
            print(f"{eps:5.3f} {tw1*1000:5.1f}ms | {zc[0]*1000:6.1f}ms {zc[1]*1000:6.0f}ms | "
                  f"{zc[3]:8.4f} | {note}")

if not found_any:
    print("No phi=0 crossings found for any (eps, T_w1) combination!")

# ==============================
# 3. Even smaller eps: 0.001 to 0.01
# ==============================
print(f"\n{'='*70}")
print("Ultra-small eps sweep (0.001 to 0.01)")
print(f"{'='*70}")

for eps in [0.001, 0.002, 0.005, 0.008, 0.01]:
    for tw1 in [0.001, 0.005, 0.01]:
        sol, t3, zeros = run_4phase(beta0, eps, T_f, tw1)
        if zeros:
            zc = zeros[0]
            print(f"eps={eps:.4f} T_w1={tw1*1000:.1f}ms | T_w2={zc[0]*1000:.1f}ms "
                  f"r={zc[3]:.6f}")
        else:
            print(f"eps={eps:.4f} T_w1={tw1*1000:.1f}ms | no phi=0")

print("\nDone!")

"""4-Phase with correct eps range (>= 0.8 for overshoot)"""
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

def run_full_cycle(beta0, eps, T_f, T_w1):
    """Run F-W1-E-W2 cycle, find phi=0 in W2, return results"""
    sigma0 = Mt * h_com * beta0
    d_fold = h_com * (1 + eps) * abs(beta0) / c_foot * np.sign(beta0)
    a = 4 * d_fold / T_f**2
    t2 = T_f + T_w1; t3 = t2 + T_f
    T_w2_max = 4 * T_quarter
    T_end = t3 + T_w2_max

    def get_dd(t):
        if t < T_f/2: return +a
        elif t < T_f: return -a
        elif t < t2: return 0.0
        elif t < t2+T_f/2: return -a
        elif t < t3: return +a
        else: return 0.0

    def rhs(t, y):
        phi, dp, sig, ds = y
        dd = get_dd(t)
        r1 = -gMR*phi; r2 = g_ps*sig - C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

    sol = solve_ivp(rhs, (0, T_end), [0, 0, sigma0, 0],
                    method='RK45', rtol=1e-10, atol=1e-12,
                    max_step=T_f/40, dense_output=True)

    # Find phi=0 crossings in W2
    t_w2 = np.linspace(t3 + 0.003, T_end, 5000)
    phi_w2 = np.array([sol.sol(t)[0] for t in t_w2])

    crossings = []
    for k in range(len(phi_w2)-1):
        if phi_w2[k] * phi_w2[k+1] < 0:
            try:
                tz = brentq(lambda t: sol.sol(t)[0], t_w2[k], t_w2[k+1], xtol=1e-10)
                yf = sol.sol(tz)
                bf = yf[2]/(Mt*h_com)
                crossings.append({
                    'T_w2': tz - t3, 't_zero': tz,
                    'beta_f': bf, 'r': bf/beta0,
                    'phi_dot': yf[1]
                })
            except:
                pass
    return sol, t3, crossings

beta0 = np.radians(5)
T_f = 0.05

# ==============================
# 1. Sweep eps >= 0.8
# ==============================
print(f"\n{'='*70}")
print(f"4-Phase with eps >= 0.8 (beta0={np.degrees(beta0):.1f}deg, T_f={T_f*1000:.0f}ms)")
print(f"{'='*70}")

eps_vals = [0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
tw1_vals = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]

print(f"\n{'eps':>5} {'Tw1':>5} | {'Tw2':>7} {'Ttot':>7} | {'r':>8} {'beta_f':>8} {'phi_d':>8} | note")
print("-" * 80)

all_results = []
for eps in eps_vals:
    for tw1 in tw1_vals:
        sol, t3, crossings = run_full_cycle(beta0, eps, T_f, tw1)
        if crossings:
            c = crossings[0]  # first crossing
            note = "CONV" if abs(c['r']) < 1 else "DIV"
            if abs(c['r']) < 0.05: note = "~ZERO"
            d_fold = h_com*(1+eps)*beta0/c_foot
            print(f"{eps:5.1f} {tw1*1000:4.0f}ms | {c['T_w2']*1000:6.1f}ms {c['t_zero']*1000:6.0f}ms | "
                  f"{c['r']:8.4f} {np.degrees(c['beta_f']):7.3f}d {c['phi_dot']:8.3f} | {note} "
                  f"(d={np.degrees(d_fold):.0f}d)")
            all_results.append((eps, tw1, c))
        else:
            print(f"{eps:5.1f} {tw1*1000:4.0f}ms | {'---':>7} {'---':>7} | {'---':>8} {'---':>8} {'---':>8} | no phi=0")

# ==============================
# 2. Best timeseries
# ==============================
conv = [(e, tw, c) for e, tw, c in all_results if abs(c['r']) < 1]
if conv:
    conv.sort(key=lambda x: abs(x[2]['r']))
    best_eps, best_tw1, best_c = conv[0]
    print(f"\n*** Best convergent: eps={best_eps}, Tw1={best_tw1*1000:.0f}ms, "
          f"r={best_c['r']:.4f}, Tw2={best_c['T_w2']*1000:.1f}ms ***")

    sol, t3, _ = run_full_cycle(beta0, best_eps, T_f, best_tw1)
    t_dense = np.linspace(0, best_c['t_zero'] + 0.02, 3000)
    phi_d = [np.degrees(sol.sol(t)[0]) for t in t_dense]
    beta_d = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_dense]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t1 = T_f; t2 = T_f+best_tw1; t_end = best_c['t_zero']
    phases = [(0,t1,'#FFD700','FOLD'),(t1,t2,'#90EE90','W1'),
              (t2,t3,'#87CEEB','EXT'),(t3,t_end,'#DDA0DD','W2')]
    for ax in axes:
        for a,b,c,l in phases:
            ax.axvspan(a*1000,b*1000,alpha=0.15,color=c)
            ax.text((a+b)/2*1000, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1]>0 else 3,
                   l, ha='center', fontsize=9, alpha=0.6)
        for tb in [t1,t2,t3]:
            ax.axvline(tb*1000, color='gray', ls=':', alpha=0.4)

    axes[0].plot(np.array(t_dense)*1000, phi_d, 'b-', lw=2)
    axes[0].axhline(0, color='red', ls='--', alpha=0.5)
    axes[0].axvline(t_end*1000, color='green', lw=2, alpha=0.7, label=f'phi=0 (Tw2={best_c["T_w2"]*1000:.1f}ms)')
    axes[0].set_ylabel('phi [deg]')
    axes[0].legend()
    axes[0].set_title(f'4-Phase BEST: eps={best_eps}, Tf={T_f*1000:.0f}ms, '
                      f'Tw1={best_tw1*1000:.0f}ms, Tw2={best_c["T_w2"]*1000:.1f}ms\n'
                      f'r={best_c["r"]:.4f}, beta: {np.degrees(beta0):.1f} -> {np.degrees(best_c["beta_f"]):.2f} deg')

    axes[1].plot(np.array(t_dense)*1000, beta_d, 'r-', lw=2)
    axes[1].axhline(np.degrees(beta0), color='gray', ls='--', alpha=0.5, label='beta0')
    axes[1].axhline(0, color='black', ls='-', lw=0.5)
    axes[1].axvline(t_end*1000, color='green', lw=2, alpha=0.7)
    axes[1].set_ylabel('beta [deg]')
    axes[1].set_xlabel('time [ms]')
    axes[1].legend()

    plt.tight_layout()
    out = __file__.replace('.py', '_best.png')
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()
else:
    print("\nNo convergent cases!")

# ==============================
# 3. r vs eps curve (fixed Tw1)
# ==============================
fig, ax = plt.subplots(figsize=(10, 6))
eps_sweep = np.linspace(0.8, 8.0, 60)

for tw1 in [0.001, 0.005, 0.01, 0.02, 0.05]:
    r_vals = []; eps_valid = []
    for eps in eps_sweep:
        _, _, crossings = run_full_cycle(beta0, eps, T_f, tw1)
        if crossings:
            r_vals.append(crossings[0]['r'])
            eps_valid.append(eps)
    if r_vals:
        ax.plot(eps_valid, r_vals, 'o-', ms=3, label=f'Tw1={tw1*1000:.0f}ms')

ax.axhline(1, color='red', ls='--', lw=2, label='r=1 (diverge)')
ax.axhline(0, color='gray', ls='--', alpha=0.3)
ax.axhline(-1, color='red', ls=':', alpha=0.5, label='r=-1')
ax.set_xlabel('epsilon')
ax.set_ylabel('r = beta_final / beta0')
ax.set_title(f'4-Phase convergence: r vs eps (Tf={T_f*1000:.0f}ms, beta0={np.degrees(beta0):.0f}deg)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-3, 3)
plt.tight_layout()
out2 = __file__.replace('.py', '_r_vs_eps.png')
plt.savefig(out2, dpi=150)
print(f"Saved: {out2}")
plt.close()

print("\nDone!")

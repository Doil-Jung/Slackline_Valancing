"""Find eps threshold where beta crosses zero at end of FOLD"""
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

print(f"c_foot = {c_foot:.4f}")
print(f"C_sd = {C_sd:.5f}")

def get_beta_at_fold_end(beta0, eps, T_f):
    """Return beta at end of FOLD phase"""
    sigma0 = Mt * h_com * beta0
    d_fold = h_com * (1 + eps) * abs(beta0) / c_foot * np.sign(beta0)
    a = 4 * d_fold / T_f**2
    
    def get_dd(t):
        if t < T_f/2: return +a
        elif t < T_f: return -a
        else: return 0.0
    def rhs(t, y):
        phi, dp, sig, ds = y
        dd = get_dd(t)
        r1 = -gMR*phi; r2 = g_ps*sig - C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]
    
    sol = solve_ivp(rhs, (0, T_f), [0, 0, sigma0, 0],
                    method='RK45', rtol=1e-12, atol=1e-14, max_step=T_f/100)
    beta_end = sol.y[2, -1] / (Mt * h_com)
    phi_end = sol.y[0, -1]
    return beta_end, phi_end, d_fold

beta0 = np.radians(5)

# ==============================
# 1. eps sweep: find beta=0 threshold
# ==============================
print(f"\n{'='*60}")
print(f"eps sweep: beta at end of FOLD (beta0=5deg)")
print(f"{'='*60}")

eps_range = np.concatenate([
    np.arange(0.1, 1.0, 0.1),
    np.arange(1.0, 5.0, 0.5),
    np.arange(5.0, 20.0, 1.0),
    np.arange(20, 51, 5)
])

for T_f in [0.03, 0.05, 0.08]:
    print(f"\n--- T_f = {T_f*1000:.0f}ms ---")
    print(f"{'eps':>6} {'delta_fold':>11} {'beta_end':>10} {'phi_end':>10}")
    prev_sign = 1
    for eps in eps_range:
        bf, pf, df = get_beta_at_fold_end(beta0, eps, T_f)
        curr_sign = 1 if bf > 0 else -1
        marker = " <== CROSSES ZERO" if curr_sign != prev_sign else ""
        print(f"{eps:6.1f} {np.degrees(df):10.1f}deg {np.degrees(bf):9.3f}deg {np.degrees(pf):9.3f}deg{marker}")
        prev_sign = curr_sign

# ==============================
# 2. Find exact threshold with brentq
# ==============================
print(f"\n{'='*60}")
print("Exact beta=0 threshold")
print(f"{'='*60}")

for T_f in [0.03, 0.05, 0.08]:
    try:
        eps_crit = brentq(lambda e: get_beta_at_fold_end(beta0, e, T_f)[0],
                          0.1, 50.0, xtol=1e-6)
        bf, pf, df = get_beta_at_fold_end(beta0, eps_crit, T_f)
        print(f"T_f={T_f*1000:.0f}ms: eps_crit = {eps_crit:.4f} "
              f"(delta_fold={np.degrees(df):.1f}deg, phi={np.degrees(pf):.2f}deg)")
    except:
        print(f"T_f={T_f*1000:.0f}ms: no crossing found in [0.1, 50]")

# ==============================
# 3. Quasi-static vs dynamic comparison
# ==============================
print(f"\n{'='*60}")
print("Quasi-static (5/12) vs Dynamic beta at fold end")
print(f"{'='*60}")

T_f = 0.05
print(f"{'eps':>6} | {'QS beta':>9} | {'Dyn beta':>9} | {'ratio':>6}")
for eps in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    qs_beta = -eps * beta0  # 5/12 model prediction
    dyn_beta, _, _ = get_beta_at_fold_end(beta0, eps, T_f)
    ratio = dyn_beta / beta0 if abs(beta0) > 1e-10 else 0
    print(f"{eps:6.1f} | {np.degrees(qs_beta):8.3f}d | {np.degrees(dyn_beta):8.3f}d | {ratio:6.3f}")

# ==============================
# 4. Plot: beta_end vs eps
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
eps_plot = np.linspace(0.1, 30, 200)

for T_f in [0.03, 0.05, 0.08]:
    betas = [np.degrees(get_beta_at_fold_end(beta0, e, T_f)[0]) for e in eps_plot]
    axes[0].plot(eps_plot, betas, '-', lw=2, label=f'T_f={T_f*1000:.0f}ms (dynamic)')

# Quasi-static
qs = [np.degrees(-e * beta0) for e in eps_plot]
axes[0].plot(eps_plot, qs, 'k--', lw=1, label='5/12 model (quasi-static)')
axes[0].axhline(0, color='red', ls='-', lw=1)
axes[0].set_xlabel('epsilon')
axes[0].set_ylabel('beta at end of FOLD [deg]')
axes[0].set_title('When does beta cross zero?')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-10, 6)

# delta_fold vs eps
for T_f in [0.03, 0.05, 0.08]:
    deltas = [np.degrees(h_com*(1+e)*beta0/c_foot) for e in eps_plot]
    axes[1].plot(eps_plot, deltas, '-', lw=2, label=f'T_f={T_f*1000:.0f}ms')
axes[1].axhline(90, color='red', ls='--', alpha=0.5, label='90deg (physical limit)')
axes[1].set_xlabel('epsilon')
axes[1].set_ylabel('delta_fold [deg]')
axes[1].set_title('Required fold angle')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

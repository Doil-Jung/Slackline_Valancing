"""
4-Phase FWE Model: F(T_f) - W1(T_w1) - E(T_e) - W2(until phi=0)
T_w2 is determined by phi's first zero-crossing after EXTEND.
Sweep over (eps, T_f, T_w1) and compute convergence ratio r.
"""
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

print(f"w_eff={w_eff:.4f} rad/s, lam_eff={lam_eff:.4f} rad/s")
print(f"T_quarter(phi)={T_quarter*1000:.1f} ms")
print(f"h={h_com:.4f} m, c_foot={c_foot:.4f}")


def solve_4phase(beta0, eps, T_f, T_w1, T_e=None):
    """4-phase cycle. Returns (T_w2, phi_at_zero, beta_final, r, total_time, sol)
    or None if no phi=0 crossing found in W2."""
    if T_e is None:
        T_e = T_f
    
    sigma0 = Mt * h_com * beta0
    d_fold = h_com * (1 + eps) * abs(beta0) / c_foot * np.sign(beta0)
    a_fold = 4 * d_fold / T_f**2
    a_ext = 4 * d_fold / T_e**2
    
    t1 = T_f
    t2 = T_f + T_w1
    t3 = T_f + T_w1 + T_e
    
    def get_dd(t):
        if t < T_f/2:           return +a_fold
        elif t < T_f:           return -a_fold
        elif t < t2:            return 0.0
        elif t < t2 + T_e/2:   return -a_ext
        elif t < t3:            return +a_ext
        else:                   return 0.0
    
    def rhs(t, y):
        phi, dp, sig, ds = y
        dd = get_dd(t)
        r1 = -gMR * phi
        r2 = g_ps * sig - C_sd * dd
        return [dp, iM11*r1 + iM12*r2, ds, iM12*r1 + iM22*r2]
    
    T_w2_max = 4 * T_quarter
    T_total_max = t3 + T_w2_max
    
    # Dense output for accurate zero-finding
    N_pts = max(2000, int(T_total_max / 0.0001))
    t_eval = np.linspace(0, T_total_max, N_pts)
    
    sol = solve_ivp(rhs, (0, T_total_max), [0, 0, sigma0, 0],
                    method='RK45', rtol=1e-10, atol=1e-12,
                    max_step=min(T_f, max(T_w1, 0.001))/20,
                    t_eval=t_eval)
    
    # Find phi=0 crossings in W2 region (t > t3 + small margin)
    w2_margin = max(T_f * 0.1, 0.005)  # skip initial transient after EXTEND
    phi_arr = sol.y[0]
    
    for k in range(len(sol.t) - 1):
        if sol.t[k] < t3 + w2_margin:
            continue
        if phi_arr[k] * phi_arr[k+1] < 0:  # sign change
            # Refine with brentq using dense output
            try:
                sol_dense = solve_ivp(rhs, (0, T_total_max), [0, 0, sigma0, 0],
                                      method='RK45', rtol=1e-10, atol=1e-12,
                                      max_step=min(T_f, max(T_w1, 0.001))/20,
                                      dense_output=True)
                t_zero = brentq(lambda t: sol_dense.sol(t)[0],
                                sol.t[k], sol.t[k+1], xtol=1e-10)
                yf = sol_dense.sol(t_zero)
                T_w2 = t_zero - t3
                beta_f = yf[2] / (Mt * h_com)
                r = beta_f / beta0
                return T_w2, yf[0], beta_f, r, t_zero, sol
            except:
                continue
    
    return None


# ==============================
# 1. Main sweep: eps x T_w1
# ==============================
print(f"\n{'='*70}")
print("4-Phase Sweep: eps x T_w1 (T_f=0.05s)")
print(f"{'='*70}")

beta0 = np.radians(5)
T_f = 0.05

print(f"\nbeta0={np.degrees(beta0):.1f}deg, T_f={T_f}s")
print(f"{'eps':>5} {'T_w1':>7} | {'T_w2':>7} {'T_tot':>7} | {'r':>8} {'beta_f':>8} | note")
print("-" * 75)

eps_vals = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
Tw1_vals = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

best_results = []

for eps in eps_vals:
    for T_w1 in Tw1_vals:
        result = solve_4phase(beta0, eps, T_f, T_w1)
        if result is not None:
            T_w2, phi_f, beta_f, r, T_tot, _ = result
            note = "CONV" if abs(r) < 1 else "DIV"
            if abs(r) < 0.01:
                note = "~ZERO"
            print(f"{eps:5.1f} {T_w1*1000:6.1f}ms | {T_w2*1000:6.1f}ms {T_tot*1000:6.0f}ms | "
                  f"{r:8.4f} {np.degrees(beta_f):7.3f}d | {note}")
            best_results.append((eps, T_w1, T_w2, T_tot, r, beta_f))
        else:
            print(f"{eps:5.1f} {T_w1*1000:6.1f}ms | {'---':>7} {'---':>7} | "
                  f"{'---':>8} {'---':>8} | no phi=0")


# ==============================
# 2. Heatmap: r vs (eps, T_w1)
# ==============================
print(f"\n{'='*70}")
print("Generating heatmap...")

eps_fine = np.linspace(0.2, 4.0, 40)
Tw1_fine = np.linspace(0.001, 0.10, 35)

R_map = np.full((len(Tw1_fine), len(eps_fine)), np.nan)
Tw2_map = np.full_like(R_map, np.nan)
Ttot_map = np.full_like(R_map, np.nan)

for i, eps in enumerate(eps_fine):
    for j, tw1 in enumerate(Tw1_fine):
        result = solve_4phase(beta0, eps, T_f, tw1)
        if result is not None:
            T_w2, _, _, r, T_tot, _ = result
            R_map[j, i] = r
            Tw2_map[j, i] = T_w2
            Ttot_map[j, i] = T_tot

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# r map
data = np.clip(R_map, -2, 2)
im = axes[0].contourf(eps_fine, Tw1_fine*1000, data, levels=np.linspace(-2, 2, 41), cmap='RdYlGn_r')
axes[0].contour(eps_fine, Tw1_fine*1000, R_map, levels=[0], colors='blue', linewidths=2)
axes[0].contour(eps_fine, Tw1_fine*1000, np.abs(R_map), levels=[1], colors='black', linewidths=2)
axes[0].set_xlabel('epsilon'); axes[0].set_ylabel('T_w1 [ms]')
axes[0].set_title(f'4-Phase: r (convergence ratio)\nT_f={T_f*1000:.0f}ms, beta0=5deg')
plt.colorbar(im, ax=axes[0], label='r = beta_f/beta0')

# T_w2 map
im2 = axes[1].contourf(eps_fine, Tw1_fine*1000, Tw2_map*1000, levels=20, cmap='viridis')
axes[1].set_xlabel('epsilon'); axes[1].set_ylabel('T_w1 [ms]')
axes[1].set_title('T_w2 (wait for phi=0) [ms]')
plt.colorbar(im2, ax=axes[1], label='T_w2 [ms]')

# Total time
im3 = axes[2].contourf(eps_fine, Tw1_fine*1000, Ttot_map*1000, levels=20, cmap='plasma')
axes[2].contour(eps_fine, Tw1_fine*1000, np.abs(R_map), levels=[1], colors='white', linewidths=2)
axes[2].set_xlabel('epsilon'); axes[2].set_ylabel('T_w1 [ms]')
axes[2].set_title('Total cycle time [ms]')
plt.colorbar(im3, ax=axes[2], label='T_total [ms]')

plt.tight_layout()
out = __file__.replace('.py', '_heatmap.png')
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close()


# ==============================
# 3. Best case timeseries
# ==============================
# Find best (smallest |r|) convergent case
conv_results = [(eps, tw1, tw2, tt, r, bf) for eps, tw1, tw2, tt, r, bf in best_results if abs(r) < 1]
if conv_results:
    conv_results.sort(key=lambda x: abs(x[4]))
    best = conv_results[0]
    print(f"\nBest convergent: eps={best[0]:.1f}, T_w1={best[1]*1000:.1f}ms, "
          f"T_w2={best[2]*1000:.1f}ms, r={best[4]:.4f}")
    
    # Re-solve for timeseries
    result = solve_4phase(beta0, best[0], T_f, best[1])
    if result:
        _, _, _, r, T_tot, sol = result
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        t_ms = sol.t * 1000
        phi_deg = np.degrees(sol.y[0])
        beta_deg = np.degrees(sol.y[2] / (Mt * h_com))
        
        # Phase boundaries
        t1 = T_f; t2 = T_f + best[1]; t3 = t2 + T_f
        phase_labels = ['FOLD', 'W1', 'EXTEND', 'W2']
        phase_bounds = [0, t1, t2, t3, T_tot]
        colors_phase = ['#FFD700', '#90EE90', '#87CEEB', '#DDA0DD']
        
        for ax in axes[:2]:
            for k in range(4):
                ax.axvspan(phase_bounds[k]*1000, phase_bounds[k+1]*1000,
                          alpha=0.15, color=colors_phase[k])
                mid = (phase_bounds[k] + phase_bounds[k+1]) / 2
                ax.text(mid*1000, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 5,
                       phase_labels[k], ha='center', fontsize=8, alpha=0.7)
        
        axes[0].plot(t_ms, phi_deg, 'b-', lw=2)
        axes[0].axhline(0, color='red', ls='--', alpha=0.5)
        axes[0].set_ylabel('phi [deg]')
        axes[0].set_title(f'4-Phase Best: eps={best[0]:.1f}, T_f={T_f*1000:.0f}ms, '
                          f'T_w1={best[1]*1000:.1f}ms, T_w2={best[2]*1000:.1f}ms, r={r:.4f}')
        
        axes[1].plot(t_ms, beta_deg, 'r-', lw=2)
        axes[1].axhline(np.degrees(beta0), color='gray', ls='--', alpha=0.5, label='beta0')
        axes[1].axhline(0, color='gray', ls='--', alpha=0.3)
        axes[1].set_ylabel('beta [deg]')
        axes[1].legend()
        
        # delta profile reconstruction
        d_fold = h_com * (1 + best[0]) * beta0 / c_foot
        a = 4 * d_fold / T_f**2
        delta_t = []
        for t in sol.t:
            if t < T_f/2: dd = a * t
            elif t < T_f: dd = a * T_f - a * t  # decel
            elif t < t2: dd = 0  # simplification, delta stays at d_fold
            elif t < t2 + T_f/2: dd = -(t - t2) * a  # extending back
            elif t < t3: dd = -a * T_f + a * (t - t2)
            else: dd = 0
            delta_t.append(dd)
        # Actually just compute delta position
        delta_pos = []
        for t in sol.t:
            if t < T_f/2:
                delta_pos.append(0.5 * a * t**2)
            elif t < T_f:
                dt2 = t - T_f/2
                delta_pos.append(0.5*a*(T_f/2)**2 + a*(T_f/2)*dt2 - 0.5*a*dt2**2)
            elif t < t2:
                delta_pos.append(d_fold)
            elif t < t2 + T_f/2:
                dt3 = t - t2
                delta_pos.append(d_fold - 0.5*a*dt3**2)
            elif t < t3:
                dt4 = t - (t2 + T_f/2)
                v_mid = a * T_f/2
                delta_pos.append(d_fold - 0.5*a*(T_f/2)**2 - v_mid*dt4 + 0.5*a*dt4**2)
            else:
                delta_pos.append(0.0)
        
        axes[2].plot(t_ms, np.degrees(delta_pos), 'g-', lw=2)
        axes[2].set_ylabel('delta [deg]')
        axes[2].set_xlabel('time [ms]')
        
        for ax in axes:
            for tb in phase_bounds[1:-1]:
                ax.axvline(tb*1000, color='gray', ls=':', alpha=0.4)
        
        # Add phase shading to delta plot too
        for k in range(4):
            axes[2].axvspan(phase_bounds[k]*1000, phase_bounds[k+1]*1000,
                           alpha=0.15, color=colors_phase[k])
        
        plt.tight_layout()
        ts_out = __file__.replace('.py', '_best_timeseries.png')
        plt.savefig(ts_out, dpi=150)
        print(f"Saved: {ts_out}")
        plt.close()
else:
    print("\nNo convergent cases found!")

# ==============================
# 4. r vs T_w1 curves for fixed eps
# ==============================
fig, ax = plt.subplots(figsize=(10, 6))
Tw1_sweep = np.linspace(0.001, 0.15, 60)

for eps in [0.5, 1.0, 1.5, 2.0, 3.0]:
    r_vals = []
    tw1_valid = []
    for tw1 in Tw1_sweep:
        res = solve_4phase(beta0, eps, T_f, tw1)
        if res is not None:
            r_vals.append(res[3])
            tw1_valid.append(tw1)
    if r_vals:
        ax.plot(np.array(tw1_valid)*1000, r_vals, 'o-', ms=3, label=f'eps={eps}')

ax.axhline(1, color='red', ls='--', label='r=1')
ax.axhline(0, color='gray', ls='--', alpha=0.3)
ax.axhline(-1, color='red', ls=':', alpha=0.5)
ax.set_xlabel('T_w1 [ms]')
ax.set_ylabel('r = beta_final / beta0')
ax.set_title(f'4-Phase: Convergence ratio vs T_w1\nT_f={T_f*1000:.0f}ms, beta0=5deg')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-3, 3)
plt.tight_layout()
curve_out = __file__.replace('.py', '_r_vs_tw1.png')
plt.savefig(curve_out, dpi=150)
print(f"Saved: {curve_out}")
plt.close()

print("\nDone!")

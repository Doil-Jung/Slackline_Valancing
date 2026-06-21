"""Fine search around convergent region + best trajectory"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

g = 9.81; R_rope = 1.0
M1, m2 = 30.0, 45.0; L1, LC1, LC2 = 0.85, 0.30, 0.40; L_upper = 0.85
Mt = M1+m2; I1=M1*L1**2/12; I2=m2*L_upper**2/12
p1=M1*LC1+m2*L1; p2=m2*LC2; ps=p1+p2
h_com=(M1*LC1+m2*(L1+LC2))/Mt; c_foot=p2/ps
J_aa=M1*LC1**2+I1+m2*L1**2; J_tt=m2*LC2**2+I2
J_at=m2*L1*LC2; J_tot=J_aa+J_tt+2*J_at
C_sd=(-J_aa*p2+J_tt*p1+J_at*(p1-p2))/ps**2
M11=Mt*R_rope**2; M12=R_rope; M22=J_tot/ps**2
det_M=M11*M22-M12**2
iM11=M22/det_M; iM12=-M12/det_M; iM22=M11/det_M
gMR=g*Mt*R_rope; g_ps=g/ps
w_eff=np.sqrt(iM11*gMR); lam_eff=np.sqrt(iM22*g_ps)
T_quarter=np.pi/(2*w_eff)

def run_cycle(beta0, eps, T_f, T_w1):
    sigma0 = Mt*h_com*beta0
    d_fold = h_com*(1+eps)*abs(beta0)/c_foot*np.sign(beta0)
    a = 4*d_fold/T_f**2
    t2 = T_f+T_w1; t3 = t2+T_f
    T_end = t3 + 4*T_quarter

    def get_dd(t):
        if t < T_f/2: return +a
        elif t < T_f: return -a
        elif t < t2: return 0.0
        elif t < t2+T_f/2: return -a
        elif t < t3: return +a
        else: return 0.0

    def rhs(t, y):
        phi,dp,sig,ds = y
        dd = get_dd(t)
        r1=-gMR*phi; r2=g_ps*sig-C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

    sol = solve_ivp(rhs, (0, T_end), [0,0,sigma0,0],
                    method='RK45', rtol=1e-10, atol=1e-12,
                    max_step=T_f/40, dense_output=True)

    t_w2 = np.linspace(t3+0.003, T_end, 5000)
    phi_w2 = np.array([sol.sol(t)[0] for t in t_w2])
    crossings = []
    for k in range(len(phi_w2)-1):
        if phi_w2[k]*phi_w2[k+1] < 0:
            try:
                tz = brentq(lambda t: sol.sol(t)[0], t_w2[k], t_w2[k+1], xtol=1e-10)
                yf = sol.sol(tz)
                bf = yf[2]/(Mt*h_com)
                crossings.append({'T_w2':tz-t3,'t':tz,'beta_f':bf,'r':bf/beta0})
            except: pass
    return sol, t3, crossings

beta0 = np.radians(5)
T_f = 0.05

# ==============================
# 1. Fine grid: eps x T_w1
# ==============================
print(f"{'='*70}")
print(f"Fine search: eps 1-10, T_w1 50-500ms")
print(f"{'='*70}")

eps_fine = np.arange(1.0, 10.5, 0.5)
tw1_fine = np.arange(0.05, 0.55, 0.02)

# Store results in grid
r_grid = np.full((len(eps_fine), len(tw1_fine)), np.nan)
conv_cases = []

for i, eps in enumerate(eps_fine):
    for j, tw1 in enumerate(tw1_fine):
        _, _, crossings = run_cycle(beta0, eps, T_f, tw1)
        if crossings:
            r = crossings[0]['r']
            r_grid[i, j] = r
            if abs(r) < 1:
                conv_cases.append((eps, tw1, crossings[0]))

print(f"\nFound {len(conv_cases)} convergent cases!")
if conv_cases:
    conv_cases.sort(key=lambda x: abs(x[2]['r']))
    print(f"\n{'eps':>5} {'Tw1':>6} | {'Tw2':>7} {'Ttot':>7} | {'r':>8} {'d_fold':>7}")
    print("-" * 60)
    for e, tw, c in conv_cases[:20]:
        d_fold = np.degrees(h_com*(1+e)*beta0/c_foot)
        ttot = c['t']
        print(f"{e:5.1f} {tw*1000:5.0f}ms | {c['T_w2']*1000:6.1f}ms {ttot*1000:6.0f}ms | "
              f"{c['r']:8.4f} {d_fold:6.1f}d")

# ==============================
# 2. Heatmap
# ==============================
fig, ax = plt.subplots(figsize=(14, 8))
# Clip r to [-3, 3] for visualization
r_plot = np.clip(r_grid, -3, 3)
im = ax.pcolormesh(tw1_fine*1000, eps_fine, r_plot, cmap='RdYlGn_r',
                   vmin=-2, vmax=2, shading='auto')
plt.colorbar(im, ax=ax, label='r = beta_f / beta0')

# Mark |r|<1 region
for e, tw, c in conv_cases:
    ax.plot(tw*1000, e, 'k*', ms=8, alpha=0.7)

# Contour at r=0
try:
    cs = ax.contour(tw1_fine*1000, eps_fine, r_grid, levels=[0], colors='white', linewidths=2)
    ax.clabel(cs, fmt='r=0')
except: pass
try:
    cs2 = ax.contour(tw1_fine*1000, eps_fine, np.abs(r_grid), levels=[1], colors='black', linewidths=2, linestyles='--')
    ax.clabel(cs2, fmt='|r|=1')
except: pass

ax.set_xlabel('T_w1 [ms]')
ax.set_ylabel('epsilon')
ax.set_title(f'4-Phase convergence map (T_f={T_f*1000:.0f}ms, beta0={np.degrees(beta0):.0f}deg)\n'
             f'Stars = convergent cases (|r|<1)')
plt.tight_layout()
out1 = __file__.replace('.py', '_heatmap.png')
plt.savefig(out1, dpi=150)
print(f"\nSaved: {out1}")
plt.close()

# ==============================
# 3. Best trajectory
# ==============================
if conv_cases:
    best_eps, best_tw1, best_c = conv_cases[0]
    print(f"\n*** BEST: eps={best_eps}, Tw1={best_tw1*1000:.0f}ms, "
          f"r={best_c['r']:.4f}, Tw2={best_c['T_w2']*1000:.1f}ms ***")

    sol, t3, _ = run_cycle(beta0, best_eps, T_f, best_tw1)
    t2 = T_f + best_tw1
    T_end = best_c['t'] + 0.05
    t_plot = np.linspace(0, T_end, 3000)
    phi_d = [np.degrees(sol.sol(t)[0]) for t in t_plot]
    beta_d = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_plot]

    # delta position
    d_fold = h_com*(1+best_eps)*beta0/c_foot
    a = 4*d_fold/T_f**2
    delta_d = []
    for t in t_plot:
        if t < T_f/2: delta_d.append(0.5*a*t**2)
        elif t < T_f:
            dt=t-T_f/2; v=a*T_f/2
            delta_d.append(0.5*a*(T_f/2)**2+v*dt-0.5*a*dt**2)
        elif t < t2: delta_d.append(d_fold)
        elif t < t2+T_f/2:
            dt=t-t2
            delta_d.append(d_fold-0.5*a*dt**2)
        elif t < t3:
            dt=t-(t2+T_f/2); v=a*T_f/2
            delta_d.append(d_fold-0.5*a*(T_f/2)**2-v*dt+0.5*a*dt**2)
        else: delta_d.append(0.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    phases = [(0,T_f,'#FFD700','FOLD'),(T_f,t2,'#90EE90','WAIT1'),
              (t2,t3,'#87CEEB','EXTEND'),(t3,T_end,'#DDA0DD','WAIT2')]
    for ax in axes:
        for a_,b_,c_,l_ in phases:
            ax.axvspan(a_*1000,min(b_,T_end)*1000,alpha=0.15,color=c_)
        for tb in [T_f,t2,t3]:
            ax.axvline(tb*1000,color='gray',ls=':',alpha=0.4)

    axes[0].plot(np.array(t_plot)*1000, phi_d, 'b-', lw=2)
    axes[0].axhline(0, color='red', ls='--', alpha=0.5)
    axes[0].axvline(best_c['t']*1000, color='green', lw=2, ls='-', label=f'phi=0 @ {best_c["t"]*1000:.0f}ms')
    axes[0].set_ylabel('phi [deg]', fontsize=12)
    axes[0].legend(fontsize=11)

    axes[1].plot(np.array(t_plot)*1000, beta_d, 'r-', lw=2)
    axes[1].axhline(0, color='black', ls='-', lw=0.5)
    axes[1].axhline(np.degrees(beta0), color='gray', ls='--', alpha=0.5, label=f'beta0={np.degrees(beta0):.0f}deg')
    bf_deg = np.degrees(best_c['beta_f'])
    axes[1].axhline(bf_deg, color='green', ls='--', alpha=0.5, label=f'beta_f={bf_deg:.2f}deg')
    axes[1].axvline(best_c['t']*1000, color='green', lw=2, ls='-')
    axes[1].set_ylabel('beta [deg]', fontsize=12)
    axes[1].legend(fontsize=11)

    axes[2].plot(np.array(t_plot)*1000, np.degrees(delta_d), 'g-', lw=2)
    axes[2].set_ylabel('delta (fold) [deg]', fontsize=12)
    axes[2].set_xlabel('time [ms]', fontsize=12)

    d_deg = np.degrees(d_fold)
    axes[0].set_title(f'BEST 4-Phase: eps={best_eps}, Tf={T_f*1000:.0f}ms, '
                      f'Tw1={best_tw1*1000:.0f}ms, Tw2={best_c["T_w2"]*1000:.0f}ms\n'
                      f'r={best_c["r"]:.4f} | delta_fold={d_deg:.1f}deg | '
                      f'beta: {np.degrees(beta0):.1f} -> {bf_deg:.2f} deg | '
                      f'T_total={best_c["t"]*1000:.0f}ms', fontsize=11)

    plt.tight_layout()
    out2 = __file__.replace('.py', '_best.png')
    plt.savefig(out2, dpi=150)
    print(f"Saved: {out2}")
    plt.close()

print("\nDone!")

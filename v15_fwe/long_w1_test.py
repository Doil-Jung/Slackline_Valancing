"""4-Phase with LONG W1: let beta diverge negative before extending"""
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

    # phi=0 crossings in W2
    t_w2 = np.linspace(t3+0.003, T_end, 5000)
    phi_w2 = np.array([sol.sol(t)[0] for t in t_w2])
    crossings = []
    for k in range(len(phi_w2)-1):
        if phi_w2[k]*phi_w2[k+1] < 0:
            try:
                tz = brentq(lambda t: sol.sol(t)[0], t_w2[k], t_w2[k+1], xtol=1e-10)
                yf = sol.sol(tz)
                bf = yf[2]/(Mt*h_com)
                crossings.append({'T_w2':tz-t3, 't':tz, 'beta_f':bf, 'r':bf/beta0})
            except: pass
    return sol, t3, crossings

beta0 = np.radians(5)
T_f = 0.05

# ==============================
# 1. Sweep: large T_w1 (50ms to 500ms)
# ==============================
print(f"{'='*70}")
print(f"4-Phase with LONG W1 (eps >= 1.0, T_w1 up to 500ms)")
print(f"{'='*70}")

eps_vals = [1.0, 1.5, 2.0, 3.0, 5.0]
tw1_vals = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

print(f"{'eps':>5} {'Tw1':>6} | {'Tw2':>7} {'Ttot':>7} | {'r':>8} {'beta_f':>8} | note")
print("-" * 70)

results = []
for eps in eps_vals:
    for tw1 in tw1_vals:
        sol, t3, crossings = run_cycle(beta0, eps, T_f, tw1)
        if crossings:
            c = crossings[0]
            note = "CONV" if abs(c['r']) < 1 else "DIV"
            print(f"{eps:5.1f} {tw1*1000:5.0f}ms | {c['T_w2']*1000:6.1f}ms {c['t']*1000:6.0f}ms | "
                  f"{c['r']:8.4f} {np.degrees(c['beta_f']):7.3f}d | {note}")
            results.append((eps, tw1, c))
        else:
            # Check beta at fold end and W1 end
            bf_fold = np.degrees(sol.sol(T_f)[2]/(Mt*h_com))
            bf_w1 = np.degrees(sol.sol(T_f+tw1)[2]/(Mt*h_com))
            print(f"{eps:5.1f} {tw1*1000:5.0f}ms | {'---':>7} {'---':>7} | {'---':>8} {'---':>8} | "
                  f"no phi=0 (b@F={bf_fold:.1f} b@W1={bf_w1:.1f})")

# ==============================
# 2. Trajectory plots for key cases
# ==============================
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
test_cases = [(1.0, 0.05), (1.0, 0.10), (1.0, 0.20),
              (2.0, 0.05), (2.0, 0.10), (2.0, 0.20),
              (3.0, 0.05), (3.0, 0.10), (3.0, 0.20)]

for idx, (eps, tw1) in enumerate(test_cases):
    sol, t3, crossings = run_cycle(beta0, eps, T_f, tw1)
    t2 = T_f+tw1
    T_end = t3 + 2*T_quarter
    t_plot = np.linspace(0, T_end, 2000)
    phi_p = [np.degrees(sol.sol(t)[0]) for t in t_plot]
    beta_p = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_plot]

    ax = axes.flat[idx]
    # Phase shading
    phases = [(0,T_f,'#FFD700','F'),(T_f,t2,'#90EE90','W1'),
              (t2,t3,'#87CEEB','E'),(t3,T_end,'#DDA0DD','W2')]
    for a,b,c,l in phases:
        ax.axvspan(a*1000,min(b,T_end)*1000,alpha=0.15,color=c)

    ax.plot(np.array(t_plot)*1000, phi_p, 'b-', lw=1.5)
    ax.axhline(0, color='red', ls='--', alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(np.array(t_plot)*1000, beta_p, 'r-', lw=1, alpha=0.5)
    ax2.axhline(0, color='black', ls='-', lw=0.5)

    # Mark crossings
    for c in crossings:
        ax.axvline(c['t']*1000, color='green', lw=2, alpha=0.7)

    bf_fold = np.degrees(sol.sol(T_f)[2]/(Mt*h_com))
    bf_w1 = np.degrees(sol.sol(t2)[2]/(Mt*h_com))
    status = f"{len(crossings)} zeros" if crossings else "no zero"
    if crossings:
        status += f", r={crossings[0]['r']:.3f}"
    ax.set_title(f'eps={eps}, Tw1={tw1*1000:.0f}ms\n'
                 f'b@F={bf_fold:.1f} b@W1={bf_w1:.1f} | {status}', fontsize=9)
    ax.set_xlabel('ms', fontsize=7)
    ax.set_ylabel('phi', fontsize=7)
    ax2.set_ylabel('beta', color='red', fontsize=7)

plt.suptitle(f'Long W1 test (T_f={T_f*1000:.0f}ms)', fontsize=12)
plt.tight_layout()
out = __file__.replace('.py', '_traj.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")
plt.close()

print("\nDone!")

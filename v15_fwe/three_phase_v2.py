"""3-Phase model with correct eps: find T_w where phi=0 at END of EXTEND"""
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

def run_3phase(beta0, eps, T_f, T_w):
    """Run F-W-E, return phi and beta at END of EXTEND"""
    sigma0 = Mt*h_com*beta0
    d_fold = h_com*(1+eps)*abs(beta0)/c_foot*np.sign(beta0)
    a = 4*d_fold/T_f**2
    t2 = T_f + T_w       # EXTEND start
    t3 = t2 + T_f         # EXTEND end = cycle end

    def get_dd(t):
        if t < T_f/2: return +a       # FOLD accel
        elif t < T_f: return -a        # FOLD decel
        elif t < t2: return 0.0        # WAIT
        elif t < t2+T_f/2: return -a   # EXTEND accel (reverse)
        elif t < t3: return +a         # EXTEND decel (reverse)
        else: return 0.0

    def rhs(t, y):
        phi,dp,sig,ds = y
        dd = get_dd(t)
        r1 = -gMR*phi; r2 = g_ps*sig - C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

    sol = solve_ivp(rhs, (0, t3+0.001), [0,0,sigma0,0],
                    method='RK45', rtol=1e-12, atol=1e-14,
                    max_step=T_f/50, dense_output=True)

    yf = sol.sol(t3)
    phi_end = yf[0]
    beta_end = yf[2]/(Mt*h_com)
    return phi_end, beta_end, sol, t3

beta0 = np.radians(5)

# ==============================
# 1. phi at EXTEND end vs T_w (find zeros)
# ==============================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

results_3p = []
for idx, (eps, T_f) in enumerate([(4, 0.05), (4, 0.10), (4, 0.15),
                                   (5, 0.05), (5, 0.10), (5, 0.15)]):
    ax = axes.flat[idx]

    tw_range = np.linspace(0.01, 2.0, 500)
    phi_ends = []
    beta_ends = []
    for tw in tw_range:
        pe, be, _, _ = run_3phase(beta0, eps, T_f, tw)
        phi_ends.append(np.degrees(pe))
        beta_ends.append(np.degrees(be))

    ax.plot(tw_range*1000, phi_ends, 'b-', lw=2, label='phi @ extend end')
    ax.axhline(0, color='red', ls='--', lw=1)

    ax2 = ax.twinx()
    ax2.plot(tw_range*1000, beta_ends, 'r-', lw=1, alpha=0.4, label='beta')
    ax2.set_ylabel('beta [deg]', color='red', fontsize=8)

    # Find phi=0 crossings
    zeros_tw = []
    for k in range(len(phi_ends)-1):
        if phi_ends[k]*phi_ends[k+1] < 0:
            # fine search
            tw_zero = brentq(lambda tw: run_3phase(beta0, eps, T_f, tw)[0],
                            tw_range[k], tw_range[k+1], xtol=1e-8)
            pe, be, _, _ = run_3phase(beta0, eps, T_f, tw_zero)
            r = be/beta0
            zeros_tw.append((tw_zero, r, be))
            ax.axvline(tw_zero*1000, color='green', lw=2, alpha=0.7)
            ax.text(tw_zero*1000+10, max(phi_ends)*0.5,
                   f'Tw={tw_zero*1000:.0f}ms\nr={r:.3f}', fontsize=7, color='green')
            results_3p.append({'eps':eps, 'T_f':T_f, 'T_w':tw_zero,
                              'r':r, 'beta_f':be, 'T_tot':2*T_f+tw_zero,
                              'd_fold':np.degrees(h_com*(1+eps)*beta0/c_foot)})

    ax.set_title(f'eps={eps}, Tf={T_f*1000:.0f}ms | {len(zeros_tw)} zeros', fontsize=10)
    ax.set_xlabel('T_w [ms]')
    ax.set_ylabel('phi @ extend end [deg]', fontsize=8)

plt.suptitle('3-Phase (F-W-E): phi at end of EXTEND vs T_w', fontsize=13)
plt.tight_layout()
out1 = __file__.replace('.py', '_phi_vs_tw.png')
plt.savefig(out1, dpi=150)
print(f"Saved: {out1}")
plt.close()

# ==============================
# 2. Print results
# ==============================
print(f"\n{'='*70}")
print("3-Phase convergent cases (phi=0 at extend end)")
print(f"{'='*70}")

if results_3p:
    conv3 = [r for r in results_3p if abs(r['r']) < 1]
    conv3.sort(key=lambda x: abs(x['r']))
    print(f"\n{'eps':>5} {'Tf':>5} {'Tw':>7} | {'Ttot':>6} | {'r':>8} | {'dfold':>6}")
    print("-" * 55)
    for c in conv3:
        print(f"{c['eps']:5.1f} {c['T_f']*1000:4.0f}ms {c['T_w']*1000:6.0f}ms | "
              f"{c['T_tot']:.2f}s | {c['r']:8.4f} | {c['d_fold']:5.0f}d")

    # Best trajectory
    if conv3:
        best = conv3[0]
        pe, be, sol, t3 = run_3phase(beta0, best['eps'], best['T_f'], best['T_w'])
        Tf = best['T_f']; Tw = best['T_w']
        t2 = Tf + Tw
        T_end = t3 + 0.05
        t_plot = np.linspace(0, T_end, 3000)
        phi_d = [np.degrees(sol.sol(t)[0]) for t in t_plot]
        beta_d = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_plot]

        d_fold = h_com*(1+best['eps'])*beta0/c_foot
        a = 4*d_fold/Tf**2
        delta_d = []
        for t in t_plot:
            if t < Tf/2: delta_d.append(0.5*a*t**2)
            elif t < Tf:
                dt=t-Tf/2; v=a*Tf/2
                delta_d.append(0.5*a*(Tf/2)**2+v*dt-0.5*a*dt**2)
            elif t < t2: delta_d.append(d_fold)
            elif t < t2+Tf/2:
                dt=t-t2
                delta_d.append(d_fold-0.5*a*dt**2)
            elif t < t3:
                dt=t-(t2+Tf/2); v=a*Tf/2
                delta_d.append(d_fold-0.5*a*(Tf/2)**2-v*dt+0.5*a*dt**2)
            else: delta_d.append(0.0)

        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        phases = [(0,Tf,'#FFD700','FOLD'),(Tf,t2,'#90EE90','WAIT'),
                  (t2,t3,'#87CEEB','EXTEND')]
        for ax in axes:
            for a_,b_,c_,l_ in phases:
                ax.axvspan(a_*1000,min(b_,T_end)*1000,alpha=0.15,color=c_)
            for tb in [Tf,t2,t3]:
                ax.axvline(tb*1000,color='gray',ls=':',alpha=0.4)

        axes[0].plot(np.array(t_plot)*1000, phi_d, 'b-', lw=2)
        axes[0].axhline(0, color='red', ls='--', alpha=0.5)
        axes[0].axvline(t3*1000, color='green', lw=2, label=f'EXTEND end: phi={np.degrees(pe):.3f}d')
        axes[0].set_ylabel('phi [deg]', fontsize=12)
        axes[0].legend(fontsize=11)

        axes[1].plot(np.array(t_plot)*1000, beta_d, 'r-', lw=2)
        axes[1].axhline(0, color='black', ls='-', lw=0.5)
        axes[1].axhline(5, color='gray', ls='--', alpha=0.3, label='beta0=5d')
        axes[1].axvline(t3*1000, color='green', lw=2)
        bf_d = np.degrees(be)
        axes[1].axhline(bf_d, color='green', ls='--', alpha=0.5, label=f'beta_f={bf_d:.2f}d')
        axes[1].set_ylabel('beta [deg]', fontsize=12)
        axes[1].legend(fontsize=11)

        axes[2].plot(np.array(t_plot)*1000, np.degrees(delta_d), 'g-', lw=2)
        axes[2].set_ylabel('delta [deg]', fontsize=12)
        axes[2].set_xlabel('time [ms]', fontsize=12)

        axes[0].set_title(
            f'BEST 3-Phase: eps={best["eps"]}, Tf={Tf*1000:.0f}ms, '
            f'Tw={Tw*1000:.0f}ms\n'
            f'r={best["r"]:.4f} | delta={best["d_fold"]:.0f}d | '
            f'T_total={best["T_tot"]:.2f}s | '
            f'beta: 5.0 -> {bf_d:.2f}d', fontsize=11)
        plt.tight_layout()
        out2 = __file__.replace('.py', '_best.png')
        plt.savefig(out2, dpi=150)
        print(f"\nSaved: {out2}")
else:
    print("No convergent cases found!")

print("\nDone!")

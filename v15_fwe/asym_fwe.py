"""Asymmetric FWE: fast FOLD + slow EXTEND"""
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

def run_asym(beta0, eps, T_fold, T_w, T_extend):
    """
    Asymmetric 3-phase: fast FOLD, slow EXTEND
    FOLD:   triangle accel profile over T_fold
    WAIT:   T_w
    EXTEND: triangle accel profile over T_extend (same displacement, slower)
    """
    sigma0 = Mt*h_com*beta0
    d_fold = h_com*(1+eps)*abs(beta0)/c_foot*np.sign(beta0)

    a_fold = 4*d_fold/T_fold**2        # fold acceleration
    a_ext  = 4*d_fold/T_extend**2      # extend acceleration (smaller if T_ext > T_fold)

    t1 = T_fold                        # fold end
    t2 = t1 + T_w                      # extend start
    t3 = t2 + T_extend                 # extend end
    T_end = t3 + 0.001

    def get_dd(t):
        # FOLD: triangle accel
        if t < T_fold/2:     return +a_fold
        elif t < T_fold:     return -a_fold
        # WAIT
        elif t < t2:         return 0.0
        # EXTEND: reverse triangle accel
        elif t < t2+T_extend/2: return -a_ext
        elif t < t3:            return +a_ext
        else:                   return 0.0

    def rhs(t, y):
        phi,dp,sig,ds = y
        dd = get_dd(t)
        r1=-gMR*phi; r2=g_ps*sig-C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

    dt_max = min(T_fold, T_extend)/50
    sol = solve_ivp(rhs, (0, T_end), [0,0,sigma0,0],
                    method='RK45', rtol=1e-12, atol=1e-14,
                    max_step=dt_max, dense_output=True)

    yf = sol.sol(t3)
    phi_end = yf[0]
    beta_end = yf[2]/(Mt*h_com)
    return phi_end, beta_end, sol, t3

beta0 = np.radians(5)

# ==============================
# Search: fast fold, slow extend
# ==============================
eps_vals = [3, 4, 5, 6]
tf_vals = [0.05, 0.08, 0.10]           # fast fold
te_vals = [0.20, 0.30, 0.40, 0.50, 0.70, 1.00]  # slow extend
tw_range = np.linspace(0.01, 1.5, 400)  # wait time scan

print(f"Asymmetric FWE: fast fold + slow extend")
print(f"{'='*80}")

all_conv = []

for eps in eps_vals:
    for T_fold in tf_vals:
        for T_ext in te_vals:
            d_fold_deg = np.degrees(h_com*(1+eps)*beta0/c_foot)
            v_fold = 2*d_fold_deg/(T_fold)    # deg/s
            v_ext  = 2*d_fold_deg/(T_ext)

            # Scan T_w for phi=0 at extend end
            phi_scan = []
            for tw in tw_range:
                pe, be, _, _ = run_asym(beta0, eps, T_fold, tw, T_ext)
                phi_scan.append(pe)

            # Find zeros
            for k in range(len(phi_scan)-1):
                if phi_scan[k]*phi_scan[k+1] < 0:
                    try:
                        tw_z = brentq(lambda tw: run_asym(beta0, eps, T_fold, tw, T_ext)[0],
                                     tw_range[k], tw_range[k+1], xtol=1e-8)
                        pe, be, _, _ = run_asym(beta0, eps, T_fold, tw_z, T_ext)
                        r = be/beta0
                        if abs(r) < 1:
                            ttot = T_fold+tw_z+T_ext
                            all_conv.append({
                                'eps':eps,'T_fold':T_fold,'T_w':tw_z,'T_ext':T_ext,
                                'r':r,'beta_f':be,'T_tot':ttot,
                                'd_fold':d_fold_deg,'v_fold':v_fold,'v_ext':v_ext
                            })
                    except: pass

print(f"\nFound {len(all_conv)} convergent cases!")

if all_conv:
    all_conv.sort(key=lambda x: abs(x['r']))
    print(f"\n{'eps':>4} {'Tf':>5} {'Tw':>6} {'Te':>5} | {'Ttot':>5} | "
          f"{'r':>8} | {'d':>4} {'vf':>6} {'ve':>6}")
    print("-" * 75)
    for c in all_conv[:25]:
        print(f"{c['eps']:4} {c['T_fold']*1000:4.0f}ms {c['T_w']*1000:5.0f}ms "
              f"{c['T_ext']*1000:4.0f}ms | {c['T_tot']:.2f}s | "
              f"{c['r']:8.4f} | {c['d_fold']:3.0f}d {c['v_fold']:5.0f}d/s {c['v_ext']:5.0f}d/s")

    # Best trajectory
    best = all_conv[0]
    pe, be, sol, t3 = run_asym(beta0, best['eps'], best['T_fold'], best['T_w'], best['T_ext'])
    Tf = best['T_fold']; Tw = best['T_w']; Te = best['T_ext']
    t1=Tf; t2=Tf+Tw

    T_end = t3+0.05
    t_plot = np.linspace(0, T_end, 3000)
    phi_d = [np.degrees(sol.sol(t)[0]) for t in t_plot]
    beta_d = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_plot]

    d_fold = h_com*(1+best['eps'])*beta0/c_foot
    af = 4*d_fold/Tf**2; ae = 4*d_fold/Te**2
    delta_d = []
    for t in t_plot:
        if t<Tf/2: delta_d.append(0.5*af*t**2)
        elif t<Tf:
            dt=t-Tf/2; delta_d.append(0.5*af*(Tf/2)**2+af*Tf/2*dt-0.5*af*dt**2)
        elif t<t2: delta_d.append(d_fold)
        elif t<t2+Te/2:
            dt=t-t2; delta_d.append(d_fold-0.5*ae*dt**2)
        elif t<t3:
            dt=t-(t2+Te/2); delta_d.append(d_fold-0.5*ae*(Te/2)**2-ae*Te/2*dt+0.5*ae*dt**2)
        else: delta_d.append(0.0)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    phases = [(0,t1,'#FFD700','FOLD'),(t1,t2,'#90EE90','WAIT'),
              (t2,t3,'#87CEEB','EXTEND')]
    for ax in axes:
        for a_,b_,c_,l_ in phases:
            ax.axvspan(a_*1000,min(b_,T_end)*1000,alpha=0.15,color=c_)
        for tb in [t1,t2,t3]:
            ax.axvline(tb*1000,color='gray',ls=':',alpha=0.4)

    axes[0].plot(np.array(t_plot)*1000, phi_d, 'b-', lw=2)
    axes[0].axhline(0, color='red', ls='--', alpha=0.5)
    axes[0].axvline(t3*1000, color='green', lw=2, label=f'EXTEND end: phi≈0')
    axes[0].set_ylabel('phi [deg]', fontsize=12)
    axes[0].legend(fontsize=11)

    axes[1].plot(np.array(t_plot)*1000, beta_d, 'r-', lw=2)
    axes[1].axhline(0, color='black', ls='-', lw=0.5)
    axes[1].axhline(5, color='gray', ls='--', alpha=0.3, label='beta0=5d')
    bf_d = np.degrees(be)
    axes[1].axhline(bf_d, color='green', ls='--', alpha=0.5, label=f'beta_f={bf_d:.2f}d')
    axes[1].axvline(t3*1000, color='green', lw=2)
    axes[1].set_ylabel('beta [deg]', fontsize=12)
    axes[1].legend(fontsize=11)

    axes[2].plot(np.array(t_plot)*1000, np.degrees(delta_d), 'g-', lw=2)
    axes[2].set_ylabel('delta [deg]', fontsize=12)
    axes[2].set_xlabel('time [ms]', fontsize=12)

    axes[0].set_title(
        f'Asymmetric FWE: eps={best["eps"]}, Tf={Tf*1000:.0f}ms, Tw={Tw*1000:.0f}ms, '
        f'Te={Te*1000:.0f}ms\n'
        f'r={best["r"]:.4f} | delta={best["d_fold"]:.0f}d | '
        f'v_fold={best["v_fold"]:.0f}d/s, v_ext={best["v_ext"]:.0f}d/s | '
        f'T_total={best["T_tot"]:.2f}s', fontsize=11)
    plt.tight_layout()
    out = __file__.replace('.py', '_best.png')
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")

print("\nDone!")

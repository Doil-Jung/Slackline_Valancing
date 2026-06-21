"""Realistic servo search: eps 2-6, T_f 200-500ms, T_w1 100-1000ms"""
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

SERVO_MAX_VEL = 360  # deg/s for STS3215

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

    t_w2 = np.linspace(t3+0.005, T_end, 5000)
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
    return sol, t3, crossings, d_fold

beta0 = np.radians(5)

# ==============================
# Grid search
# ==============================
eps_vals = np.arange(2.0, 7.5, 0.5)
tf_vals = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
tw1_vals = np.arange(0.10, 1.05, 0.05)

print(f"Search grid: {len(eps_vals)} eps x {len(tf_vals)} Tf x {len(tw1_vals)} Tw1 "
      f"= {len(eps_vals)*len(tf_vals)*len(tw1_vals)} combos")

all_conv = []
count = 0
total = len(eps_vals)*len(tf_vals)*len(tw1_vals)

for eps in eps_vals:
    for T_f in tf_vals:
        d_fold = h_com*(1+eps)*beta0/c_foot
        d_fold_deg = np.degrees(d_fold)
        peak_vel = np.degrees(2*d_fold/T_f)  # deg/s

        if peak_vel > SERVO_MAX_VEL:
            count += len(tw1_vals)
            continue  # skip servo-impossible cases

        for tw1 in tw1_vals:
            count += 1
            _, _, crossings, _ = run_cycle(beta0, eps, T_f, tw1)
            if crossings:
                c = crossings[0]
                if abs(c['r']) < 1:
                    all_conv.append({
                        'eps': eps, 'T_f': T_f, 'T_w1': tw1,
                        'T_w2': c['T_w2'], 'T_tot': c['t'],
                        'r': c['r'], 'beta_f': c['beta_f'],
                        'd_fold_deg': d_fold_deg,
                        'peak_vel': peak_vel
                    })

    print(f"  eps={eps:.1f} done ({count}/{total})")

print(f"\n{'='*80}")
print(f"Found {len(all_conv)} convergent, servo-feasible cases!")
print(f"{'='*80}")

if all_conv:
    all_conv.sort(key=lambda x: abs(x['r']))
    print(f"\n{'eps':>5} {'Tf':>5} {'Tw1':>5} | {'Tw2':>6} {'Ttot':>6} | "
          f"{'r':>8} | {'dfold':>6} {'v_max':>6} | {'bf':>7}")
    print("-" * 80)
    for c in all_conv[:30]:
        print(f"{c['eps']:5.1f} {c['T_f']*1000:4.0f}ms {c['T_w1']*1000:4.0f}ms | "
              f"{c['T_w2']*1000:5.0f}ms {c['T_tot']*1000:5.0f}ms | "
              f"{c['r']:8.4f} | {c['d_fold_deg']:5.1f}d {c['peak_vel']:5.0f}d/s | "
              f"{np.degrees(c['beta_f']):6.2f}d")

    # ==============================
    # Best trajectory
    # ==============================
    best = all_conv[0]
    print(f"\n*** BEST: eps={best['eps']}, Tf={best['T_f']*1000:.0f}ms, "
          f"Tw1={best['T_w1']*1000:.0f}ms ***")
    print(f"    r={best['r']:.4f}, Tw2={best['T_w2']*1000:.0f}ms, "
          f"Ttot={best['T_tot']*1000:.0f}ms")
    print(f"    delta_fold={best['d_fold_deg']:.1f}deg, "
          f"peak_vel={best['peak_vel']:.0f}deg/s")

    sol, t3, _, d_fold = run_cycle(beta0, best['eps'], best['T_f'], best['T_w1'])
    t2 = best['T_f'] + best['T_w1']
    T_end = best['T_tot'] + 0.1
    t_plot = np.linspace(0, T_end, 3000)
    phi_d = [np.degrees(sol.sol(t)[0]) for t in t_plot]
    beta_d = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_plot]

    a = 4*d_fold/best['T_f']**2
    Tf = best['T_f']
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
    phases = [(0,Tf,'#FFD700','FOLD'),(Tf,t2,'#90EE90','WAIT1'),
              (t2,t3,'#87CEEB','EXTEND'),(t3,T_end,'#DDA0DD','WAIT2')]
    for ax in axes:
        for a_,b_,c_,l_ in phases:
            ax.axvspan(a_*1000,min(b_,T_end)*1000,alpha=0.15,color=c_)
            ax.text((a_+min(b_,T_end))/2*1000, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1]>0 else 5,
                   l_, ha='center', fontsize=10, alpha=0.6)
        for tb in [Tf,t2,t3]:
            ax.axvline(tb*1000,color='gray',ls=':',alpha=0.4)

    axes[0].plot(np.array(t_plot)*1000, phi_d, 'b-', lw=2)
    axes[0].axhline(0, color='red', ls='--', alpha=0.5)
    axes[0].axvline(best['T_tot']*1000, color='green', lw=2,
                    label=f'phi=0 @ {best["T_tot"]*1000:.0f}ms')
    axes[0].set_ylabel('phi [deg]', fontsize=12)
    axes[0].legend(fontsize=11)

    axes[1].plot(np.array(t_plot)*1000, beta_d, 'r-', lw=2)
    axes[1].axhline(0, color='black', ls='-', lw=0.5)
    axes[1].axhline(5, color='gray', ls='--', alpha=0.3, label='beta0=5d')
    axes[1].axvline(best['T_tot']*1000, color='green', lw=2)
    bf_d = np.degrees(best['beta_f'])
    axes[1].axhline(bf_d, color='green', ls='--', alpha=0.5,
                    label=f'beta_f={bf_d:.2f}d')
    axes[1].set_ylabel('beta [deg]', fontsize=12)
    axes[1].legend(fontsize=11)

    axes[2].plot(np.array(t_plot)*1000, np.degrees(delta_d), 'g-', lw=2)
    axes[2].set_ylabel('delta [deg]', fontsize=12)
    axes[2].set_xlabel('time [ms]', fontsize=12)

    axes[0].set_title(
        f'BEST servo-feasible: eps={best["eps"]}, Tf={Tf*1000:.0f}ms, '
        f'Tw1={best["T_w1"]*1000:.0f}ms, Tw2={best["T_w2"]*1000:.0f}ms\n'
        f'r={best["r"]:.4f} | delta={best["d_fold_deg"]:.0f}d | '
        f'v_max={best["peak_vel"]:.0f}d/s | '
        f'T_total={best["T_tot"]*1000:.0f}ms | '
        f'beta: 5.0 -> {bf_d:.2f}d', fontsize=11)

    plt.tight_layout()
    out = __file__.replace('.py', '_best.png')
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()

print("\nDone!")

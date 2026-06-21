"""4-Phase convergence check with ROBOT parameters (no head)"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

g = 9.81

# ==============================
# Robot parameters (from preset_robot.json, no head)
# ==============================
M1 = 0.1       # lower body mass (kg)
m2 = 0.25      # upper body mass (kg) - no head
L1 = 0.2       # lower body length (m)
L2 = 0.2       # upper body length (m)
R_rope = 0.3   # rope radius (m)
LC1 = L1/2; LC2 = L2/2

Mt = M1 + m2
I1 = M1*L1**2/12; I2 = m2*L2**2/12
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

print("="*60)
print("ROBOT parameters (no head)")
print("="*60)
print(f"  M1={M1}kg  m2={m2}kg  Mt={Mt}kg")
print(f"  L1={L1}m  L2={L2}m  R={R_rope}m")
print(f"  h_com = {h_com:.4f}m")
print(f"  c_foot = {c_foot:.4f}")
print(f"  w_eff = {w_eff:.4f} rad/s  (T_quarter = {T_quarter*1000:.1f}ms)")
print(f"  lam_eff = {lam_eff:.4f} rad/s  (1/lam = {1000/lam_eff:.1f}ms)")
print(f"  d_fold(eps=4, beta0=5d) = {np.degrees(h_com*(1+4)*np.radians(5)/c_foot):.1f}deg")
print()

# ==============================
# 4-Phase: F-W1-E-W2 with phi=0 detection in W2
# ==============================
def run_4phase(beta0, eps, T_f, T_w1):
    sigma0 = Mt*h_com*beta0
    d_fold = h_com*(1+eps)*abs(beta0)/c_foot*np.sign(beta0)
    a = 4*d_fold/T_f**2
    t2 = T_f + T_w1; t3 = t2 + T_f
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
        r1 = -gMR*phi; r2 = g_ps*sig - C_sd*dd
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
                crossings.append({'T_w2':tz-t3,'t':tz,'beta_f':bf,'r':bf/beta0})
            except: pass
    return sol, t3, crossings

beta0 = np.radians(5)

# STS3215 servo: max 360 deg/s
SERVO_MAX = 360

eps_vals = np.arange(2, 12, 0.5)
tf_vals = [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
tw1_vals = np.arange(0.01, 2.0, 0.03)

print(f"Searching: {len(eps_vals)} eps x {len(tf_vals)} Tf x {len(tw1_vals)} Tw1")
all_conv = []

for eps in eps_vals:
    d_deg = np.degrees(h_com*(1+eps)*beta0/c_foot)
    for T_f in tf_vals:
        v_peak = 2*d_deg/T_f
        servo_ok = v_peak <= SERVO_MAX

        for tw1 in tw1_vals:
            _, _, crossings = run_4phase(beta0, eps, T_f, tw1)
            if crossings:
                c = crossings[0]
                if abs(c['r']) < 1:
                    all_conv.append({
                        'eps':eps, 'T_f':T_f, 'T_w1':tw1,
                        'T_w2':c['T_w2'], 'T_tot':c['t'],
                        'r':c['r'], 'beta_f':c['beta_f'],
                        'd_fold':d_deg, 'v_peak':v_peak, 'servo_ok':servo_ok
                    })
    print(f"  eps={eps:.1f} done ({len(all_conv)} conv)")

print(f"\n{'='*80}")
print(f"Total convergent: {len(all_conv)}")
servo_ok_list = [c for c in all_conv if c['servo_ok']]
print(f"Servo-feasible (v<={SERVO_MAX}d/s): {len(servo_ok_list)}")
print(f"{'='*80}")

# Print all results
for label, data_list in [("ALL convergent", all_conv), ("SERVO-FEASIBLE", servo_ok_list)]:
    if not data_list: continue
    data_list.sort(key=lambda x: abs(x['r']))
    print(f"\n--- {label} (top 25) ---")
    print(f"{'eps':>5} {'Tf':>5} {'Tw1':>6} | {'Tw2':>6} {'Ttot':>6} | "
          f"{'r':>8} | {'d':>5} {'vpeak':>7} | {'servo':>5}")
    print("-" * 75)
    for c in data_list[:25]:
        print(f"{c['eps']:5.1f} {c['T_f']*1000:4.0f}ms {c['T_w1']*1000:5.0f}ms | "
              f"{c['T_w2']*1000:5.0f}ms {c['T_tot']:.2f}s | "
              f"{c['r']:8.4f} | {c['d_fold']:4.0f}d {c['v_peak']:6.0f}d/s | "
              f"{'OK' if c['servo_ok'] else 'FAST'}")

# Best trajectory (servo-feasible first, then any)
best_list = servo_ok_list if servo_ok_list else all_conv
if best_list:
    best_list.sort(key=lambda x: abs(x['r']))
    best = best_list[0]
    sol, t3, _ = run_4phase(beta0, best['eps'], best['T_f'], best['T_w1'])
    Tf=best['T_f']; t2=Tf+best['T_w1']
    T_end = best['T_tot']+0.02
    t_plot = np.linspace(0, T_end, 3000)
    phi_d = [np.degrees(sol.sol(t)[0]) for t in t_plot]
    beta_d = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_plot]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    phases = [(0,Tf,'#FFD700','F'),(Tf,t2,'#90EE90','W1'),
              (t2,t3,'#87CEEB','E'),(t3,T_end,'#DDA0DD','W2')]
    for ax in axes:
        for a_,b_,c_,l_ in phases:
            ax.axvspan(a_*1000,min(b_,T_end)*1000,alpha=0.15,color=c_)

    axes[0].plot(np.array(t_plot)*1000, phi_d, 'b-', lw=2)
    axes[0].axhline(0, color='red', ls='--'); axes[0].set_ylabel('phi [deg]')
    axes[0].axvline(best['T_tot']*1000, color='green', lw=2, label=f'phi=0 @ {best["T_tot"]*1000:.0f}ms')
    axes[0].legend()

    axes[1].plot(np.array(t_plot)*1000, beta_d, 'r-', lw=2)
    axes[1].axhline(0, color='black', ls='-', lw=0.5)
    axes[1].set_ylabel('beta [deg]'); axes[1].set_xlabel('time [ms]')
    bf_d=np.degrees(best['beta_f'])
    axes[1].axhline(bf_d, color='green', ls='--', label=f'beta_f={bf_d:.2f}d')
    axes[1].legend()

    tag = "servo-OK" if best['servo_ok'] else "FAST-servo"
    axes[0].set_title(
        f'ROBOT 4-Phase ({tag}): eps={best["eps"]}, Tf={Tf*1000:.0f}ms, '
        f'Tw1={best["T_w1"]*1000:.0f}ms\n'
        f'r={best["r"]:.4f} | d={best["d_fold"]:.0f}d | v={best["v_peak"]:.0f}d/s | '
        f'T={best["T_tot"]:.2f}s', fontsize=11)
    plt.tight_layout()
    out = __file__.replace('.py', '_best.png')
    plt.savefig(out, dpi=150); print(f"\nSaved: {out}")

print("\nDone!")

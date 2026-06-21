"""FAST 4-Phase check for ROBOT params — coarse grid + low tolerance"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time as _time

g = 9.81
# Robot params (no head)
M1=0.1; m2=0.25; L1=0.2; L2=0.2; R_rope=0.3
LC1=L1/2; LC2=L2/2; Mt=M1+m2
I1=M1*L1**2/12; I2=m2*L2**2/12
p1=M1*LC1+m2*L1; p2=m2*LC2; ps=p1+p2
h_com=(M1*LC1+m2*(L1+LC2))/Mt; c_foot=p2/ps
J_aa=M1*LC1**2+I1+m2*L1**2; J_tt=m2*LC2**2+I2
J_at=m2*L1*LC2; J_tot=J_aa+J_tt+2*J_at
C_sd=(-J_aa*p2+J_tt*p1+J_at*(p1-p2))/ps**2
M11=Mt*R_rope**2; M12=R_rope; M22=J_tot/ps**2
det_M=M11*M22-M12**2
iM11=M22/det_M; iM12=-M12/det_M; iM22=M11/det_M
gMR=g*Mt*R_rope; g_ps=g/ps
w_eff=np.sqrt(abs(iM11*gMR)); lam_eff=np.sqrt(abs(iM22*g_ps))
T_quarter=np.pi/(2*w_eff)

print("ROBOT params (no head)")
print(f"  M1={M1} m2={m2} Mt={Mt} L1={L1} L2={L2} R={R_rope}")
print(f"  h_com={h_com:.4f} c_foot={c_foot:.4f}")
print(f"  w_eff={w_eff:.2f} (Tq={T_quarter*1000:.0f}ms)")
print(f"  lam_eff={lam_eff:.2f} (1/lam={1/lam_eff*1000:.0f}ms)")
d5 = np.degrees(h_com*5*np.radians(5)/c_foot)
print(f"  d_fold(eps=4,b0=5d)={np.degrees(h_com*5*np.radians(5)/c_foot):.1f}deg")
print()

beta0 = np.radians(5)

def run_fast(eps, T_f, T_w1):
    """Run 4-phase, return phi at W2 sample points for zero detection"""
    s0 = Mt*h_com*beta0
    df = h_com*(1+eps)*beta0/c_foot
    a = 4*df/T_f**2
    t2=T_f+T_w1; t3=t2+T_f; Te=t3+3*T_quarter

    def rhs(t,y):
        p,dp,s,ds=y
        if t<T_f/2: dd=+a
        elif t<T_f: dd=-a
        elif t<t2: dd=0
        elif t<t2+T_f/2: dd=-a
        elif t<t3: dd=+a
        else: dd=0
        return[dp, iM11*(-gMR*p)+iM12*(g_ps*s-C_sd*dd),
               ds, iM12*(-gMR*p)+iM22*(g_ps*s-C_sd*dd)]

    sol = solve_ivp(rhs,(0,Te),[0,0,s0,0],method='RK45',
                    rtol=1e-8, atol=1e-10, max_step=T_f/20,
                    dense_output=True)

    # Sample W2 region
    tw2 = np.linspace(t3+0.002, Te, 500)
    pw2 = np.array([sol.sol(t)[0] for t in tw2])

    for k in range(len(pw2)-1):
        if pw2[k]*pw2[k+1] < 0:
            try:
                tz = brentq(lambda t:sol.sol(t)[0], tw2[k], tw2[k+1], xtol=1e-8)
                bf = sol.sol(tz)[2]/(Mt*h_com)
                return bf/beta0, tz, np.degrees(df), np.degrees(2*df/T_f)
            except: pass
    return None, None, None, None

t0 = _time.time()

# Coarse grid
eps_v = np.arange(2, 15, 1.0)
tf_v = [0.02, 0.05, 0.10, 0.15]
tw1_v = np.arange(0.01, 2.0, 0.05)

print(f"Grid: {len(eps_v)}x{len(tf_v)}x{len(tw1_v)} = {len(eps_v)*len(tf_v)*len(tw1_v)} combos")

conv = []
for eps in eps_v:
    for Tf in tf_v:
        for tw1 in tw1_v:
            r, tt, dd, vp = run_fast(eps, Tf, tw1)
            if r is not None and abs(r) < 1:
                conv.append((eps, Tf, tw1, r, tt, dd, vp, vp<=360))
    elapsed = _time.time()-t0
    print(f"  eps={eps:.0f} [{elapsed:.1f}s] conv={len(conv)}")

print(f"\nTotal: {len(conv)} convergent in {_time.time()-t0:.1f}s")
servo_ok = [c for c in conv if c[7]]
print(f"Servo-feasible: {len(servo_ok)}")

for label, lst in [("ALL", conv), ("SERVO-OK", servo_ok)]:
    if not lst: continue
    lst.sort(key=lambda x: abs(x[3]))
    print(f"\n--- {label} top 15 ---")
    print(f"{'eps':>5} {'Tf':>5} {'Tw1':>6} | {'r':>8} {'Ttot':>6} | {'d':>5} {'v':>7}")
    for e,tf,tw1,r,tt,d,v,sok in lst[:15]:
        print(f"{e:5.1f} {tf*1000:4.0f}ms {tw1*1000:5.0f}ms | {r:8.4f} {tt:.2f}s | {d:4.0f}d {v:6.0f}d/s {'*' if sok else ''}")

# Best trajectory
best = (servo_ok if servo_ok else conv)
if best:
    best.sort(key=lambda x: abs(x[3]))
    e,tf,tw1,r,tt,d,v,sok = best[0]
    # Re-run with dense output for plotting
    s0=Mt*h_com*beta0; df=h_com*(1+e)*beta0/c_foot; a=4*df/tf**2
    t2=tf+tw1; t3=t2+tf; Te=tt+0.02
    def rhs2(t,y):
        p,dp,s,ds=y
        if t<tf/2: dd=+a
        elif t<tf: dd=-a
        elif t<t2: dd=0
        elif t<t2+tf/2: dd=-a
        elif t<t3: dd=+a
        else: dd=0
        return[dp,iM11*(-gMR*p)+iM12*(g_ps*s-C_sd*dd),
               ds,iM12*(-gMR*p)+iM22*(g_ps*s-C_sd*dd)]
    sol=solve_ivp(rhs2,(0,Te),[0,0,s0,0],method='RK45',rtol=1e-8,atol=1e-10,
                  max_step=tf/20,dense_output=True)
    tp=np.linspace(0,Te,2000)
    phi_d=[np.degrees(sol.sol(t)[0]) for t in tp]
    beta_d=[np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in tp]

    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    phases=[(0,tf,'#FFD700','F'),(tf,t2,'#90EE90','W1'),
            (t2,t3,'#87CEEB','E'),(t3,Te,'#DDA0DD','W2')]
    for ax in axes:
        for a_,b_,c_,l_ in phases:
            ax.axvspan(a_*1000,min(b_,Te)*1000,alpha=0.15,color=c_)

    axes[0].plot(tp*1000,phi_d,'b-',lw=2)
    axes[0].axhline(0,color='red',ls='--')
    axes[0].axvline(tt*1000,color='green',lw=2,label=f'phi=0 @ {tt*1000:.0f}ms')
    axes[0].set_ylabel('phi [deg]'); axes[0].legend()

    axes[1].plot(tp*1000,beta_d,'r-',lw=2)
    axes[1].axhline(0,color='black',ls='-',lw=0.5)
    axes[1].set_ylabel('beta [deg]'); axes[1].set_xlabel('ms')

    tag="SERVO-OK" if sok else "FAST"
    axes[0].set_title(f'ROBOT 4-Phase ({tag}): eps={e}, Tf={tf*1000:.0f}ms, '
                      f'Tw1={tw1*1000:.0f}ms | r={r:.4f} d={d:.0f}d v={v:.0f}d/s')
    plt.tight_layout()
    out=__file__.replace('.py','_best.png')
    plt.savefig(out,dpi=150); print(f"\nSaved: {out}")

print("\nDone!")

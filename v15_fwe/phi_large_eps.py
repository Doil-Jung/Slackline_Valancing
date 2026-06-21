"""Direct phi trajectory in W2 for large eps"""
import numpy as np
from scipy.integrate import solve_ivp
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
w_eff=np.sqrt(iM11*gMR)
T_quarter=np.pi/(2*w_eff)

beta0 = np.radians(5)
T_f = 0.05; T_w1 = 0.005

fig, axes = plt.subplots(3, 3, figsize=(18, 14))

for idx, eps in enumerate([0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]):
    sigma0 = Mt*h_com*beta0
    d_fold = h_com*(1+eps)*beta0/c_foot
    a = 4*d_fold/T_f**2
    t2 = T_f+T_w1; t3 = t2+T_f
    T_end = t3 + 4*T_quarter

    def get_dd(t, _a=a, _T_f=T_f, _t2=t2, _t3=t3):
        if t < _T_f/2: return +_a
        elif t < _T_f: return -_a
        elif t < _t2: return 0.0
        elif t < _t2+_T_f/2: return -_a
        elif t < _t3: return +_a
        else: return 0.0

    def rhs(t, y, _a=a, _T_f=T_f, _t2=t2, _t3=t3):
        phi,dp,sig,ds = y
        dd = get_dd(t, _a, _T_f, _t2, _t3)
        r1=-gMR*phi; r2=g_ps*sig-C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

    sol = solve_ivp(rhs, (0, T_end), [0,0,sigma0,0],
                    method='RK45', rtol=1e-10, atol=1e-12, max_step=T_f/40,
                    dense_output=True)

    t_plot = np.linspace(0, T_end, 2000)
    phi_p = [np.degrees(sol.sol(t)[0]) for t in t_plot]
    beta_p = [np.degrees(sol.sol(t)[2]/(Mt*h_com)) for t in t_plot]

    ax = axes.flat[idx]
    ax.plot(np.array(t_plot)*1000, phi_p, 'b-', lw=1.5, label='phi')
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    ax.axvline(t3*1000, color='gray', ls=':', alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(np.array(t_plot)*1000, beta_p, 'r-', lw=0.8, alpha=0.4, label='beta')
    ax2.axhline(0, color='black', ls='-', lw=0.5)

    # Check beta at fold end
    bf_end = np.degrees(sol.sol(T_f)[2]/(Mt*h_com))
    # Check phi at W2 start
    phi_w2 = np.degrees(sol.sol(t3)[0])

    ax.set_title(f'eps={eps} (d={np.degrees(d_fold):.0f}d)\n'
                 f'beta@fold={bf_end:.1f}d, phi@W2={phi_w2:.1f}d', fontsize=9)
    ax.set_xlabel('ms', fontsize=7)
    ax.set_ylabel('phi [deg]', fontsize=7)
    ax2.set_ylabel('beta [deg]', color='red', fontsize=7)

plt.suptitle(f'phi trajectory for large eps (T_f={T_f*1000:.0f}ms, Tw1={T_w1*1000:.0f}ms)', fontsize=12)
plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=150)
print(f"Saved: {out}")

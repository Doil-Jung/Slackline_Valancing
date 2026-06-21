"""FOLD phase beta trajectory diagnostic: ms-level tracking"""
import numpy as np
from scipy.integrate import solve_ivp
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

beta0 = np.radians(5)
eps = 0.1; T_f = 0.05; T_w1 = 0.005

sigma0 = Mt * h_com * beta0
d_fold = h_com * (1 + eps) * beta0 / c_foot
a = 4 * d_fold / T_f**2

print(f"beta0 = {np.degrees(beta0):.1f} deg")
print(f"eps = {eps}")
print(f"delta_fold = {np.degrees(d_fold):.2f} deg ({d_fold:.4f} rad)")
print(f"a (accel) = {a:.1f} rad/s^2")
print(f"sigma0 = {sigma0:.4f}")
print(f"K_delta = {-iM22*C_sd:.4f}")
print(f"K_delta * a = {-iM22*C_sd*a:.1f}")
print(f"lambda^2 * sigma0 = {iM22*g_ps*sigma0:.1f}")
print(f"Ratio K_delta*a / (lam^2*sigma0) = {abs(-iM22*C_sd*a/(iM22*g_ps*sigma0)):.1f}")

# === Run ODE ===
t2 = T_f + T_w1; t3 = t2 + T_f
T_end = t3 + 0.3

def get_dd(t):
    if t < T_f/2: return +a
    elif t < T_f: return -a
    elif t < t2: return 0.0
    elif t < t2+T_f/2: return -a
    elif t < t3: return +a
    else: return 0.0

def rhs(t, y):
    phi, dp, sig, ds = y
    dd = get_dd(t)
    r1 = -gMR*phi; r2 = g_ps*sig - C_sd*dd
    return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

t_eval = np.linspace(0, T_end, 5000)
sol = solve_ivp(rhs, (0, T_end), [0, 0, sigma0, 0],
                method='RK45', rtol=1e-12, atol=1e-14, t_eval=t_eval)

phi = sol.y[0]
beta = sol.y[2] / (Mt * h_com)

# delta position
delta_pos = np.zeros_like(sol.t)
for i, t in enumerate(sol.t):
    if t < T_f/2:
        delta_pos[i] = 0.5*a*t**2
    elif t < T_f:
        dt = t - T_f/2; v = a*T_f/2
        delta_pos[i] = 0.5*a*(T_f/2)**2 + v*dt - 0.5*a*dt**2
    elif t < t2:
        delta_pos[i] = d_fold
    elif t < t2+T_f/2:
        dt = t-t2
        delta_pos[i] = d_fold - 0.5*a*dt**2
    elif t < t3:
        dt = t-(t2+T_f/2); v = a*T_f/2
        delta_pos[i] = d_fold - 0.5*a*(T_f/2)**2 - v*dt + 0.5*a*dt**2
    else:
        delta_pos[i] = 0.0

# Print key moments
print(f"\n{'time':>7} {'phi[deg]':>9} {'beta[deg]':>10} {'delta[deg]':>11} {'delta_dd':>9}")
print("-" * 50)
key_times = [0, T_f/4, T_f/2, 3*T_f/4, T_f, T_f+T_w1/2, t2, t2+T_f/2, t3, t3+0.05, t3+0.1]
for t_key in key_times:
    idx = np.argmin(np.abs(sol.t - t_key))
    dd = get_dd(t_key)
    print(f"{sol.t[idx]*1000:6.1f}ms {np.degrees(phi[idx]):9.4f} {np.degrees(beta[idx]):10.4f} "
          f"{np.degrees(delta_pos[idx]):11.4f} {dd:9.1f}")

# === Plot ===
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

# Phase shading
phases = [(0, T_f, '#FFD700', 'FOLD'), (T_f, t2, '#90EE90', 'W1'),
          (t2, t3, '#87CEEB', 'EXT'), (t3, T_end, '#DDA0DD', 'W2')]
for ax in axes:
    for t0, t1, c, lbl in phases:
        ax.axvspan(t0*1000, t1*1000, alpha=0.15, color=c)

axes[0].plot(sol.t*1000, np.degrees(phi), 'b-', lw=2)
axes[0].set_ylabel('phi [deg]')
axes[0].axhline(0, color='gray', ls='--', alpha=0.5)

axes[1].plot(sol.t*1000, np.degrees(beta), 'r-', lw=2)
axes[1].axhline(0, color='black', ls='-', lw=1)
axes[1].axhline(np.degrees(beta0), color='gray', ls='--', alpha=0.5, label='beta0')
axes[1].axhline(-np.degrees(beta0*eps), color='green', ls='--', alpha=0.5, label=f'-eps*beta0 (target)')
axes[1].set_ylabel('beta [deg]')
axes[1].legend()

axes[2].plot(sol.t*1000, np.degrees(delta_pos), 'g-', lw=2)
axes[2].set_ylabel('delta [deg]')

# sigma_dd decomposition
sigma_dd_phi = np.array([iM12 * (-gMR * phi[i]) for i in range(len(sol.t))])  # from phi coupling (in sigma eq)
sigma_dd_self = np.array([iM22 * g_ps * sol.y[2][i] for i in range(len(sol.t))])  # lambda^2 * sigma
sigma_dd_delta = np.array([-iM22 * C_sd * get_dd(t) for t in sol.t])  # from delta
# Actually need to use full inverse: sigma_dd = iM12*r1 + iM22*r2
r1_arr = np.array([-gMR * phi[i] for i in range(len(sol.t))])
r2_arr = np.array([g_ps * sol.y[2][i] - C_sd * get_dd(t) for i, t in enumerate(sol.t)])
sigma_dd_total = iM12 * r1_arr + iM22 * r2_arr

axes[3].plot(sol.t*1000, sigma_dd_delta, 'g-', lw=1, label='delta forcing', alpha=0.7)
axes[3].plot(sol.t*1000, iM22*g_ps*sol.y[2], 'r-', lw=1, label='lambda^2*sigma (self)', alpha=0.7)
axes[3].plot(sol.t*1000, iM12*r1_arr, 'b-', lw=1, label='phi coupling', alpha=0.7)
axes[3].plot(sol.t*1000, sigma_dd_total, 'k-', lw=2, label='total sigma_dd')
axes[3].set_ylabel('sigma_dd [rad/s^2]')
axes[3].set_xlabel('time [ms]')
axes[3].legend(fontsize=8)
axes[3].set_ylim(-300, 300)

axes[0].set_title(f'FOLD diagnostic: eps={eps}, T_f={T_f*1000:.0f}ms, beta0={np.degrees(beta0):.1f}deg\n'
                  f'delta_fold={np.degrees(d_fold):.1f}deg, a={a:.0f}rad/s^2')
plt.tight_layout()
out = __file__.replace('.py', '.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

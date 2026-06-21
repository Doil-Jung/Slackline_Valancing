"""
라그랑지안 기반 FWE 모델 — 방향 2 구현
φ̈ = -ω²φ + K_φδ·δ̈(t)  (σ 결합만 무시)
σ̈ = K_φ·φ(t) + λ²·σ + K_δ·δ̈(t)

삼각형 δ 프로파일, 해석해 vs 완전결합 ODE 비교
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================
# 1. 시스템 파라미터 (사람 크기 모델)
# =============================================
g = 9.81
R_rope = 1.0  # 줄 곡률 반경

# 개별 강체 파라미터
M1 = 30.0    # 하체 질량 [kg]
m2 = 45.0    # 상체 질량 [kg]
L1 = 0.85    # 하체 길이 (발목→힙) [m]
LC1 = 0.30   # 하체 CoM (발목 기준) [m]
LC2 = 0.40   # 상체 CoM (힙 기준) [m]
L_upper = 0.85  # 상체 길이 [m]

Mt = M1 + m2
I1 = M1 * L1**2 / 12  # 하체 관성모멘트
I2 = m2 * L_upper**2 / 12  # 상체 관성모멘트

# 결합 변수
p1 = M1 * LC1 + m2 * L1
p2 = m2 * LC2
ps = p1 + p2
h_com = (M1 * LC1 + m2 * (L1 + LC2)) / Mt
c_foot = p2 / ps

print("=" * 55)
print("System Parameters")
print("=" * 55)
print(f"M_t = {Mt} kg, h = {h_com:.4f} m, R = {R_rope} m")
print(f"p1 = {p1:.3f}, p2 = {p2:.3f}, ps = {ps:.3f}")
print(f"c_foot = {c_foot:.4f}")

# 관성 부분행렬 계수
J_aa = M1 * LC1**2 + I1 + m2 * L1**2
J_tt = m2 * LC2**2 + I2
J_at = m2 * L1 * LC2
J_tot = J_aa + J_tt + 2 * J_at

print(f"\nJ_aa={J_aa:.3f}, J_tt={J_tt:.3f}, J_at={J_at:.3f}")
print(f"J_tot={J_tot:.3f}")

# sigma_dot * delta_dot coupling
C_sd = (-J_aa * p2 + J_tt * p1 + J_at * (p1 - p2)) / ps**2
print(f"C_sd = {C_sd:.5f}")

# =============================================
# 2. Inertia matrix inversion -> coupling coefficients
# =============================================
M11 = Mt * R_rope**2
M12 = R_rope
M22 = J_tot / ps**2
det_M = M11 * M22 - M12**2

iM11 = M22 / det_M
iM12 = -M12 / det_M
iM22 = M11 / det_M

print(f"\nInertia: [[{M11:.2f}, {M12:.2f}], [{M12:.2f}, {M22:.5f}]]")
print(f"det(M) = {det_M:.4f}")

# RHS coefficients
gMR = g * Mt * R_rope
g_ps = g / ps

# φ̈ 계수들
w2_eff = iM11 * gMR                    # φ의 자기 복원
K_phi_in_phi = -iM12 * g_ps            # σ → φ 교차 (작음)
K_phidelta = -iM12 * C_sd              # δ̈ → φ 가력

# σ̈ 계수들
K_phi = -iM12 * gMR                    # φ → σ 교차 (큼!)
lam2_eff = iM22 * g_ps                 # σ 자기 반발
K_delta = -iM22 * C_sd                 # δ̈ → σ 가력

w_eff = np.sqrt(w2_eff)
lam_eff = np.sqrt(lam2_eff)

print(f"\n{'='*55}")
print("Coupling coefficients (after M-inverse)")
print(f"{'='*55}")
print(f"phi_dd = -{w2_eff:.2f}*phi + ({K_phi_in_phi:.4f})*sigma + ({K_phidelta:.4f})*delta_dd")
print(f"sigma_dd = +{K_phi:.1f}*phi + {lam2_eff:.2f}*sigma + ({K_delta:.2f})*delta_dd")
print(f"\nw_eff = {w_eff:.4f} rad/s (T_q = {np.pi/(2*w_eff):.4f} s)")
print(f"lam_eff = {lam_eff:.4f} rad/s")
print(f"\nsigma->phi cross: {abs(K_phi_in_phi)/w2_eff*100:.2f}% (small=OK to ignore)")
print(f"phi->sigma cross: {abs(K_phi)/lam2_eff:.1f}x (large=strong coupling)")
print(f"delta_dd->phi coeff: {K_phidelta:.4f}")
print(f"delta_dd->sigma coeff: {K_delta:.4f}")

# =============================================
# 3. 삼각형 δ 프로파일의 해석해 (방향 2)
# =============================================
def solve_direction2_cycle(beta0, eps, T_f, T_w, R=1.0):
    """방향 2: φ에 δ̈ 가력 포함, σ 결합만 무시.
    삼각형 δ 프로파일. 5 sub-phase:
      FOLD_acc(T_f/2) → FOLD_dec(T_f/2) → WAIT(T_w) → EXT_acc(T_f/2) → EXT_dec(T_f/2)
    """
    sigma0 = Mt * h_com * beta0
    delta_fold = h_com * (1 + eps) * abs(beta0) / c_foot * np.sign(beta0)
    a = 4 * delta_fold / T_f**2  # 삼각형 가속도 크기

    w = w_eff; la = lam_eff
    # sub-phase별 δ̈ 값
    dd_phases = [+a, -a, 0.0, -a, +a]
    dt_phases = [T_f/2, T_f/2, T_w, T_f/2, T_f/2]

    phi, dphi = 0.0, 0.0
    sig, dsig = sigma0, 0.0

    for dd, dt in zip(dd_phases, dt_phases):
        if dt < 1e-15:
            continue
        # --- φ 해: φ̈ = -ω²φ + K_phidelta·dd ---
        phi_p = K_phidelta * dd / w**2 if abs(dd) > 1e-15 else 0.0
        A_phi = phi - phi_p
        B_phi = dphi / w
        phi_new = A_phi * np.cos(w*dt) + B_phi * np.sin(w*dt) + phi_p
        dphi_new = -A_phi * w * np.sin(w*dt) + B_phi * w * np.cos(w*dt)

        # --- σ 해: σ̈ = K_phi·φ(t) + λ²·σ + K_delta·dd ---
        # φ(t) = A_phi·cos(wt) + B_phi·sin(wt) + phi_p
        # 특수해: cos → -K_phi·A_phi/(w²+la²)·cos
        #         sin → -K_phi·B_phi/(w²+la²)·sin
        #         상수 → -(K_phi·phi_p + K_delta·dd)/la²
        denom_trig = w**2 + la**2
        sp_cos = -K_phi * A_phi / denom_trig
        sp_sin = -K_phi * B_phi / denom_trig
        const_rhs = K_phi * phi_p + K_delta * dd
        sp_const = -const_rhs / la**2 if abs(la) > 1e-15 else 0.0

        # 초기조건으로 A_sig, B_sig 결정
        # σ(0) = sig = A_sig + sp_cos + sp_const
        # σ̇(0) = dsig = B_sig·la + sp_sin·w  (???)
        # 정확히: σ(t) = A_sig·cosh(la·t) + B_sig·sinh(la·t) + sp_cos·cos(wt) + sp_sin·sin(wt) + sp_const
        # σ̇(t) = A_sig·la·sinh(la·t) + B_sig·la·cosh(la·t) - sp_cos·w·sin(wt) + sp_sin·w·cos(wt)
        A_sig = sig - sp_cos - sp_const
        B_sig = (dsig - sp_sin * w) / la if abs(la) > 1e-15 else 0.0

        sig_new = (A_sig * np.cosh(la*dt) + B_sig * np.sinh(la*dt)
                   + sp_cos * np.cos(w*dt) + sp_sin * np.sin(w*dt) + sp_const)
        dsig_new = (A_sig * la * np.sinh(la*dt) + B_sig * la * np.cosh(la*dt)
                    - sp_cos * w * np.sin(w*dt) + sp_sin * w * np.cos(w*dt))

        phi, dphi = phi_new, dphi_new
        sig, dsig = sig_new, dsig_new

    beta_final = sig / (Mt * h_com)
    return phi, dphi, beta_final, dsig / (Mt * h_com)


# =============================================
# 4. 완전 결합 ODE (기준 솔루션)
# =============================================
def solve_full_coupled(beta0, eps, T_f, T_w, R=1.0):
    """완전 결합 4-상태 ODE: state = [φ, φ̇, σ, σ̇]"""
    sigma0 = Mt * h_com * beta0
    delta_fold = h_com * (1 + eps) * abs(beta0) / c_foot * np.sign(beta0)
    a = 4 * delta_fold / T_f**2
    T_total = 2 * T_f + T_w

    def get_ddelta(t):
        if t < T_f / 2:
            return +a
        elif t < T_f:
            return -a
        elif t < T_f + T_w:
            return 0.0
        elif t < T_f + T_w + T_f / 2:
            return -a
        else:
            return +a

    def rhs(t, y):
        phi, dphi, sig, dsig = y
        dd = get_ddelta(t)
        # M·[φ̈, σ̈]ᵀ = [-gMR·φ, g·σ/ps - C_sd·dd]ᵀ
        rhs1 = -gMR * phi
        rhs2 = g_ps * sig - C_sd * dd
        phi_dd = (iM11 * rhs1 + iM12 * rhs2)
        sig_dd = (iM12 * rhs1 + iM22 * rhs2)
        return [dphi, phi_dd, dsig, sig_dd]

    y0 = [0.0, 0.0, sigma0, 0.0]
    t_span = (0, T_total)
    sol = solve_ivp(rhs, t_span, y0, method='RK45', rtol=1e-10, atol=1e-12,
                    max_step=T_f / 100)
    yf = sol.y[:, -1]
    beta_f = yf[2] / (Mt * h_com)
    return yf[0], yf[1], beta_f, yf[3] / (Mt * h_com)


# =============================================
# 5. 옵션 B (비교용: φ 완전 자유 진자)
# =============================================
def solve_optionB(beta0, eps, T_f, T_w, R=1.0):
    """옵션 B: φ̈ = -ω²φ (δ̈ 가력 없음), σ̈ = K_φ·φ + λ²σ + K_δ·δ̈"""
    sigma0 = Mt * h_com * beta0
    delta_fold = h_com * (1 + eps) * abs(beta0) / c_foot * np.sign(beta0)
    a = 4 * delta_fold / T_f**2
    w = w_eff; la = lam_eff

    dd_phases = [+a, -a, 0.0, -a, +a]
    dt_phases = [T_f/2, T_f/2, T_w, T_f/2, T_f/2]

    phi, dphi = 0.0, 0.0
    sig, dsig = sigma0, 0.0

    for dd, dt in zip(dd_phases, dt_phases):
        if dt < 1e-15:
            continue
        # φ: 순수 자유 진자 (δ̈ 가력 없음!)
        A_phi = phi
        B_phi = dphi / w
        phi_new = A_phi * np.cos(w*dt) + B_phi * np.sin(w*dt)
        dphi_new = -A_phi * w * np.sin(w*dt) + B_phi * w * np.cos(w*dt)

        # σ: 동일한 구조
        denom_trig = w**2 + la**2
        sp_cos = -K_phi * A_phi / denom_trig
        sp_sin = -K_phi * B_phi / denom_trig
        const_rhs = K_phi * 0.0 + K_delta * dd  # phi_p = 0 (no forcing)
        sp_const = -const_rhs / la**2 if abs(la) > 1e-15 else 0.0

        A_sig = sig - sp_cos - sp_const
        B_sig = (dsig - sp_sin * w) / la if abs(la) > 1e-15 else 0.0

        sig_new = (A_sig * np.cosh(la*dt) + B_sig * np.sinh(la*dt)
                   + sp_cos * np.cos(w*dt) + sp_sin * np.sin(w*dt) + sp_const)
        dsig_new = (A_sig * la * np.sinh(la*dt) + B_sig * la * np.cosh(la*dt)
                    - sp_cos * w * np.sin(w*dt) + sp_sin * w * np.cos(w*dt))

        phi, dphi = phi_new, dphi_new
        sig, dsig = sig_new, dsig_new

    beta_f = sig / (Mt * h_com)
    return phi, dphi, beta_f, dsig / (Mt * h_com)


# =============================================
# 6. 세 방법 비교
# =============================================
print(f"\n{'='*70}")
print("세 방법 비교: 옵션B vs 방향2 vs 완전결합")
print(f"{'='*70}")

beta0 = np.radians(5)
test_cases = [
    (0.5, 0.05, 0.05), (0.5, 0.10, 0.10),
    (1.0, 0.05, 0.05), (1.0, 0.05, 0.075),
    (1.0, 0.10, 0.05), (1.0, 0.10, 0.10),
    (1.5, 0.05, 0.05), (1.5, 0.10, 0.075),
    (2.0, 0.05, 0.05), (2.0, 0.10, 0.10),
]

print(f"\nbeta0 = {np.degrees(beta0):.1f} deg")
print(f"{'eps':>4} {'T_f':>5} {'T_w':>5} | {'OptB_r':>7} {'Dir2_r':>7} {'Full_r':>7} | "
      f"{'OptB_pf':>8} {'Dir2_pf':>8} {'Full_pf':>8} | {'Dir2 err':>8}")
print("-" * 95)

for eps, T_f, T_w in test_cases:
    pB = solve_optionB(beta0, eps, T_f, T_w)
    p2 = solve_direction2_cycle(beta0, eps, T_f, T_w)
    pF = solve_full_coupled(beta0, eps, T_f, T_w)

    rB = abs(pB[2] / beta0)
    r2 = abs(p2[2] / beta0)
    rF = abs(pF[2] / beta0)

    err_dir2 = abs(r2 - rF) / max(rF, 1e-10) * 100

    print(f"{eps:4.1f} {T_f:5.3f} {T_w:5.3f} | {rB:7.4f} {r2:7.4f} {rF:7.4f} | "
          f"{np.degrees(pB[0]):8.3f}d {np.degrees(p2[0]):8.3f}d {np.degrees(pF[0]):8.3f}d | "
          f"{err_dir2:7.2f}%")


# =============================================
# 7. 수렴비 히트맵: 방향2 vs 완전결합
# =============================================
print(f"\n{'='*55}")
print("Generating heatmap...")

eps_vals = np.linspace(0.2, 3.0, 30)
Tw_vals = np.linspace(0.01, 0.20, 25)
T_f_fixed = 0.08

R_dir2 = np.zeros((len(Tw_vals), len(eps_vals)))
R_full = np.zeros_like(R_dir2)

for i, eps in enumerate(eps_vals):
    for j, Tw in enumerate(Tw_vals):
        try:
            _, _, b2, _ = solve_direction2_cycle(beta0, eps, T_f_fixed, Tw)
            R_dir2[j, i] = abs(b2 / beta0)
        except:
            R_dir2[j, i] = np.nan
        try:
            _, _, bF, _ = solve_full_coupled(beta0, eps, T_f_fixed, Tw)
            R_full[j, i] = abs(bF / beta0)
        except:
            R_full[j, i] = np.nan

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, data, title in zip(axes[:2], [R_dir2, R_full], ['Direction 2 (analytic)', 'Full coupled (ODE)']):
    data_clip = np.clip(data, 0, 3)
    im = ax.contourf(eps_vals, Tw_vals, data_clip, levels=np.linspace(0, 2, 21), cmap='RdYlGn_r')
    ax.contour(eps_vals, Tw_vals, data, levels=[1.0], colors='black', linewidths=2)
    ax.set_xlabel('epsilon (overshoot)')
    ax.set_ylabel('T_w (wait time) [s]')
    ax.set_title(f'{title}\nT_f={T_f_fixed}s, beta0=5deg')
    plt.colorbar(im, ax=ax, label='convergence ratio r')

# 오차 맵
err = np.abs(R_dir2 - R_full) / np.maximum(R_full, 0.01) * 100
err_clip = np.clip(err, 0, 50)
im3 = axes[2].contourf(eps_vals, Tw_vals, err_clip, levels=20, cmap='hot_r')
axes[2].set_xlabel('ε')
axes[2].set_ylabel('T_w [s]')
axes[2].set_title('방향2 vs 완전결합 오차 [%]')
plt.colorbar(im3, ax=axes[2], label='상대 오차 %')

plt.tight_layout()
out_path = __file__.replace('.py', '_heatmap.png')
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.close()

# =============================================
# 8. 시계열 비교 (대표 케이스)
# =============================================
eps_demo, Tf_demo, Tw_demo = 1.0, 0.10, 0.08
sigma0 = Mt * h_com * beta0
delta_fold = h_com * (1 + eps_demo) * beta0 / c_foot
a_demo = 4 * delta_fold / Tf_demo**2
T_total = 2 * Tf_demo + Tw_demo

# 완전 결합 시계열
def get_ddelta_demo(t):
    if t < Tf_demo / 2: return +a_demo
    elif t < Tf_demo: return -a_demo
    elif t < Tf_demo + Tw_demo: return 0.0
    elif t < Tf_demo + Tw_demo + Tf_demo / 2: return -a_demo
    else: return +a_demo

def rhs_full(t, y):
    phi, dphi, sig, dsig = y
    dd = get_ddelta_demo(t)
    r1 = -gMR * phi
    r2 = g_ps * sig - C_sd * dd
    return [dphi, iM11*r1 + iM12*r2, dsig, iM12*r1 + iM22*r2]

t_eval = np.linspace(0, T_total, 500)
sol = solve_ivp(rhs_full, (0, T_total), [0, 0, sigma0, 0],
                t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(sol.t * 1000, np.degrees(sol.y[0]), 'b-', lw=2, label='phi')
axes[0].set_ylabel('phi [deg]')
axes[0].legend()
axes[0].set_title(f'Full coupled: eps={eps_demo}, T_f={Tf_demo}s, T_w={Tw_demo}s')
axes[0].axhline(0, color='gray', ls='--', alpha=0.5)
for xv in [Tf_demo/2, Tf_demo, Tf_demo+Tw_demo, Tf_demo+Tw_demo+Tf_demo/2]:
    axes[0].axvline(xv*1000, color='gray', ls=':', alpha=0.4)

beta_ts = sol.y[2] / (Mt * h_com)
axes[1].plot(sol.t * 1000, np.degrees(beta_ts), 'r-', lw=2, label='beta')
axes[1].axhline(np.degrees(beta0), color='gray', ls='--', alpha=0.5, label='beta0')
axes[1].axhline(0, color='gray', ls='--', alpha=0.3)
axes[1].set_xlabel('시간 [ms]')
axes[1].set_ylabel('β [deg]')
axes[1].legend()

plt.tight_layout()
ts_path = __file__.replace('.py', '_timeseries.png')
plt.savefig(ts_path, dpi=150)
print(f"Saved: {ts_path}")
plt.close()

print("\nDone!")

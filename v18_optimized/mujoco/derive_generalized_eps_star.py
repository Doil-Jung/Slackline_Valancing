# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
일반화 ε* 유도: 비정지 초기조건에서의 안정 모드 진입 조건
===========================================================
핵심: c₂ + d₂ = 0 (발산 지수항 A=0)
→ 불안정 모드가 e^{-λt}로 자연 감쇠
→ ε 하나로 달성 가능

검증:
  1. 정지 출발 환원: ε*_gen = ε* = -h/(r·R+h)
  2. 해석적 궤적: ε*_gen에서 불안정 모드 감쇠 확인
  3. 비정지 초기조건별 ε*_gen 테이블
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import os

# ============================================================
# 파라미터 (batch_experiment.py와 동일)
# ============================================================
GRAV = 9.81
M1 = 30.0; M2 = 40.0; L1 = 0.85; L2 = 0.80
M_HEAD = 5.0; HEAD_R = L2/4
LC1 = L1/2; I1 = M1*L1**2/12

def compute_effective_upper():
    m2 = M2; m_h = M_HEAD
    d2 = L2/2; d_h = L2 + HEAD_R
    m_tot = m2 + m_h
    lc2 = (m2*d2 + m_h*d_h) / m_tot
    i2_body = m2*L2**2/12 + m2*(d2-lc2)**2
    i2_head = 2/5*m_h*HEAD_R**2 + m_h*(d_h-lc2)**2
    return m_tot, lc2, i2_body + i2_head

M2_EFF, LC2_EFF, I2_EFF = compute_effective_upper()
g = GRAV; m2 = M2_EFF; Mt = M1 + m2
p1 = M1*LC1 + m2*L1; p2 = m2*LC2_EFF; p3 = m2*L1*LC2_EFF
ps = p1 + p2; h = ps / Mt
J_aa = M1*LC1**2 + I1 + m2*L1**2
J_tt = m2*LC2_EFF**2 + I2_EFF; J_at = p3

# 좌표 변환 행렬 (φ,α,θ) → (φ,β,δ)
T_inv = np.array([[1, 0, 0],
                   [0, Mt*h/ps, -p2/ps],
                   [0, Mt*h/ps,  p1/ps]])

print("=" * 70)
print("  일반화 ε* 유도: c₂ + d₂ = 0 조건")
print("=" * 70)
print(f"\n  시스템 파라미터:")
print(f"    M1={M1}, M2_EFF={M2_EFF:.1f}, Mt={Mt:.1f} kg")
print(f"    LC1={LC1:.3f}, LC2_EFF={LC2_EFF:.3f} m")
print(f"    p1={p1:.2f}, p2={p2:.2f}, ps={ps:.2f}")
print(f"    h = ps/Mt = {h:.4f} m")

# ============================================================
# 모드 분석 함수
# ============================================================
def mode_analysis(R):
    """2×2 결합 모드 분석 (δ 고정)"""
    M = np.array([[Mt*R**2, R*p1, R*p2],
                   [R*p1, J_aa, J_at],
                   [R*p2, J_at, J_tt]])
    G = np.diag([-Mt*g*R, p1*g, p2*g])
    M_pbd = T_inv.T @ M @ T_inv
    G_pbd = T_inv.T @ G @ T_inv
    M22 = M_pbd[:2, :2]
    G22 = G_pbd[:2, :2]
    A2 = np.linalg.solve(M22, G22)
    
    ev, V = np.linalg.eig(A2)
    idx = np.argsort(ev.real)
    ev = ev[idx].real
    V = V[:, idx].real
    
    # 안정 모드 (λ₁ < 0), 불안정 모드 (λ₂ > 0)
    omega_eff = np.sqrt(abs(ev[0]))
    lambda_eff = np.sqrt(abs(ev[1]))
    
    v1 = V[:, 0]  # 안정 모드 고유벡터
    v2 = V[:, 1]  # 불안정 모드 고유벡터
    
    # 왼쪽 고유벡터 (V⁻¹의 행)
    V_inv = np.linalg.inv(V)
    u1 = V_inv[0, :]  # u₁ᵀ
    u2 = V_inv[1, :]  # u₂ᵀ
    
    return omega_eff, lambda_eff, v1, v2, u1, u2, A2

# ============================================================
# ε* 공식들
# ============================================================
def eps_star_stationary(R):
    """기존 ε* (정지 출발): c₂ = 0"""
    omega, lam, v1, v2, u1, u2, _ = mode_analysis(R)
    r = v1[0] / v1[1]
    return -h / (r*R + h)

def eps_star_general(R, phi_i, beta_i, phi_dot_i, beta_dot_i):
    """일반화 ε*: c₂ + d₂ = 0 (발산 지수항 제거)
    
    c₂ = u₂ᵀ · q₀(ε)
    d₂ = u₂ᵀ · q̇₀ / λ_eff
    
    c₂ + d₂ = 0:
    u₂ᵀ · [q₀(ε) + q̇₀/λ_eff] = 0
    """
    omega, lam, v1, v2, u1, u2, _ = mode_analysis(R)
    
    # q_pred = q₀_base + q̇₀/λ_eff  (속도를 유효 거리로 환산)
    q_pred = np.array([
        phi_i + h*beta_i/R + phi_dot_i/lam,
        beta_dot_i/lam
    ])
    
    # 접기 방향 벡터
    e = np.array([h/R, -1.0])
    
    # ε*_gen = -u₂ᵀ·q_pred / (βᵢ·u₂ᵀ·e)
    numerator = u2 @ q_pred
    denominator = beta_i * (u2 @ e)
    
    if abs(denominator) < 1e-15:
        return np.nan
    
    return -numerator / denominator

# ============================================================
# 해석적 궤적 계산
# ============================================================
def compute_trajectory(R, eps, phi_i, beta_i, phi_dot_i, beta_dot_i, t_max=2.0, dt=0.001):
    """
    Wait 단계 해석적 궤적 (가설 관례: x_foot = +R·φ)
    접기 후 초기조건 → 모드 분해 → 시간 전개
    """
    omega, lam, v1, v2, u1, u2, A2 = mode_analysis(R)
    V = np.column_stack([v1, v2])
    V_inv = np.linalg.inv(V)
    
    # 접기 후 초기조건
    phi_0 = phi_i + h*(1+eps)*beta_i/R
    beta_0 = -eps*beta_i
    q0 = np.array([phi_0, beta_0])
    q0_dot = np.array([phi_dot_i, beta_dot_i])
    
    # 모드 계수
    modal_pos = V_inv @ q0       # [c₁, c₂]
    modal_vel = V_inv @ q0_dot   # [c₁_dot, c₂_dot]
    
    c1, c2 = modal_pos
    d1 = modal_vel[0] / omega    # η̇₁(0) = d₁·ω_eff
    d2 = modal_vel[1] / lam      # η̇₂(0) = d₂·λ_eff
    
    # 발산 계수 A = (c₂+d₂)/2
    A_unstable = (c2 + d2) / 2
    B_decay = (c2 - d2) / 2
    
    # 시간 배열
    t = np.arange(0, t_max, dt)
    
    # 모드별 시간 전개
    eta1 = c1*np.cos(omega*t) + d1*np.sin(omega*t)
    eta2 = c2*np.cosh(lam*t) + d2*np.sinh(lam*t)
    
    # 물리 좌표
    phi = v1[0]*eta1 + v2[0]*eta2
    beta = v1[1]*eta1 + v2[1]*eta2
    
    # x_CoM (가설 관례: x_CoM = R·φ + h·β)
    x_com = R*phi + h*beta
    
    return {
        't': t, 'phi': phi, 'beta': beta, 'x_com': x_com,
        'eta1': eta1, 'eta2': eta2,
        'c1': c1, 'c2': c2, 'd1': d1, 'd2': d2,
        'A_unstable': A_unstable, 'B_decay': B_decay,
        'omega': omega, 'lambda': lam
    }

# ============================================================
# 테스트 1: 정지 출발 환원 검증
# ============================================================
print("\n" + "=" * 70)
print("  테스트 1: 정지 출발 환원 (ε*_gen → ε*)")
print("=" * 70)
print(f"\n  {'R':>5} | {'ε*_stat':>10} | {'ε*_gen':>10} | {'차이':>12} | {'ω_eff':>8} | {'λ_eff':>8}")
print(f"  {'-'*70}")

for R in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    es = eps_star_stationary(R)
    eg = eps_star_general(R, phi_i=0, beta_i=np.radians(2),
                          phi_dot_i=0, beta_dot_i=0)
    omega, lam, _, _, _, _, _ = mode_analysis(R)
    diff = abs(eg - es)
    print(f"  {R:5.1f} | {es:10.6f} | {eg:10.6f} | {diff:12.2e} | {omega:8.3f} | {lam:8.3f}")

# ============================================================
# 테스트 2: 비정지 초기조건별 ε*_gen
# ============================================================
print("\n" + "=" * 70)
print("  테스트 2: 비정지 초기조건별 ε*_gen (R=0.7)")
print("=" * 70)

R_test = 0.7
beta_i = np.radians(2)
omega_t, lam_t, v1_t, v2_t, u1_t, u2_t, _ = mode_analysis(R_test)

print(f"\n  R={R_test}m, β_i={np.degrees(beta_i):.0f}°")
print(f"  ε*_stat = {eps_star_stationary(R_test):.4f}")
print(f"  ω_eff = {omega_t:.3f} rad/s, λ_eff = {lam_t:.3f} rad/s")
print(f"  감쇠 시간 1/λ_eff = {1/lam_t:.3f} s")
print()

print(f"  {'φ_i[°]':>7} {'β̇_i[°/s]':>10} {'φ̇_i[°/s]':>10} | {'ε*_gen':>8} | {'Δε':>8} | {'c₂':>10} {'d₂':>10} {'c₂+d₂':>10}")
print(f"  {'-'*90}")

test_cases = [
    # (phi_i_deg, beta_dot_deg_s, phi_dot_deg_s)
    (0,   0,   0),    # 정지 출발 (기준)
    (0,  10,   0),    # β 각속도만
    (0, -10,   0),    # β 역방향 각속도
    (0,   0,  30),    # φ 각속도만
    (0,   0, -30),    # φ 역방향 각속도
    (5,   0,   0),    # φ 변위만
    (-5,  0,   0),    # φ 역방향 변위
    (5,  10,  30),    # 모든 비정지
    (5, -10, -30),    # 모든 역방향
]

es_ref = eps_star_stationary(R_test)

for phi_deg, bdot_deg, pdot_deg in test_cases:
    phi_i = np.radians(phi_deg)
    bdot_i = np.radians(bdot_deg)
    pdot_i = np.radians(pdot_deg)
    
    eg = eps_star_general(R_test, phi_i, beta_i, pdot_i, bdot_i)
    
    # 모드 계수 확인
    traj = compute_trajectory(R_test, eg, phi_i, beta_i, pdot_i, bdot_i, t_max=0.01)
    
    print(f"  {phi_deg:7.1f} {bdot_deg:10.1f} {pdot_deg:10.1f} | "
          f"{eg:8.4f} | {eg-es_ref:+8.4f} | "
          f"{traj['c2']:10.2e} {traj['d2']:10.2e} {traj['c2']+traj['d2']:10.2e}")

# ============================================================
# 테스트 3: 해석적 궤적 비교 그래프
# ============================================================
print("\n" + "=" * 70)
print("  테스트 3: 해석적 궤적 비교 그래프 생성")
print("=" * 70)

# 비정지 초기조건 설정
R_plot = 0.7
phi_i_plot = np.radians(3)
beta_i_plot = np.radians(2)
phi_dot_plot = np.radians(20)
beta_dot_plot = np.radians(10)

es_stat = eps_star_stationary(R_plot)
es_gen = eps_star_general(R_plot, phi_i_plot, beta_i_plot, phi_dot_plot, beta_dot_plot)
es_wrong = es_stat  # 정지 ε*를 비정지에 적용 (잘못된 선택)
es_arb = 1.0  # 임의 ε

print(f"  R={R_plot}m")
print(f"  초기조건: φ={np.degrees(phi_i_plot):.1f}°, β={np.degrees(beta_i_plot):.1f}°, "
      f"φ̇={np.degrees(phi_dot_plot):.1f}°/s, β̇={np.degrees(beta_dot_plot):.1f}°/s")
print(f"  ε*_stat = {es_stat:.4f}")
print(f"  ε*_gen  = {es_gen:.4f}")
print(f"  ε_arb   = {es_arb:.4f}")

# 궤적 계산
cases = {
    f'ε*_gen={es_gen:.3f} (c₂+d₂=0)': (es_gen, '#E63946', 2.5, '-'),
    f'ε*_stat={es_stat:.3f} (정지ε*)': (es_wrong, '#457B9D', 2.0, '--'),
    f'ε={es_arb:.1f} (임의)': (es_arb, '#264653', 1.5, ':'),
}

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

for label, (eps_val, color, lw, ls) in cases.items():
    traj = compute_trajectory(R_plot, eps_val, phi_i_plot, beta_i_plot,
                              phi_dot_plot, beta_dot_plot, t_max=2.0)
    t = traj['t']
    
    # x_CoM
    axes[0, 0].plot(t, traj['x_com']*1000, color=color, lw=lw, ls=ls, label=label)
    # φ
    axes[1, 0].plot(t, np.degrees(traj['phi']), color=color, lw=lw, ls=ls, label=label)
    # β
    axes[2, 0].plot(t, np.degrees(traj['beta']), color=color, lw=lw, ls=ls, label=label)
    
    # η₂ (불안정 모드)
    axes[0, 1].plot(t, traj['eta2'], color=color, lw=lw, ls=ls, label=label)
    # η₁ (안정 모드)
    axes[1, 1].plot(t, traj['eta1'], color=color, lw=lw, ls=ls, label=label)
    
    # 발산 계수
    A_val = traj['A_unstable']
    B_val = traj['B_decay']
    # η₂ 분해: A·e^{+λt} + B·e^{-λt}
    eta2_grow = A_val * np.exp(traj['lambda']*t)
    eta2_decay = B_val * np.exp(-traj['lambda']*t)
    if 'gen' in label:
        axes[2, 1].plot(t, eta2_grow, color=color, lw=1.5, ls='-', label=f'A·e^{{+λt}} (A={A_val:.2e})')
        axes[2, 1].plot(t, eta2_decay, color=color, lw=1.5, ls='--', label=f'B·e^{{-λt}} (B={B_val:.2e})')

# 축 설정
titles_left = ['x_CoM [mm]', 'φ [deg]', 'β [deg]']
titles_right = ['η₂ (불안정 모드)', 'η₁ (안정 모드)', 'η₂ 분해: A·e^{+λt} + B·e^{-λt}']

for i in range(3):
    axes[i, 0].set_ylabel(titles_left[i], fontsize=11)
    axes[i, 1].set_ylabel(titles_right[i], fontsize=11)
    for j in range(2):
        axes[i, j].grid(True, alpha=0.2)
        axes[i, j].axhline(0, color='k', lw=0.3)
        axes[i, j].legend(fontsize=8, loc='best')
        if i == 2:
            axes[i, j].set_xlabel('시간 [s]', fontsize=11)

axes[0, 0].set_ylim(-300, 300)

fig.suptitle(f'일반화 ε* 해석적 궤적 비교 (R={R_plot}m)\n'
             f'초기: φ={np.degrees(phi_i_plot):.0f}°, β={np.degrees(beta_i_plot):.0f}°, '
             f'φ̇={np.degrees(phi_dot_plot):.0f}°/s, β̇={np.degrees(beta_dot_plot):.0f}°/s',
             fontsize=13, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'generalized_eps_star_analytical.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  그래프 저장: {save_path}")

# ============================================================
# 테스트 4: R별 ε*_gen 테이블 (비정지 대표 케이스)
# ============================================================
print("\n" + "=" * 70)
print("  테스트 4: R별 ε*_gen (비정지 대표: φ=3°, β̇=10°/s, φ̇=20°/s)")
print("=" * 70)

phi_i_t4 = np.radians(3)
beta_i_t4 = np.radians(2)
pdot_t4 = np.radians(20)
bdot_t4 = np.radians(10)

print(f"\n  {'R':>5} | {'ε*_stat':>8} | {'ε*_gen':>8} | {'Δε':>8} | {'A(stat)':>10} | {'A(gen)':>10} | {'감쇠τ':>7}")
print(f"  {'-'*75}")

for R in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]:
    es = eps_star_stationary(R)
    eg = eps_star_general(R, phi_i_t4, beta_i_t4, pdot_t4, bdot_t4)
    _, lam_R, _, _, _, _, _ = mode_analysis(R)
    
    # 발산 계수 비교
    traj_s = compute_trajectory(R, es, phi_i_t4, beta_i_t4, pdot_t4, bdot_t4, t_max=0.01)
    traj_g = compute_trajectory(R, eg, phi_i_t4, beta_i_t4, pdot_t4, bdot_t4, t_max=0.01)
    
    print(f"  {R:5.1f} | {es:8.4f} | {eg:8.4f} | {eg-es:+8.4f} | "
          f"{traj_s['A_unstable']:10.2e} | {traj_g['A_unstable']:10.2e} | "
          f"{1/lam_R:7.3f}s")

# ============================================================
# 테스트 5: 자기일관성 체크 — ε* 진동 중 상태에서 ε*_gen
# ============================================================
print("\n" + "=" * 70)
print("  테스트 5: 자기일관성 (ε* 진동 중 → extend → ε*_gen)")
print("=" * 70)

R_sc = 0.7
es_sc = eps_star_stationary(R_sc)
beta0_sc = np.radians(2)
omega_sc, lam_sc, v1_sc, v2_sc, _, _, _ = mode_analysis(R_sc)

print(f"\n  R={R_sc}m, β₀=2°, ε*={es_sc:.4f}")
print(f"  사이클1: 정지 출발 + ε* → 안정 진동")

# 사이클 1: 정지 출발
traj1 = compute_trajectory(R_sc, es_sc, 0, beta0_sc, 0, 0, t_max=2.0)

print(f"    c₂ = {traj1['c2']:.2e}, d₂ = {traj1['d2']:.2e}, A = {traj1['A_unstable']:.2e}")
print(f"    ω_eff = {omega_sc:.3f} rad/s, T_진동 = {2*np.pi/omega_sc:.3f} s")

# 다양한 extend 시점에서 자기일관성 확인
print(f"\n  extend 시점별 ε*_gen 및 발산 계수:")
print(f"  {'T_ext':>7} | {'φ_ext':>8} | {'β_ext':>8} | {'φ̇_ext':>9} | {'β̇_ext':>9} | {'ε*_gen':>8} | {'A_gen':>10}")
print(f"  {'-'*80}")

for T_ext_frac in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    T_ext = T_ext_frac * (2*np.pi/omega_sc)  # 진동 주기의 일부
    idx = int(T_ext / 0.001)
    if idx >= len(traj1['t']):
        idx = len(traj1['t']) - 1
    
    phi_ext = traj1['phi'][idx]
    beta_ext = traj1['beta'][idx]
    
    # 속도 (유한 차분 근사)
    if idx > 0:
        phi_dot_ext = (traj1['phi'][idx] - traj1['phi'][idx-1]) / 0.001
        beta_dot_ext = (traj1['beta'][idx] - traj1['beta'][idx-1]) / 0.001
    else:
        phi_dot_ext = 0
        beta_dot_ext = 0
    
    # Extend: β → 0, φ → x_CoM/R
    x_com_ext = R_sc*phi_ext + h*beta_ext
    phi_after_ext = x_com_ext / R_sc
    beta_after_ext = 0.0
    # 속도는 불변
    
    # 다음 사이클: extend 후 상태가 β=0이므로 새 기울기가 필요
    # 실제로는 β≠0인 시점에서 접기 시작 — β_after_ext가 0이면 접기 의미 없음
    # → x_CoM ≠ 0이면 φ ≠ 0, 이 상태에서 β가 빠르게 변할 것
    
    # 여기서는 extend를 "β를 0으로" 가 아니라 "δ를 되돌리기"로 해석
    # extend 후: 새로운 (φ, β)에서 β ≠ 0인 상태
    # 실제로는 extend가 β를 바꾸지 않고 δ만 바꿈
    # 단순화: extend 후 바로 β_ext 값으로 다시 접기
    
    # β_ext ≠ 0인 경우만 의미있음
    if abs(beta_ext) < 1e-10:
        print(f"  {T_ext:7.3f} | {np.degrees(phi_ext):8.3f} | {np.degrees(beta_ext):8.5f} | "
              f"{np.degrees(phi_dot_ext):9.3f} | {np.degrees(beta_dot_ext):9.3f} | {'β≈0 skip':>8} | {'—':>10}")
        continue
    
    # ε*_gen: extend 시점의 상태에서 바로 접기
    eg_sc = eps_star_general(R_sc, phi_ext, beta_ext, phi_dot_ext, beta_dot_ext)
    traj2 = compute_trajectory(R_sc, eg_sc, phi_ext, beta_ext,
                                phi_dot_ext, beta_dot_ext, t_max=0.01)
    
    print(f"  {T_ext:7.3f} | {np.degrees(phi_ext):8.3f} | {np.degrees(beta_ext):8.5f} | "
          f"{np.degrees(phi_dot_ext):9.3f} | {np.degrees(beta_dot_ext):9.3f} | "
          f"{eg_sc:8.4f} | {traj2['A_unstable']:10.2e}")

# ============================================================
# 자기일관성 궤적 그래프
# ============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# 사이클1: 정지 출발 + ε*
T_ext_pick = 0.3 * (2*np.pi/omega_sc)
idx_pick = min(int(T_ext_pick / 0.001), len(traj1['t'])-1)

# Extend 시점의 상태
phi_e = traj1['phi'][idx_pick]
beta_e = traj1['beta'][idx_pick]
if idx_pick > 0:
    pdot_e = (traj1['phi'][idx_pick] - traj1['phi'][idx_pick-1]) / 0.001
    bdot_e = (traj1['beta'][idx_pick] - traj1['beta'][idx_pick-1]) / 0.001
else:
    pdot_e = 0; bdot_e = 0

eg_pick = eps_star_general(R_sc, phi_e, beta_e, pdot_e, bdot_e)

# 사이클2: ε*_gen vs ε*_stat
traj2_gen = compute_trajectory(R_sc, eg_pick, phi_e, beta_e, pdot_e, bdot_e, t_max=2.0)
traj2_stat = compute_trajectory(R_sc, es_sc, phi_e, beta_e, pdot_e, bdot_e, t_max=2.0)

t1 = traj1['t'][:idx_pick]
t2_gen = traj2_gen['t'] + T_ext_pick
t2_stat = traj2_stat['t'] + T_ext_pick

# 플롯
ax = axes2[0, 0]
ax.plot(t1, traj1['x_com'][:idx_pick]*1000, 'k-', lw=2, label='사이클1 (ε*)')
ax.axvline(T_ext_pick, color='gray', ls=':', lw=1, label=f'Extend (T={T_ext_pick:.2f}s)')
ax.plot(t2_gen, traj2_gen['x_com']*1000, '#E63946', lw=2, label=f'사이클2 (ε*_gen={eg_pick:.3f})')
ax.plot(t2_stat, traj2_stat['x_com']*1000, '#457B9D', lw=2, ls='--', label=f'사이클2 (ε*_stat={es_sc:.3f})')
ax.set_ylabel('x_CoM [mm]'); ax.set_ylim(-300, 300)
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
ax.set_title('x_CoM: 2사이클 자기일관성', fontweight='bold')

ax = axes2[0, 1]
ax.plot(t1, traj1['eta2'][:idx_pick], 'k-', lw=2, label='사이클1 η₂')
ax.axvline(T_ext_pick, color='gray', ls=':', lw=1)
ax.plot(t2_gen, traj2_gen['eta2'], '#E63946', lw=2, label=f'ε*_gen (A={traj2_gen["A_unstable"]:.2e})')
ax.plot(t2_stat, traj2_stat['eta2'], '#457B9D', lw=2, ls='--', label=f'ε*_stat (A={traj2_stat["A_unstable"]:.2e})')
ax.set_ylabel('η₂ (불안정 모드)'); ax.legend(fontsize=8)
ax.grid(True, alpha=0.2); ax.set_title('불안정 모드: ε*_gen 감쇠 vs ε*_stat 발산', fontweight='bold')

ax = axes2[1, 0]
ax.plot(t1, np.degrees(traj1['phi'][:idx_pick]), 'k-', lw=2, label='사이클1 φ')
ax.axvline(T_ext_pick, color='gray', ls=':', lw=1)
ax.plot(t2_gen, np.degrees(traj2_gen['phi']), '#E63946', lw=2, label='ε*_gen')
ax.plot(t2_stat, np.degrees(traj2_stat['phi']), '#457B9D', lw=2, ls='--', label='ε*_stat')
ax.set_ylabel('φ [deg]'); ax.set_xlabel('시간 [s]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

ax = axes2[1, 1]
ax.plot(t1, np.degrees(traj1['beta'][:idx_pick]), 'k-', lw=2, label='사이클1 β')
ax.axvline(T_ext_pick, color='gray', ls=':', lw=1)
ax.plot(t2_gen, np.degrees(traj2_gen['beta']), '#E63946', lw=2, label='ε*_gen')
ax.plot(t2_stat, np.degrees(traj2_stat['beta']), '#457B9D', lw=2, ls='--', label='ε*_stat')
ax.set_ylabel('β [deg]'); ax.set_xlabel('시간 [s]')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

fig2.suptitle(f'자기일관성 테스트 (R={R_sc}m)\n'
              f'사이클1: 정지→ε* | Extend | 사이클2: ε*_gen vs ε*_stat',
              fontsize=13, fontweight='bold')
plt.tight_layout()

save_path2 = os.path.join(os.path.dirname(__file__), '..', 'docs', 'generalized_eps_star_self_consistency.png')
plt.savefig(save_path2, dpi=150, bbox_inches='tight')
print(f"  자기일관성 그래프 저장: {save_path2}")

print("\n" + "=" * 70)
print("  Phase 1 완료!")
print("=" * 70)

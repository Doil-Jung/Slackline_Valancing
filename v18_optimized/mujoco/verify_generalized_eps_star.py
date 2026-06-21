# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
일반화 ε*_gen MuJoCo 검증
===========================
소각근사 유효 범위 내의 작은 비정지 초기조건에서:
  - ε*_gen (c₂+d₂=0) 적용 → 안정 진동 확인
  - ε*_stat (정지 ε*) 적용 → 발산 확인 (대조군)
  - 해석 궤적 vs MuJoCo 비교
"""
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (
    generate_mjcf, get_state,
    M1, M2_EFF, LC1, LC2_EFF, I2_EFF, L1, GRAV, DT
)

# ============================================================
# 파라미터 (derive 스크립트와 동일)
# ============================================================
g = GRAV; m2 = M2_EFF; Mt = M1 + m2; I1 = M1*L1**2/12
p1 = M1*LC1 + m2*L1; p2 = m2*LC2_EFF; p3 = m2*L1*LC2_EFF
ps = p1 + p2; h = ps / Mt
T_inv = np.array([[1,0,0],[0,Mt*h/ps,-p2/ps],[0,Mt*h/ps,p1/ps]])
J_aa = M1*LC1**2 + I1 + m2*L1**2
J_tt = m2*LC2_EFF**2 + I2_EFF; J_at = p3

def mode_analysis(R):
    M = np.array([[Mt*R**2,R*p1,R*p2],[R*p1,J_aa,J_at],[R*p2,J_at,J_tt]])
    G = np.diag([-Mt*g*R, p1*g, p2*g])
    Mp = T_inv.T @ M @ T_inv; Gp = T_inv.T @ G @ T_inv
    A2 = np.linalg.solve(Mp[:2,:2], Gp[:2,:2])
    ev, V = np.linalg.eig(A2)
    idx = np.argsort(ev.real); ev = ev[idx].real; V = V[:,idx].real
    V_inv = np.linalg.inv(V)
    return np.sqrt(abs(ev[0])), np.sqrt(abs(ev[1])), V[:,0], V[:,1], V_inv[0], V_inv[1], V, V_inv

def eps_star_stat(R):
    omega, lam, v1, v2, u1, u2, V, Vi = mode_analysis(R)
    r = v1[0]/v1[1]
    return -h/(r*R+h)

def eps_star_gen(R, phi_i, beta_i, phi_dot_i, beta_dot_i):
    omega, lam, v1, v2, u1, u2, V, Vi = mode_analysis(R)
    q_pred = np.array([phi_i + h*beta_i/R + phi_dot_i/lam, beta_dot_i/lam])
    e = np.array([h/R, -1.0])
    num = u2 @ q_pred; den = beta_i * (u2 @ e)
    if abs(den) < 1e-15: return np.nan
    return -num / den

def analytical_trajectory(R, eps, phi_i, beta_i, pdot_i, bdot_i, t_max=2.0, dt=0.001):
    omega, lam, v1, v2, u1, u2, V, Vi = mode_analysis(R)
    phi0 = phi_i + h*(1+eps)*beta_i/R
    beta0 = -eps*beta_i
    q0 = np.array([phi0, beta0]); qd0 = np.array([pdot_i, bdot_i])
    mp = Vi @ q0; mv = Vi @ qd0
    c1,c2 = mp; d1 = mv[0]/omega; d2 = mv[1]/lam
    t = np.arange(0, t_max, dt)
    eta1 = c1*np.cos(omega*t) + d1*np.sin(omega*t)
    eta2 = c2*np.cosh(lam*t) + d2*np.sinh(lam*t)
    phi = v1[0]*eta1 + v2[0]*eta2
    beta = v1[1]*eta1 + v2[1]*eta2
    xcom = R*phi + h*beta
    return t, xcom, phi, beta, c2, d2, (c2+d2)/2

# ============================================================
# MuJoCo 시뮬레이션 (비정지 초기조건)
# ============================================================
def run_wait_nonstationary(R, eps, phi_i_hyp, beta_i, pdot_i_hyp, bdot_i,
                           sim_time=2.0, Kp=10000, Kd=200):
    """
    비정지 초기조건에서 Wait 단계 시뮬레이션.
    phi_i_hyp, pdot_i_hyp: 가설 관례 (x_foot = +R·φ)
    MuJoCo 관례: x_foot = -R·φ_mj → φ_mj = -φ_hyp
    """
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 관절 주소
    ra = model.jnt_qposadr[model.joint('phi_Y').id]
    aa = model.jnt_qposadr[model.joint('ankle').id]
    ha = model.jnt_qposadr[model.joint('hip').id]
    try: rb = model.jnt_qposadr[model.joint('rope_b_Y').id]
    except: rb = None

    # 속도 주소
    va = model.jnt_dofadr[model.joint('phi_Y').id]
    va_ankle = model.jnt_dofadr[model.joint('ankle').id]
    va_hip = model.jnt_dofadr[model.joint('hip').id]
    try: vb = model.jnt_dofadr[model.joint('rope_b_Y').id]
    except: vb = None

    # --- 접기 후 초기조건 (가설 관례 → MuJoCo 관례) ---
    phi_after_hyp = phi_i_hyp + h*(1+eps)*beta_i/R
    beta_after = -eps*beta_i
    
    # MuJoCo 관례: φ_mj = -φ_hyp
    phi_mj = -phi_after_hyp
    
    # δ = 0이므로: α = β, θ = α
    alpha_after = beta_after
    
    # 위치 설정
    data.qpos[ra] = phi_mj
    data.qpos[aa] = alpha_after - phi_mj   # ankle = α_abs - φ_mj
    data.qpos[ha] = 0                       # δ = 0
    if rb is not None:
        data.qpos[rb] = phi_mj              # rope_b 동기화

    # --- 속도 설정 ---
    phi_dot_mj = -pdot_i_hyp              # MuJoCo 관례
    alpha_dot = bdot_i                     # δ̇=0이므로 α̇ = β̇
    
    data.qvel[va] = phi_dot_mj
    data.qvel[va_ankle] = alpha_dot - phi_dot_mj  # ankle_vel = α̇ - φ̇_mj
    data.qvel[va_hip] = 0                           # δ̇ = 0
    if vb is not None:
        data.qvel[vb] = phi_dot_mj

    mujoco.mj_forward(model, data)

    # --- 시뮬레이션 ---
    n = int(sim_time / DT)
    times = []; xcoms = []; phis = []; betas = []

    for i in range(n):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > np.radians(60):
            break
        phi = x[0]; alpha = x[1]; theta = x[2]
        delta = theta - alpha; delta_d = x[5] - x[4]
        tau = np.clip(Kp*(0-delta) - Kd*delta_d, -50000, 50000)
        data.ctrl[0] = tau

        if i % 10 == 0:
            beta_val = (p1*alpha + p2*theta)/(Mt*h)
            xcom = -R*phi + h*beta_val
            times.append(data.time)
            xcoms.append(xcom)
            phis.append(-phi)  # 가설 관례로 변환
            betas.append(beta_val)

        mujoco.mj_step(model, data)

    return np.array(times), np.array(xcoms), np.array(phis), np.array(betas)

# ============================================================
# 테스트 실행
# ============================================================
print("=" * 70)
print("  일반화 ε*_gen MuJoCo 검증")
print("=" * 70)

# --- 소각 근사 범위 내 초기조건들 ---
test_configs = [
    # (label, R, phi_i_deg, beta_i_deg, pdot_deg_s, bdot_deg_s)
    ("A: φ만 비정지",      0.7, 1.0, 2.0,  0.0,  0.0),
    ("B: β̇만 비정지",     0.7, 0.0, 2.0,  0.0,  3.0),
    ("C: φ̇만 비정지",     0.7, 0.0, 2.0,  5.0,  0.0),
    ("D: 소량 비정지",     0.7, 0.5, 2.0,  3.0,  2.0),
    ("E: R=1.0 소량",      1.0, 0.5, 2.0,  3.0,  2.0),
    ("F: R=0.5 소량",      0.5, 0.3, 2.0,  2.0,  1.0),
]

# 먼저 ε*_gen 값 출력
print(f"\n  {'케이스':>18} | {'R':>4} | {'φ_i':>5} {'β_i':>5} {'φ̇_i':>5} {'β̇_i':>5} | {'ε*_s':>6} {'ε*_g':>6} {'Δε':>7} | {'φ_max°':>7}")
print(f"  {'-'*95}")

for label, R, pd, bd, pdd, bdd in test_configs:
    pi = np.radians(pd); bi = np.radians(bd)
    pdi = np.radians(pdd); bdi = np.radians(bdd)
    es = eps_star_stat(R)
    eg = eps_star_gen(R, pi, bi, pdi, bdi)
    # 접기 후 φ (가설) — 소각 범위 체크
    phi_after = pi + h*(1+eg)*bi/R
    print(f"  {label:>18} | {R:4.1f} | {pd:5.1f} {bd:5.1f} {pdd:5.1f} {bdd:5.1f} | "
          f"{es:6.3f} {eg:6.3f} {eg-es:+7.3f} | {np.degrees(phi_after):7.2f}")

# ============================================================
# MuJoCo 시뮬레이션 + 그래프
# ============================================================
n_tests = len(test_configs)
fig, axes = plt.subplots(n_tests, 3, figsize=(18, 4*n_tests), squeeze=False)

for idx, (label, R, pd, bd, pdd, bdd) in enumerate(test_configs):
    pi = np.radians(pd); bi = np.radians(bd)
    pdi = np.radians(pdd); bdi = np.radians(bdd)
    
    es = eps_star_stat(R)
    eg = eps_star_gen(R, pi, bi, pdi, bdi)
    
    print(f"\n  [{idx+1}/{n_tests}] {label}: R={R}, ε*_gen={eg:.4f}, ε*_stat={es:.4f}")
    
    # 해석 궤적
    t_an_g, xc_an_g, phi_an_g, beta_an_g, c2_g, d2_g, A_g = \
        analytical_trajectory(R, eg, pi, bi, pdi, bdi)
    t_an_s, xc_an_s, phi_an_s, beta_an_s, c2_s, d2_s, A_s = \
        analytical_trajectory(R, es, pi, bi, pdi, bdi)
    
    print(f"    해석 ε*_gen: A={A_g:.2e}, c₂={c2_g:.2e}, d₂={d2_g:.2e}")
    print(f"    해석 ε*_stat: A={A_s:.2e}, c₂={c2_s:.2e}, d₂={d2_s:.2e}")
    
    # MuJoCo 시뮬레이션
    t_mj_g, xc_mj_g, phi_mj_g, beta_mj_g = \
        run_wait_nonstationary(R, eg, pi, bi, pdi, bdi, sim_time=2.0)
    t_mj_s, xc_mj_s, phi_mj_s, beta_mj_s = \
        run_wait_nonstationary(R, es, pi, bi, pdi, bdi, sim_time=2.0)
    
    xmax_g = np.max(np.abs(xc_mj_g))*1000 if len(xc_mj_g) > 0 else 999
    xmax_s = np.max(np.abs(xc_mj_s))*1000 if len(xc_mj_s) > 0 else 999
    dur_g = t_mj_g[-1] if len(t_mj_g) > 0 else 0
    dur_s = t_mj_s[-1] if len(t_mj_s) > 0 else 0
    
    print(f"    MuJoCo ε*_gen: xmax={xmax_g:.1f}mm, 지속={dur_g:.2f}s")
    print(f"    MuJoCo ε*_stat: xmax={xmax_s:.1f}mm, 지속={dur_s:.2f}s")
    
    # --- x_CoM ---
    ax = axes[idx][0]
    ax.plot(t_an_g, xc_an_g*1000, '#E63946', lw=1, ls=':', alpha=0.7, label='해석 ε*_gen')
    ax.plot(t_mj_g, xc_mj_g*1000, '#E63946', lw=2, label=f'MuJoCo ε*_gen={eg:.3f}')
    ax.plot(t_an_s, xc_an_s*1000, '#457B9D', lw=1, ls=':', alpha=0.7, label='해석 ε*_stat')
    ax.plot(t_mj_s, xc_mj_s*1000, '#457B9D', lw=2, ls='--', label=f'MuJoCo ε*_stat={es:.3f}')
    ax.axhline(0, color='k', lw=0.3)
    ax.set_ylim(-200, 200)
    ax.set_ylabel(f'{label}\nx_CoM [mm]', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)
    if idx == 0: ax.set_title('x_CoM', fontsize=11, fontweight='bold')
    if idx == n_tests-1: ax.set_xlabel('시간 [s]')
    
    # --- φ ---
    ax = axes[idx][1]
    ax.plot(t_an_g, np.degrees(phi_an_g), '#E63946', lw=1, ls=':', alpha=0.7)
    ax.plot(t_mj_g, np.degrees(phi_mj_g), '#E63946', lw=2, label='ε*_gen')
    ax.plot(t_an_s, np.degrees(phi_an_s), '#457B9D', lw=1, ls=':', alpha=0.7)
    ax.plot(t_mj_s, np.degrees(phi_mj_s), '#457B9D', lw=2, ls='--', label='ε*_stat')
    ax.axhline(0, color='k', lw=0.3)
    ax.set_ylabel('φ [deg]', fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    if idx == 0: ax.set_title('φ (줄 각도)', fontsize=11, fontweight='bold')
    if idx == n_tests-1: ax.set_xlabel('시간 [s]')
    
    # --- β ---
    ax = axes[idx][2]
    ax.plot(t_an_g, np.degrees(beta_an_g), '#E63946', lw=1, ls=':', alpha=0.7)
    ax.plot(t_mj_g, np.degrees(beta_mj_g), '#E63946', lw=2, label='ε*_gen')
    ax.plot(t_an_s, np.degrees(beta_an_s), '#457B9D', lw=1, ls=':', alpha=0.7)
    ax.plot(t_mj_s, np.degrees(beta_mj_s), '#457B9D', lw=2, ls='--', label='ε*_stat')
    ax.axhline(0, color='k', lw=0.3)
    ax.set_ylabel('β [deg]', fontsize=9)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2)
    if idx == 0: ax.set_title('β (몸 기울기)', fontsize=11, fontweight='bold')
    if idx == n_tests-1: ax.set_xlabel('시간 [s]')

fig.suptitle('일반화 ε*_gen MuJoCo 검증\n'
             '빨간 실선: ε*_gen (c₂+d₂=0), 파란 점선: ε*_stat (정지 ε*)\n'
             '점선: 해석 궤적, 실선: MuJoCo',
             fontsize=13, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'generalized_eps_star_mujoco.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  그래프 저장: {save_path}")

# ============================================================
# T_cross 비교 (해석 vs MuJoCo, ε*_gen)
# ============================================================
print("\n" + "=" * 70)
print("  해석 vs MuJoCo 정량 비교 (ε*_gen)")
print("=" * 70)

print(f"\n  {'케이스':>18} | {'ε*_gen':>7} | {'해석 xmax':>10} | {'MJ xmax':>10} | {'MJ 지속':>7} | {'안정?':>5}")
print(f"  {'-'*80}")

for label, R, pd, bd, pdd, bdd in test_configs:
    pi = np.radians(pd); bi = np.radians(bd)
    pdi = np.radians(pdd); bdi = np.radians(bdd)
    eg = eps_star_gen(R, pi, bi, pdi, bdi)
    
    _, xc_an, _, _, _, _, _ = analytical_trajectory(R, eg, pi, bi, pdi, bdi)
    t_mj, xc_mj, _, _ = run_wait_nonstationary(R, eg, pi, bi, pdi, bdi)
    
    xmax_an = np.max(np.abs(xc_an))*1000
    xmax_mj = np.max(np.abs(xc_mj))*1000 if len(xc_mj) > 0 else 999
    dur = t_mj[-1] if len(t_mj) > 0 else 0
    stable = "✓" if dur >= 1.95 and xmax_mj < 200 else "✗"
    
    print(f"  {label:>18} | {eg:7.3f} | {xmax_an:8.1f}mm | {xmax_mj:8.1f}mm | {dur:5.2f}s | {stable:>5}")

print("\n  완료!")

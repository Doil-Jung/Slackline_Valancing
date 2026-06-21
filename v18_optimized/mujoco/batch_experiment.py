# -*- coding: utf-8 -*-
"""
MuJoCo 배치 실험: V18 최적화 형상 (H=80cm, 3kg)
===============================================
v14_mujoco.py의 모델/LQR을 재사용하여 headless 배치 실행.

실험 목록:
  A1: δ_peak vs β₀ (비례 제어 확인) — 여러 R
  A2: ε_LQR vs R
  A3: β₀ 소거 확인 (δ/β₀ 비율 일정성)
  B1: R 스윕 → 수렴 시간
  B2: R 스윕 → 최대 토크
  B3: R 스윕 → 안정 영역 (최대 복원 가능 β₀)
  D1: 시계열 플롯 (φ, α, θ, δ)
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================
#  v14_mujoco 파라미터 (동일하게 유지)
# ============================================================
# 최적화 형상 파라미터 (H=80cm, 3kg)
GRAV = 9.81; DT = 0.001
M1 = 0.80; M2 = 1.70; L1 = 0.24; L2 = 0.56; BD = 0.10
M_HEAD = 0.50; HEAD_R = L2/4
LOWER_W = L2/4; UPPER_W = L2/4
LC1 = L1/2; I1 = M1*L1**2/12
L_POLE = 0.80; POLE_H = 1.00
B_THETA = 0.01; TAU_MAX = 2.2
ROPE_MASS = 0.01
POLE_RADIUS = max(0.005, L2*0.05)
POLE_TOP_R = max(0.008, L2*0.075)
ROPE_RADIUS = max(0.003, L2*0.015)
HIP_R = max(0.008, L2*0.08)
HIP_H = max(0.005, BD*0.15)

def compute_effective_upper():
    m2 = M2; m_h = M_HEAD
    d2 = L2/2; d_h = L2 + HEAD_R
    m_tot = m2 + m_h
    lc2 = (m2*d2 + m_h*d_h) / m_tot
    i2_body = m2*L2**2/12 + m2*(d2-lc2)**2
    i2_head = 2/5*m_h*HEAD_R**2 + m_h*(d_h-lc2)**2
    return m_tot, lc2, i2_body + i2_head

M2_EFF, LC2_EFF, I2_EFF = compute_effective_upper()

# ============================================================
#  모델 생성 (v14와 동일)
# ============================================================
def generate_mjcf(R_target):
    HALF_POLE = L_POLE/2.0
    rh = HALF_POLE - BD/2.0
    rv = R_target
    hp = HALF_POLE
    FOOT_Z = POLE_H - rv
    lw2 = LOWER_W/2; uw2 = UPPER_W/2; bd2 = BD/2
    xml = f"""<?xml version="1.0"?>
<mujoco model="slackline_v14_batch">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0.001"/>
  </default>
  <worldbody>
    <geom type="plane" size="5 5 0.01" rgba="0.3 0.3 0.35 1" contype="1" conaffinity="1"/>
    <light pos="3 0 6" dir="-0.5 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="pole_a_cyl" type="cylinder" pos="0 {hp} {POLE_H/2}" size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom name="pole_a_top" type="sphere" pos="0 {hp} {POLE_H}" size="{POLE_TOP_R}" rgba="0.9 0.9 0.9 1"/>
    <geom name="pole_b_cyl" type="cylinder" pos="0 {-hp} {POLE_H/2}" size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom name="pole_b_top" type="sphere" pos="0 {-hp} {POLE_H}" size="{POLE_TOP_R}" rgba="0.9 0.9 0.9 1"/>
    <body name="rope_a_mount" pos="0 {hp} {POLE_H}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_a_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_a" type="capsule" size="{ROPE_RADIUS}" fromto="0 0 0 0 {-rh} {-rv}" rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="lower_body" pos="0 {-rh} {-rv}">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom name="lower_geom" type="box" size="{lw2} {bd2} {L1/2}" pos="0 {-bd2} {L1/2}" mass="{M1}" rgba="0.02 0.84 0.63 0.8"/>
        <body name="ankle_b_target" pos="0 {-BD} 0"/>
        <geom name="hip_cyl" type="cylinder" size="{HIP_R} {HIP_H}" pos="0 {-bd2} {L1}" euler="90 0 0" rgba="1 0.75 0.04 1" mass="0.01"/>
        <body name="upper_body" pos="0 {-bd2} {L1}">
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}"/>
          <geom name="upper_geom" type="box" size="{uw2} {bd2} {L2/2}" pos="0 0 {L2/2}" mass="{M2}" rgba="0.51 0.22 0.93 0.8"/>
          <geom name="head" type="sphere" size="{HEAD_R}" pos="0 0 {L2 + HEAD_R}" mass="{M_HEAD}" rgba="1.0 0.85 0.7 0.9"/>
        </body>
      </body>
    </body>
    <body name="rope_b_mount" pos="0 {-hp} {POLE_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_b_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_b" type="capsule" size="{ROPE_RADIUS}" fromto="0 0 0 0 {rh} {-rv}" rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="rope_b_end" pos="0 {rh} {-rv}"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0" solref="0.001 1" solimp="0.999 0.999 0.0001"/>
  </equality>
  <actuator>
    <motor name="hip_motor" joint="hip" ctrlrange="{-TAU_MAX} {TAU_MAX}" ctrllimited="true" gear="1"/>
  </actuator>
</mujoco>"""
    return xml

# ============================================================
#  LQR 계산
# ============================================================
def compute_lqr(R, Q_diag=None, R_cost=0.001):
    from scipy.linalg import solve_discrete_are, expm
    if Q_diag is None:
        Q_diag = [1, 50, 50, 0.1, 5, 5]
    m2e = M2_EFF; lc2e = LC2_EFF; i2e = I2_EFF
    p1 = M1*LC1 + m2e*L1; p2 = m2e*lc2e; p3 = m2e*L1*lc2e; Mt = M1+m2e
    M_eq = np.array([
        [Mt*R**2, R*p1, R*p2],
        [R*p1, M1*LC1**2+I1+m2e*L1**2, p3],
        [R*p2, p3, m2e*lc2e**2+i2e]])
    G_mat = np.array([[-Mt*GRAV*R,0,0],[0,p1*GRAV,0],[0,0,p2*GRAV]])
    D_mat = np.diag([0, 0, -B_THETA])
    E_mat = np.array([[0],[-1],[1]])
    Mi = np.linalg.inv(M_eq); n = 6
    Ac = np.zeros((n,n)); Bc = np.zeros((n,1))
    Ac[:3,3:] = np.eye(3)
    Ac[3:,:3] = Mi @ G_mat; Ac[3:,3:] = Mi @ D_mat; Bc[3:,:] = Mi @ E_mat
    Z = np.zeros((n+1,n+1)); Z[:n,:n]=Ac*DT; Z[:n,n:]=Bc*DT
    eZ = expm(Z); Ad=eZ[:n,:n]; Bd=eZ[:n,n:n+1]
    Q = np.diag(Q_diag); Rc = np.array([[R_cost]])
    P = solve_discrete_are(Ad, Bd, Q, Rc)
    K = np.linalg.inv(Rc + Bd.T@P@Bd) @ (Bd.T@P@Ad)
    K[0,0] *= -1; K[0,3] *= -1
    return K[0]

# ============================================================
#  상태 추출 (v14와 동일)
# ============================================================
def get_state(model, data):
    phi = data.joint('phi_Y').qpos[0]
    phi_dot = data.joint('phi_Y').qvel[0]
    lb_id = model.body('lower_body').id
    xmat = data.xmat[lb_id].reshape(3,3)
    alpha = np.arctan2(-xmat[2,0], xmat[2,2])
    ub_id = model.body('upper_body').id
    xmat_u = data.xmat[ub_id].reshape(3,3)
    theta = np.arctan2(-xmat_u[2,0], xmat_u[2,2])
    lb_angvel = data.cvel[lb_id][:3]
    ub_angvel = data.cvel[ub_id][:3]
    alpha_dot = lb_angvel[1]
    theta_dot = ub_angvel[1]
    return np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])

# ============================================================
#  단일 시뮬레이션 실행
# ============================================================
def run_sim(R, beta0_deg, sim_time=10.0, K_lqr=None):
    """Run one headless simulation. Returns dict of results."""
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    if K_lqr is None:
        K_lqr = compute_lqr(R)
    
    beta0 = np.radians(beta0_deg)
    hip_jid = model.joint('hip').id
    hip_qadr = model.jnt_qposadr[hip_jid]
    ankle_jid = model.joint('ankle').id
    ankle_qadr = model.jnt_qposadr[ankle_jid]
    
    # 초기 조건: α = β₀, θ = β₀ (몸 전체가 β₀ 기울어짐)
    data.qpos[ankle_qadr] = beta0
    data.qpos[hip_qadr] = 0  # hip relative = θ - α = 0
    mujoco.mj_forward(model, data)
    
    n_steps = int(sim_time / DT)
    log_t = []; log_phi = []; log_alpha = []; log_theta = []; log_tau = []
    
    max_delta = 0.0; t_delta_peak = 0.0
    max_tau = 0.0
    converged = False; t_converge = sim_time
    failed = False
    
    for i in range(n_steps):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > np.radians(80):
            failed = True
            break
        
        tau = -float(K_lqr @ x)
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        data.ctrl[0] = tau
        
        # 기록
        delta = x[2] - x[1]  # θ - α
        if abs(delta) > abs(max_delta):
            max_delta = delta
            t_delta_peak = data.time
        if abs(tau) > max_tau:
            max_tau = abs(tau)
        
        # 수렴 판정: |α| < 0.5° AND |θ| < 0.5° AND |φ| < 0.5°
        if not converged and abs(x[0]) < np.radians(0.5) and \
           abs(x[1]) < np.radians(0.5) and abs(x[2]) < np.radians(0.5):
            converged = True
            t_converge = data.time
        
        # 샘플링 (매 10스텝)
        if i % 10 == 0:
            log_t.append(data.time)
            log_phi.append(x[0])
            log_alpha.append(x[1])
            log_theta.append(x[2])
            log_tau.append(tau)
        
        mujoco.mj_step(model, data)
    
    return {
        'R': R, 'beta0_deg': beta0_deg, 'beta0_rad': beta0,
        'delta_peak': max_delta, 't_delta_peak': t_delta_peak,
        'max_tau': max_tau, 't_converge': t_converge,
        'converged': converged, 'failed': failed,
        'log_t': np.array(log_t), 'log_phi': np.array(log_phi),
        'log_alpha': np.array(log_alpha), 'log_theta': np.array(log_theta),
        'log_tau': np.array(log_tau)
    }

# ============================================================
#  배치 실험
# ============================================================
def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
    os.makedirs(out_dir, exist_ok=True)
    
    R_list = [0.3, 0.4, 0.5, 0.6, 0.7]  # 소형 로봇 범위
    beta_list = [1, 2, 3, 5, 7, 10, 12, 15]
    
    print("=" * 70)
    print("  MuJoCo 배치 실험: 메인 식 검증")
    print("=" * 70)
    
    # --- LQR 사전 계산 ---
    K_cache = {}
    for R in R_list:
        K_cache[R] = compute_lqr(R)
        print(f"  R={R:.1f}m: K=[{', '.join(f'{k:.1f}' for k in K_cache[R])}]")
    
    # --- 전체 실험 실행 ---
    results = {}
    total = len(R_list) * len(beta_list)
    count = 0
    
    for R in R_list:
        for b0 in beta_list:
            count += 1
            print(f"\r  [{count}/{total}] R={R:.1f}m, β₀={b0}°...", end='', flush=True)
            res = run_sim(R, b0, sim_time=10.0, K_lqr=K_cache[R])
            results[(R, b0)] = res
    
    print(f"\n  완료! 총 {len(results)}개 시뮬레이션")
    
    # ============================================================
    #  결과 출력 & 그래프
    # ============================================================
    
    # --- 결과 테이블 ---
    print(f"\n{'R':>5} {'β₀':>5} {'δ_peak':>10} {'δ/β₀':>8} {'ε_MJ':>8} {'t_conv':>8} {'τ_max':>8} {'상태':>6}")
    print("-" * 70)
    c_foot = 0.126
    h_com = (M1*LC1 + M2_EFF*(L1+LC2_EFF)) / (M1+M2_EFF)
    
    for R in R_list:
        for b0 in beta_list:
            res = results[(R, b0)]
            b0_rad = np.radians(b0)
            d_peak = abs(res['delta_peak'])
            ratio = d_peak / b0_rad if b0_rad > 0 else 0
            eps_mj = c_foot * d_peak / (h_com * b0_rad) - 1 if b0_rad > 0 else 0
            status = "FAIL" if res['failed'] else ("OK" if res['converged'] else "SLOW")
            print(f"{R:5.1f} {b0:5d} {np.degrees(d_peak):10.2f}° {ratio:8.2f} {eps_mj:8.3f} "
                  f"{res['t_converge']:8.2f}s {res['max_tau']:8.0f}N {status:>6}")
    
    # ============================================================
    #  Graph A1: δ_peak vs β₀ (비례 관계 확인)
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6', '#34495e']
    
    # --- A1: δ_peak vs β₀ ---
    ax = axes[0, 0]
    for R, color in zip(R_list, colors):
        betas = []; deltas = []
        for b0 in beta_list:
            res = results[(R, b0)]
            if not res['failed']:
                betas.append(b0)
                deltas.append(np.degrees(abs(res['delta_peak'])))
        if betas:
            ax.plot(betas, deltas, 'o-', color=color, markersize=5, linewidth=2,
                    label=f'R={R}m')
    ax.set_xlabel('β₀ [deg]', fontsize=12)
    ax.set_ylabel('δ_peak [deg]', fontsize=12)
    ax.set_title('A1: δ_peak vs β₀\n(MuJoCo, 비선형)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- A2: ε vs R ---
    ax = axes[0, 1]
    for b0 in [3, 5, 10]:
        Rs = []; epsilons = []
        for R in R_list:
            res = results[(R, b0)]
            if not res['failed']:
                b0_rad = np.radians(b0)
                d_peak = abs(res['delta_peak'])
                eps = c_foot * d_peak / (h_com * b0_rad) - 1
                Rs.append(R)
                epsilons.append(eps)
        if Rs:
            ax.plot(Rs, epsilons, 'o-', markersize=5, linewidth=2, label=f'β₀={b0}°')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('ε (MuJoCo)', fontsize=12)
    ax.set_title('A2: ε vs R\n(R이 ε를 결정)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- A3: δ/β₀ vs β₀ (비율 일정성) ---
    ax = axes[0, 2]
    for R, color in zip(R_list, colors):
        betas = []; ratios = []
        for b0 in beta_list:
            res = results[(R, b0)]
            if not res['failed']:
                b0_rad = np.radians(b0)
                ratio = abs(res['delta_peak']) / b0_rad
                betas.append(b0)
                ratios.append(ratio)
        if betas:
            ax.plot(betas, ratios, 'o-', color=color, markersize=5, linewidth=2,
                    label=f'R={R}m')
    ax.set_xlabel('β₀ [deg]', fontsize=12)
    ax.set_ylabel('δ_peak / β₀', fontsize=12)
    ax.set_title('A3: δ/β₀ 비율 일정성\n(수평 = 비례 제어)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- B1: 수렴 시간 vs R ---
    ax = axes[1, 0]
    for b0 in [3, 5, 10]:
        Rs = []; t_convs = []
        for R in R_list:
            res = results[(R, b0)]
            if not res['failed'] and res['converged']:
                Rs.append(R)
                t_convs.append(res['t_converge'])
        if Rs:
            ax.plot(Rs, t_convs, 'o-', markersize=5, linewidth=2, label=f'β₀={b0}°')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('수렴 시간 [s]', fontsize=12)
    ax.set_title('B1: 수렴 시간 vs R\n(최소 = 최적 R)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- B2: 최대 토크 vs R ---
    ax = axes[1, 1]
    for b0 in [3, 5, 10]:
        Rs = []; taus = []
        for R in R_list:
            res = results[(R, b0)]
            if not res['failed']:
                Rs.append(R)
                taus.append(res['max_tau'])
        if Rs:
            ax.plot(Rs, taus, 'o-', markersize=5, linewidth=2, label=f'β₀={b0}°')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('최대 토크 [N·m]', fontsize=12)
    ax.set_title('B2: 최대 토크 vs R', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- B3: 안정 영역 (최대 복원 가능 β₀) ---
    ax = axes[1, 2]
    max_betas = []
    for R in R_list:
        max_b = 0
        for b0 in beta_list:
            res = results[(R, b0)]
            if not res['failed'] and res['converged']:
                max_b = b0
        max_betas.append(max_b)
    ax.bar([str(R) for R in R_list], max_betas, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('R [m]', fontsize=12)
    ax.set_ylabel('최대 복원 가능 β₀ [deg]', fontsize=12)
    ax.set_title('B3: 안정 영역\n(높을수록 강건)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MuJoCo 비선형 시뮬레이션: 메인 식 검증', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = os.path.join(out_dir, 'mujoco_batch_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  그래프 저장: {save_path}")
    
    # ============================================================
    #  Graph D1: 시계열 (대표 케이스: R=0.7, β₀=5°)
    # ============================================================
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    res_demo = results.get((0.7, 5)) or results.get((1.0, 5))
    if res_demo and not res_demo['failed']:
        t = res_demo['log_t']
        ax = axes2[0]
        ax.plot(t, np.degrees(res_demo['log_phi']), label='φ (줄)', linewidth=2)
        ax.plot(t, np.degrees(res_demo['log_alpha']), label='α (하체)', linewidth=2)
        ax.plot(t, np.degrees(res_demo['log_theta']), label='θ (상체)', linewidth=2)
        delta = res_demo['log_theta'] - res_demo['log_alpha']
        ax.plot(t, np.degrees(delta), '--', label='δ=θ-α (접힘)', linewidth=2, color='gray')
        ax.set_ylabel('각도 [deg]', fontsize=12)
        ax.set_title(f'D1: 시계열 (R={res_demo["R"]}m, β₀={res_demo["beta0_deg"]}°)', 
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        ax = axes2[1]
        ax.plot(t, res_demo['log_tau'], color='#e74c3c', linewidth=1.5)
        ax.set_xlabel('시간 [s]', fontsize=12)
        ax.set_ylabel('토크 τ [N·m]', fontsize=12)
        ax.set_title('토크 시계열', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    save_path2 = os.path.join(out_dir, 'mujoco_timeseries.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"  시계열 그래프 저장: {save_path2}")
    
    print("\n  ✅ 배치 실험 완료!")
    plt.show()


if __name__ == '__main__':
    main()

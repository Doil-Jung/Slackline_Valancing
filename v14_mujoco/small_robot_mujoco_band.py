# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
MuJoCo 기반 안정 띠 폭 직접 측정
— 소형 로봇 유망 파라미터에서 ε 스캔하여 실제 안정 띠 확인
"""
import numpy as np, mujoco, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

GRAV = 9.81; DT = 0.001

# ============================================================
# 질량 모델 (small_robot_scan.py 와 동일)
# ============================================================
RHO_LINEAR = 0.30
M_SERVO = 0.067
M_ELECTRONICS = 0.10

def compute_params(L1, L2):
    M1 = RHO_LINEAR * L1 + M_SERVO
    M2_body = RHO_LINEAR * L2 + M_SERVO
    M_head = 0.03 * (L2 / 0.25)
    HEAD_R = L2 / 4
    m2_eff = M2_body + M_head + M_ELECTRONICS
    Mt = M1 + m2_eff
    LC1 = L1 / 2; I1 = M1 * L1**2 / 12
    d_body = L2/2; d_head = L2 + HEAD_R; d_elec = L2*0.6
    LC2_eff = (M2_body*d_body + M_head*d_head + M_ELECTRONICS*d_elec) / m2_eff
    I2_body = M2_body*L2**2/12 + M2_body*(d_body-LC2_eff)**2
    I2_head = 2/5*M_head*HEAD_R**2 + M_head*(d_head-LC2_eff)**2
    I2_elec = M_ELECTRONICS*(d_elec-LC2_eff)**2
    I2_eff = I2_body + I2_head + I2_elec
    p1 = M1*LC1 + m2_eff*L1; p2 = m2_eff*LC2_eff
    ps = p1+p2; h = ps/Mt
    return {
        'M1':M1,'M2_eff':m2_eff,'Mt':Mt,'L1':L1,'L2':L2,'L_TOTAL':L1+L2,
        'LC1':LC1,'LC2_eff':LC2_eff,'I1':I1,'I2_eff':I2_eff,
        'p1':p1,'p2':p2,'ps':ps,'h':h,
        'J_aa':M1*LC1**2+I1+m2_eff*L1**2,
        'J_tt':m2_eff*LC2_eff**2+I2_eff,
        'J_at':m2_eff*L1*LC2_eff,
        'HEAD_R':HEAD_R,'M_head':M_head,
    }

def mode_analysis(params, R):
    Mt=params['Mt']; p1=params['p1']; p2=params['p2']
    ps=params['ps']; h=params['h']
    T_inv = np.array([[1,0,0],[0,Mt*h/ps,-p2/ps],[0,Mt*h/ps,p1/ps]])
    M = np.array([[Mt*R**2,R*p1,R*p2],[R*p1,params['J_aa'],params['J_at']],
                   [R*p2,params['J_at'],params['J_tt']]])
    G = np.diag([-Mt*GRAV*R, p1*GRAV, p2*GRAV])
    Mp = T_inv.T@M@T_inv; Gp = T_inv.T@G@T_inv
    A2 = np.linalg.solve(Mp[:2,:2], Gp[:2,:2])
    ev,V = np.linalg.eig(A2)
    idx = np.argsort(ev.real); ev=ev[idx].real; V=V[:,idx].real
    Vi = np.linalg.inv(V)
    if ev[0]>=0 or ev[1]<=0: return None
    return {'omega':np.sqrt(abs(ev[0])),'lam':np.sqrt(abs(ev[1])),
            'v1':V[:,0],'v2':V[:,1],'u1':Vi[0,:],'u2':Vi[1,:],'h':h}

def eps_star_gen(mode, h, R, phi_i, beta_i, pdot_i, bdot_i):
    u2=mode['u2']; lam=mode['lam']
    q_pred = np.array([phi_i+h*beta_i/R+pdot_i/lam, bdot_i/lam])
    e = np.array([h/R, -1.0])
    den = beta_i*(u2@e)
    if abs(den)<1e-15: return np.nan
    return -(u2@q_pred)/den

# ============================================================
# MuJoCo 모델 생성 (소형 로봇용)
# ============================================================
def generate_small_mjcf(params, R):
    L1=params['L1']; L2=params['L2']; M1=params['M1']
    M2_eff=params['M2_eff']; Mt=params['Mt']
    LC1=params['LC1']; LC2_eff=params['LC2_eff']
    HEAD_R=params['HEAD_R']; M_head=params['M_head']
    BD = min(L1*0.4, 0.20)   # hip depth (비례)
    
    # 기둥 높이와 폴 길이
    POLE_H = L1 + L2 + R + 0.3  # 로봇 높이 + R + 여유
    L_POLE = 1.0  # 기둥 간격 고정
    HALF_POLE = L_POLE / 2
    
    rh = HALF_POLE - BD/2; rv = R; hp = HALF_POLE
    
    # 줄 반지름, 관절 크기 비례
    ROPE_R = max(0.002, L2*0.015)
    POLE_R = max(0.005, L2*0.05)
    POLE_TOP_R = max(0.005, L2*0.06)
    HIP_R = max(0.005, L2*0.06)
    HIP_H = max(0.003, BD*0.1)
    LOWER_W = max(0.01, L2/5)
    UPPER_W = max(0.01, L2/5)
    
    # 질량 배분
    M2_body = M2_eff - M_head - M_ELECTRONICS
    ROPE_MASS = 0.005
    
    return f"""<mujoco model="small_slackline">
  <option gravity="0 0 -{GRAV}" timestep="{DT}"/>
  <default>
    <joint damping="0" armature="0"/>
    <geom condim="1" friction="1 0.005 0.001"/>
  </default>
  <worldbody>
    <light pos="0 -2 3" dir="0 1 -0.5" diffuse="1 1 1"/>
    <geom type="cylinder" pos="-{hp} 0 {POLE_H/2}" size="{POLE_R} {POLE_H/2}" rgba="0.4 0.4 0.4 1"/>
    <geom type="cylinder" pos=" {hp} 0 {POLE_H/2}" size="{POLE_R} {POLE_H/2}" rgba="0.4 0.4 0.4 1"/>
    <geom type="sphere" pos="-{hp} 0 {POLE_H}" size="{POLE_TOP_R}" rgba=".8 .2 .2 1"/>
    <geom type="sphere" pos=" {hp} 0 {POLE_H}" size="{POLE_TOP_R}" rgba=".8 .2 .2 1"/>
    <body name="rope_a" pos="-{hp} 0 {POLE_H}">
      <joint name="rope_a_Y" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 {rh} 0 -{rv}" size="{ROPE_R}" mass="{ROPE_MASS/2}"/>
    </body>
    <body name="rope_b" pos="{hp} 0 {POLE_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 -{rh} 0 -{rv}" size="{ROPE_R}" mass="{ROPE_MASS/2}"/>
    </body>
    <body name="foot" pos="0 0 {POLE_H - rv}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="{max(0.005,L1*0.03)}" mass="0.005" rgba=".1 .1 .1 1"/>
      <body name="lower" pos="0 0 0">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom type="box" pos="0 0 {LC1}" size="{LOWER_W/2} {LOWER_W/2} {L1/2}" 
              mass="{M1}" rgba="0.3 0.5 0.8 1"/>
        <body name="upper" pos="0 0 {L1}">
          <joint name="hip" type="hinge" axis="0 1 0" limited="true" range="-120 120"/>
          <geom type="cylinder" pos="0 0 0" size="{HIP_R} {HIP_H}" mass="0.001" rgba=".5 .5 .5 1"/>
          <geom type="box" pos="0 0 {LC2_eff}" size="{UPPER_W/2} {UPPER_W/2} {L2/2}"
                mass="{M2_body + M_ELECTRONICS}" rgba="0.8 0.3 0.3 1"/>
          <body name="head" pos="0 0 {L2 + HEAD_R}">
            <geom type="sphere" size="{HEAD_R}" mass="{M_head}" rgba="1 0.85 0.7 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect body1="rope_a" body2="foot" anchor="{rh} 0 -{rv}" solref="-10000 -200"/>
    <connect body1="rope_b" body2="foot" anchor="-{rh} 0 -{rv}" solref="-10000 -200"/>
  </equality>
  <actuator>
    <motor name="hip_motor" joint="hip" ctrllimited="true" ctrlrange="-500 500" gear="1"/>
  </actuator>
</mujoco>"""

def get_state(model, data):
    phi = data.qpos[model.jnt_qposadr[model.joint('phi_Y').id]]
    ankle_q = data.qpos[model.jnt_qposadr[model.joint('ankle').id]]
    hip_q = data.qpos[model.jnt_qposadr[model.joint('hip').id]]
    alpha = ankle_q + phi; theta = hip_q + alpha
    phi_dot = data.qvel[model.jnt_dofadr[model.joint('phi_Y').id]]
    ankle_dot = data.qvel[model.jnt_dofadr[model.joint('ankle').id]]
    hip_dot = data.qvel[model.jnt_dofadr[model.joint('hip').id]]
    alpha_dot = ankle_dot + phi_dot; theta_dot = hip_dot + alpha_dot
    return np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])

def run_sim(params, R, eps, phi_i, beta_i, pdot_i, bdot_i, sim_t=2.0):
    """MuJoCo 시뮬레이션, max|x_CoM| 반환"""
    h = params['h']; p1=params['p1']; p2=params['p2']; Mt=params['Mt']
    
    phi_after = phi_i + h*(1+eps)*beta_i/R
    beta_after = -eps*beta_i
    phi_mj = -phi_after
    alpha_after = beta_after  # β≈α when δ=0
    
    xml = generate_small_mjcf(params, R)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except:
        return 9999.0
    data = mujoco.MjData(model)
    
    ra = model.jnt_qposadr[model.joint('phi_Y').id]
    aa = model.jnt_qposadr[model.joint('ankle').id]
    ha = model.jnt_qposadr[model.joint('hip').id]
    try: rb = model.jnt_qposadr[model.joint('rope_b_Y').id]
    except: rb = None
    va = model.jnt_dofadr[model.joint('phi_Y').id]
    va_a = model.jnt_dofadr[model.joint('ankle').id]
    va_h = model.jnt_dofadr[model.joint('hip').id]
    try: vb = model.jnt_dofadr[model.joint('rope_b_Y').id]
    except: vb = None
    
    data.qpos[ra] = phi_mj; data.qpos[aa] = alpha_after - phi_mj; data.qpos[ha] = 0
    if rb is not None: data.qpos[rb] = phi_mj
    phi_dot_mj = -pdot_i
    data.qvel[va] = phi_dot_mj; data.qvel[va_a] = bdot_i - phi_dot_mj; data.qvel[va_h] = 0
    if vb is not None: data.qvel[vb] = phi_dot_mj
    mujoco.mj_forward(model, data)
    
    max_xcom = 0; n = int(sim_t / DT)
    for i in range(n):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > 1.4: return 9999.0
        delta = x[2]-x[1]; delta_d = x[5]-x[4]
        data.ctrl[0] = np.clip(5000*(0-delta)-100*delta_d, -500, 500)
        if i % 10 == 0:
            phi_v=x[0]; alpha_v=x[1]; theta_v=x[2]
            beta_v = (p1*alpha_v + p2*theta_v) / (Mt*h)
            xcom = (-R*phi_v + h*beta_v) * 1000  # mm
            max_xcom = max(max_xcom, abs(xcom))
        mujoco.mj_step(model, data)
    return max_xcom

# ============================================================
# 유망 파라미터 후보에서 MuJoCo 안정 띠 측정
# ============================================================
beta_i = np.radians(2); phi_i = np.radians(0.5)
pdot_i = np.radians(5); bdot_i = np.radians(3)

# 정지 출발 조건도 같이 테스트
beta_i_s = np.radians(2); phi_i_s = 0; pdot_i_s = 0; bdot_i_s = 0

candidates = [
    # (L1, L2, R) — 다양한 크기/비율
    (0.20, 0.20, 0.20), (0.20, 0.20, 0.35), (0.20, 0.20, 0.50),
    (0.25, 0.25, 0.25), (0.25, 0.25, 0.35), (0.25, 0.25, 0.50),
    (0.30, 0.30, 0.30), (0.30, 0.30, 0.40), (0.30, 0.30, 0.55),
    (0.35, 0.35, 0.35), (0.35, 0.35, 0.50),
    (0.40, 0.40, 0.40), (0.40, 0.40, 0.55),
    (0.50, 0.50, 0.50), (0.50, 0.50, 0.60),
    # 비대칭
    (0.20, 0.30, 0.35), (0.30, 0.20, 0.35),
    (0.25, 0.35, 0.40), (0.35, 0.25, 0.40),
]

print("=" * 90)
print("  MuJoCo 기반 안정 띠 직접 측정")
print("=" * 90)

all_results = []

for ci, (L1, L2, R) in enumerate(candidates):
    params = compute_params(L1, L2)
    mode = mode_analysis(params, R)
    if mode is None:
        print(f"  [{ci+1:2d}] L1={L1*100:.0f} L2={L2*100:.0f} R={R*100:.0f} — 모드 분석 실패")
        continue
    
    h = params['h']
    
    # 정지 ε*
    r_ratio = mode['v1'][0] / mode['v1'][1]
    es = -h / (r_ratio * R + h)
    
    # 비정지 ε*_gen
    eg = eps_star_gen(mode, h, R, phi_i, beta_i, pdot_i, bdot_i)
    # 정지 ε*_gen (= ε*_stat 검증용)
    eg_s = eps_star_gen(mode, h, R, 0, beta_i_s, 0, 0)
    
    phi_after = np.degrees(phi_i + h*(1+eg)*beta_i/R)
    
    print(f"\n  [{ci+1:2d}] L1={L1*100:.0f}cm L2={L2*100:.0f}cm R={R*100:.0f}cm | "
          f"Mt={params['Mt']*1000:.0f}g h={h*100:.1f}cm | "
          f"ε*_stat={es:.3f} ε*_gen={eg:.3f} | φ₀={phi_after:.1f}°")
    
    # ε 스캔 — ε*_gen 주변 ±1.5 범위
    # 정지 조건
    eps_center_s = eg_s
    eps_lo_s = max(0.1, eps_center_s - 1.5)
    eps_hi_s = eps_center_s + 1.5
    eps_scan_s = np.arange(eps_lo_s, eps_hi_s, 0.05)
    
    survive_s = []
    for e in eps_scan_s:
        xmax = run_sim(params, R, e, 0, beta_i_s, 0, 0, sim_t=2.0)
        survive_s.append(xmax < 200)  # 200mm 이내 = 생존
    
    band_s = np.sum(survive_s) * 0.05  # 생존 ε 수 × 간격
    opt_idx_s = np.argmin([run_sim(params, R, e, 0, beta_i_s, 0, 0, 0.5) 
                           if survive_s[j] else 9999 
                           for j, e in enumerate(eps_scan_s)])
    eps_opt_s = eps_scan_s[opt_idx_s] if any(survive_s) else np.nan
    
    # 비정지 조건
    eps_center = eg
    eps_lo = max(0.1, eps_center - 1.5)
    eps_hi = eps_center + 1.5
    eps_scan = np.arange(eps_lo, eps_hi, 0.05)
    
    survive = []
    for e in eps_scan:
        xmax = run_sim(params, R, e, phi_i, beta_i, pdot_i, bdot_i, sim_t=2.0)
        survive.append(xmax < 200)
    
    band = np.sum(survive) * 0.05
    opt_idx = np.argmin([run_sim(params, R, e, phi_i, beta_i, pdot_i, bdot_i, 0.5)
                         if survive[j] else 9999
                         for j, e in enumerate(eps_scan)])
    eps_opt = eps_scan[opt_idx] if any(survive) else np.nan
    
    print(f"    정지: Δε={band_s:.2f} (생존 {np.sum(survive_s)}개) "
          f"ε_opt={eps_opt_s:.3f} vs ε*_stat={es:.3f}")
    print(f"    비정지: Δε={band:.2f} (생존 {np.sum(survive)}개) "
          f"ε_opt={eps_opt:.3f} vs ε*_gen={eg:.3f}")
    
    all_results.append({
        'L1':L1, 'L2':L2, 'R':R, 'Mt':params['Mt'],
        'h':h, 'es':es, 'eg':eg, 'eg_s':eg_s,
        'band_s':band_s, 'band':band,
        'eps_opt_s':eps_opt_s, 'eps_opt':eps_opt,
        'phi_after':phi_after,
        'omega':mode['omega'], 'lam':mode['lam'],
        'tau_decay':1/mode['lam'], 'T_osc':2*np.pi/mode['omega'],
        'eps_scan_s':eps_scan_s, 'survive_s':survive_s,
        'eps_scan':eps_scan, 'survive':survive,
    })

# ============================================================
# 결과 요약 그래프
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

nr = len(all_results)

# (a) 안정 띠 폭 비교
ax = axes[0][0]
labels = [f"{r['L1']*100:.0f}+{r['L2']*100:.0f}\nR={r['R']*100:.0f}" for r in all_results]
x_pos = np.arange(nr)
ax.bar(x_pos - 0.15, [r['band_s'] for r in all_results], 0.3, 
       color='#457B9D', alpha=0.8, label='정지 Δε')
ax.bar(x_pos + 0.15, [r['band'] for r in all_results], 0.3,
       color='#E63946', alpha=0.8, label='비정지 Δε')
ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=6, rotation=45)
ax.set_ylabel('안정 띠 폭 Δε', fontsize=11)
ax.set_title('안정 띠 폭: 정지 vs 비정지', fontsize=12, fontweight='bold')
ax.axhline(0.1, color='red', ls=':', lw=1, label='최소 요구')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

# (b) ε*_gen 오차
ax = axes[0][1]
errs_s = [abs(r['eps_opt_s']-r['eg_s'])/r['eg_s']*100 if not np.isnan(r['eps_opt_s']) else 100 
          for r in all_results]
errs = [abs(r['eps_opt']-r['eg'])/r['eg']*100 if not np.isnan(r['eps_opt']) else 100 
        for r in all_results]
ax.bar(x_pos - 0.15, errs_s, 0.3, color='#457B9D', alpha=0.8, label='정지')
ax.bar(x_pos + 0.15, errs, 0.3, color='#E63946', alpha=0.8, label='비정지')
ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=6, rotation=45)
ax.set_ylabel('|ε*_해석 - ε*_MuJoCo| / ε*_해석 [%]', fontsize=10)
ax.set_title('ε*_gen vs MuJoCo 최적 오차', fontsize=12, fontweight='bold')
ax.set_ylim(0, 20); ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

# (c) R vs 안정 띠 (총 길이별 색)
ax = axes[1][0]
for r in all_results:
    lt = r['L1']+r['L2']
    col = '#E63946' if lt<0.45 else '#F4A261' if lt<0.65 else '#2A9D8F' if lt<0.85 else '#264653'
    ax.scatter(r['R'], r['band_s'], color=col, s=80, marker='o', edgecolor='k', lw=0.5)
    ax.scatter(r['R'], r['band'], color=col, s=80, marker='^', edgecolor='k', lw=0.5)

ax.set_xlabel('R [m]', fontsize=11); ax.set_ylabel('안정 띠 폭 Δε', fontsize=11)
ax.set_title('R vs 안정 띠 (○=정지, △=비정지)', fontsize=12, fontweight='bold')
ax.axhline(0.1, color='red', ls=':', lw=1)
ax.grid(True, alpha=0.2)

# 범례
from matplotlib.lines import Line2D
legend_els = [
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#E63946',ms=8,label='총<45cm'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#F4A261',ms=8,label='45-65cm'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#2A9D8F',ms=8,label='65-85cm'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#264653',ms=8,label='85cm+'),
]
ax.legend(handles=legend_els, fontsize=8)

# (d) 최종 요약 테이블
ax = axes[1][1]; ax.axis('off')
cols = ['L1+L2', 'R', 'Mt', 'ε*gen', 'Δε(정지)', 'Δε(비정지)', 'φ₀']
rows = []
for r in sorted(all_results, key=lambda x: -x['band_s']):
    rows.append([
        f"{(r['L1']+r['L2'])*100:.0f}cm",
        f"{r['R']*100:.0f}cm",
        f"{r['Mt']*1000:.0f}g",
        f"{r['eg']:.2f}",
        f"{r['band_s']:.2f}",
        f"{r['band']:.2f}",
        f"{r['phi_after']:.1f}°",
    ])

table = ax.table(cellText=rows[:15], colLabels=cols, loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(8)
table.scale(1, 1.3)
# 헤더 색
for j in range(len(cols)):
    table[0, j].set_facecolor('#457B9D')
    table[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('상위 15개 (정지 Δε 순)', fontsize=12, fontweight='bold', pad=20)

fig.suptitle('소형 로봇: MuJoCo 기반 안정 띠 직접 측정', fontsize=14, fontweight='bold')
plt.tight_layout()
save = os.path.join(os.path.dirname(__file__), '..', 'docs', 'small_robot_mujoco_band.png')
plt.savefig(save, dpi=150, bbox_inches='tight')
print(f"\n  그래프 저장: {save}")
print("  완료!")

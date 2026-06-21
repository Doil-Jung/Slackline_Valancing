# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
소형 로봇 파라미터 스캔 (v2)
— 기존 검증된 scan_eps_gen_heatmap.py의 run_sim_trace를 그대로 사용
— 파라미터(질량, 길이)만 소형으로 교체
— generate_mjcf도 기존 batch_experiment.py의 것을 기반으로 파라미터만 교체
"""
import numpy as np, mujoco, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

GRAV = 9.81; DT = 0.001

# ============================================================
# 기존 batch_experiment.py 의 generate_mjcf 를 파라미터화
# ============================================================
def generate_mjcf_param(R_target, M1, M2, L1, L2, BD, M_HEAD, HEAD_R,
                         LOWER_W, UPPER_W, LC1, I1, L_POLE, POLE_H,
                         B_THETA, TAU_MAX, ROPE_MASS, POLE_RADIUS,
                         POLE_TOP_R, ROPE_RADIUS, HIP_R, HIP_H):
    """batch_experiment.py의 generate_mjcf 구조 그대로, 파라미터만 인자로 받음
    
    핵심: phi_Y가 rope_a_mount에 있고, body가 로프 끝에 자식으로 연결.
    대형 모델(batch_experiment.py)과 동일한 구조.
    """
    HALF_POLE = L_POLE / 2.0
    rh = HALF_POLE - BD / 2.0
    rv = R_target
    hp = HALF_POLE
    FOOT_Z = POLE_H - rv
    lw2 = LOWER_W / 2; uw2 = UPPER_W / 2; bd2 = BD / 2
    return f"""<?xml version="1.0"?>
<mujoco model="slackline_small">
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

# ============================================================
# 소형 로봇 파라미터 계산
# ============================================================
RHO_LINEAR = 0.30; M_SERVO = 0.067; M_ELECTRONICS = 0.10

def make_robot(L1, L2):
    """소형 로봇 파라미터 → 기존 코드와 동일 형태로 반환"""
    M1 = RHO_LINEAR * L1 + M_SERVO
    M2_body = RHO_LINEAR * L2 + M_SERVO
    M_HEAD = 0.03 * (L2 / 0.25)
    HEAD_R = L2 / 4
    # M2는 상체 로드 + 전자장치 (머리 제외)
    M2 = M2_body + M_ELECTRONICS
    
    LC1 = L1 / 2; I1 = M1 * L1**2 / 12
    BD = min(L1 * 0.4, 0.20)
    
    # 유효 상체 파라미터
    m2_total = M2 + M_HEAD
    d_body = L2 / 2; d_head = L2 + HEAD_R
    LC2_EFF = (M2 * d_body + M_HEAD * d_head) / m2_total
    I2_body = M2 * L2**2 / 12 + M2 * (d_body - LC2_EFF)**2
    I2_head = 2/5 * M_HEAD * HEAD_R**2 + M_HEAD * (d_head - LC2_EFF)**2
    I2_EFF = I2_body + I2_head
    
    Mt = M1 + m2_total
    p1 = M1*LC1 + m2_total*L1; p2 = m2_total*LC2_EFF
    ps = p1+p2; h = ps/Mt
    J_aa = M1*LC1**2 + I1 + m2_total*L1**2
    J_tt = m2_total*LC2_EFF**2 + I2_EFF
    J_at = m2_total*L1*LC2_EFF
    
    # MJCF 파라미터
    LOWER_W = max(0.02, L2/5); UPPER_W = max(0.02, L2/5)
    L_POLE = 1.0  # 기둥 간격
    POLE_H = max(2.0, L1+L2+1.0)  # 충분한 높이
    ROPE_MASS = 0.01
    POLE_RADIUS = max(0.005, L2*0.05)
    POLE_TOP_R = max(0.008, L2*0.075)
    ROPE_RADIUS = max(0.003, L2*0.015)
    HIP_R = max(0.008, L2*0.08)
    HIP_H = max(0.005, BD*0.15)
    # 토크: 대형은 50000, 소형은 질량 비례로 스케일
    TAU_MAX = max(50, 50000 * Mt / 80.0)
    B_THETA = 3.0
    
    return dict(
        M1=M1, M2=M2, L1=L1, L2=L2, BD=BD, M_HEAD=M_HEAD, HEAD_R=HEAD_R,
        LOWER_W=LOWER_W, UPPER_W=UPPER_W, LC1=LC1, I1=I1,
        L_POLE=L_POLE, POLE_H=POLE_H, B_THETA=B_THETA, TAU_MAX=TAU_MAX,
        ROPE_MASS=ROPE_MASS, POLE_RADIUS=POLE_RADIUS, POLE_TOP_R=POLE_TOP_R,
        ROPE_RADIUS=ROPE_RADIUS, HIP_R=HIP_R, HIP_H=HIP_H,
        # 해석용
        Mt=Mt, m2_total=m2_total, LC2_EFF=LC2_EFF, I2_EFF=I2_EFF,
        p1=p1, p2=p2, ps=ps, h=h,
        J_aa=J_aa, J_tt=J_tt, J_at=J_at,
    )

def mode_analysis_param(robot, R):
    Mt=robot['Mt']; h=robot['h']; p1=robot['p1']; p2=robot['p2']; ps=robot['ps']
    T_inv=np.array([[1,0,0],[0,Mt*h/ps,-p2/ps],[0,Mt*h/ps,p1/ps]])
    M=np.array([[Mt*R**2,R*p1,R*p2],[R*p1,robot['J_aa'],robot['J_at']],
                [R*p2,robot['J_at'],robot['J_tt']]])
    G=np.diag([-Mt*GRAV*R, p1*GRAV, p2*GRAV])
    Mp=T_inv.T@M@T_inv; Gp=T_inv.T@G@T_inv
    A2=np.linalg.solve(Mp[:2,:2], Gp[:2,:2])
    ev,V=np.linalg.eig(A2); idx=np.argsort(ev.real); ev=ev[idx].real; V=V[:,idx].real
    Vi=np.linalg.inv(V)
    return np.sqrt(abs(ev[0])), np.sqrt(abs(ev[1])), V[:,0], V[:,1], Vi[0], Vi[1]

def eps_star_gen_param(robot, R, phi_i, beta_i, pdot_i, bdot_i):
    _,lam,_,_,_,u2 = mode_analysis_param(robot, R)
    h = robot['h']
    q_pred = np.array([phi_i+h*beta_i/R+pdot_i/lam, bdot_i/lam])
    e = np.array([h/R, -1.0])
    den = beta_i*(u2@e)
    if abs(den)<1e-15: return np.nan
    return -(u2@q_pred)/den

def eps_star_stat_param(robot, R):
    _,_,v1,_,_,_ = mode_analysis_param(robot, R)
    h = robot['h']; r = v1[0]/v1[1]
    return -h/(r*R+h)

# ============================================================
# 기존 run_sim_trace와 동일한 로직, 파라미터만 외부 주입
# ============================================================
def run_sim_trace(robot, R, eps, phi_i, beta_i, pdot_i, bdot_i, sim_t=2.0, rec_dt=0.005):
    """기존 scan_eps_gen_heatmap.py의 run_sim_trace와 동일 로직"""
    h = robot['h']; p1 = robot['p1']; p2 = robot['p2']; Mt = robot['Mt']
    
    xml = generate_mjcf_param(R, robot['M1'], robot['M2'], robot['L1'], robot['L2'],
                               robot['BD'], robot['M_HEAD'], robot['HEAD_R'],
                               robot['LOWER_W'], robot['UPPER_W'], robot['LC1'], robot['I1'],
                               robot['L_POLE'], robot['POLE_H'], robot['B_THETA'],
                               robot['TAU_MAX'], robot['ROPE_MASS'], robot['POLE_RADIUS'],
                               robot['POLE_TOP_R'], robot['ROPE_RADIUS'],
                               robot['HIP_R'], robot['HIP_H'])
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    ra=model.jnt_qposadr[model.joint('phi_Y').id]
    aa=model.jnt_qposadr[model.joint('ankle').id]
    ha=model.jnt_qposadr[model.joint('hip').id]
    try: rb=model.jnt_qposadr[model.joint('rope_b_Y').id]
    except: rb=None
    va=model.jnt_dofadr[model.joint('phi_Y').id]
    va_a=model.jnt_dofadr[model.joint('ankle').id]
    va_h=model.jnt_dofadr[model.joint('hip').id]
    try: vb=model.jnt_dofadr[model.joint('rope_b_Y').id]
    except: vb=None

    # 초기조건 — 기존 코드와 동일
    phi_after = phi_i + h*(1+eps)*beta_i/R
    beta_after = -eps*beta_i
    phi_mj = -phi_after; alpha_after = beta_after
    data.qpos[ra]=phi_mj; data.qpos[aa]=alpha_after-phi_mj; data.qpos[ha]=0
    if rb: data.qpos[rb]=phi_mj
    phi_dot_mj=-pdot_i; alpha_dot=bdot_i
    data.qvel[va]=phi_dot_mj; data.qvel[va_a]=alpha_dot-phi_dot_mj
    data.qvel[va_h]=0
    if vb: data.qvel[vb]=phi_dot_mj
    mujoco.mj_forward(model,data)

    # 제어 게인 — 기존 코드와 동일 (Kp=10000, Kd=200)
    KP = 10000; KD = 200; CTRL_MAX = 50000

    rec_step = max(1, int(rec_dt/DT))
    n=int(sim_t/DT)
    times=[]; xcoms=[]
    for i in range(n):
        x=get_state(model,data)
        if np.any(np.isnan(x)) or abs(x[1])>1.4: break
        delta=x[2]-x[1]; delta_d=x[5]-x[4]
        data.ctrl[0]=np.clip(KP*(0-delta)-KD*delta_d, -CTRL_MAX, CTRL_MAX)
        if i%rec_step==0:
            beta_v=(p1*x[1]+p2*x[2])/(Mt*h)
            xcom=(-R*x[0]+h*beta_v)*1000
            times.append(data.time); xcoms.append(xcom)
        mujoco.mj_step(model,data)
    return np.array(times), np.array(xcoms)

# ============================================================
# 스캔 실행: 다양한 (L1, L2, R) 조합
# ============================================================
scan_cases = [
    (0.20, 0.20, 0.35),  # 총 40cm
    (0.25, 0.25, 0.35),  # 총 50cm
    (0.30, 0.30, 0.40),  # 총 60cm
    (0.40, 0.40, 0.40),  # 총 80cm
    (0.40, 0.40, 0.55),  # 총 80cm R55
    (0.50, 0.50, 0.50),  # 총 100cm
    (0.50, 0.50, 0.60),  # 총 100cm R60
]

# 정지 조건 + 비정지 조건
beta_i = np.radians(2)

conditions = [
    ("정지", 0, beta_i, 0, 0),
    ("비정지", np.radians(0.5), beta_i, np.radians(5), np.radians(3)),
]

print("=" * 90)
print("  소형 로봇 파라미터 스캔 v2 (기존 검증 코드 기반)")
print("=" * 90)

for L1, L2, R in scan_cases:
    robot = make_robot(L1, L2)
    om, lam, v1, v2, u1, u2 = mode_analysis_param(robot, R)
    es = eps_star_stat_param(robot, R)
    h = robot['h']
    
    print(f"\n  L1={L1*100:.0f}cm L2={L2*100:.0f}cm R={R*100:.0f}cm | "
          f"Mt={robot['Mt']*1000:.0f}g h={h*100:.1f}cm | "
          f"ω={om:.2f} λ={lam:.2f} | ε*_stat={es:.3f}")
    
    for cond_name, pi, bi, pdi, bdi in conditions:
        eg = eps_star_gen_param(robot, R, pi, bi, pdi, bdi)
        
        # ε*_gen에서 시뮬레이션
        t, xc = run_sim_trace(robot, R, eg, pi, bi, pdi, bdi, sim_t=2.0)
        survived = len(t) > 0 and t[-1] >= 1.9
        max_xcom = np.max(np.abs(xc)) if len(xc) > 0 else 9999
        t_die = t[-1] if len(t) > 0 else 0
        
        # ε*_gen ± 0.1 에서 생존 확인 (안정 띠)
        band_count = 0
        for de in np.arange(-1.0, 1.01, 0.05):
            e_test = eg + de
            if e_test < 0.1: continue
            t2, xc2 = run_sim_trace(robot, R, e_test, pi, bi, pdi, bdi, sim_t=2.0)
            if len(t2) > 0 and t2[-1] >= 1.9:
                band_count += 1
        band_width = band_count * 0.05
        
        status = f"ALIVE ({max_xcom:.0f}mm)" if survived else f"DEAD@{t_die:.2f}s"
        print(f"    {cond_name:4s}: ε*_gen={eg:.3f} → {status:20s} | Δε={band_width:.2f}")

print(f"\n  완료!")

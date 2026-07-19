# -*- coding: utf-8 -*-
"""
mjcf_v19.py — v19 재설계 실기(實機) MJCF 생성
==============================================
7/08 재설계 스펙을 기하 그대로 모델링:
  프레임(알루 2020, 400×600×1220) → 상단 피벗축(베어링+엔코더, 힌지 φ)
  → 카본 사다리꼴 크랭크(빗변 50cm ×2 + 발축 바 10cm, 질량 분포 실물)
  → 발목 베어링(힌지 α_rel) → 하체(L1) → 힙 XM430(힌지 δ, armature·마찰·감쇠)
  → 상체(L2, IMU 사이트)

★직렬 체인이다 — v14 의 닫힌 루프(equality) 불필요. 재설계가 평면운동을
  기계적으로 강제하므로 시상면 힌지 3개로 정확히 표현된다.

부호 규약 (AGENTS §9 함정):
  MuJoCo 힌지축은 전부 +Y. 몸 피치(α,θ)는 mj 양각 = 모델 양각(오른쪽 기울기 +).
  ★φ만 반전: 모델 φ = −q_phi_mj (v14 의 'K[0]*=−1' 함정과 동일 기원).
  변환은 sim_engine.get_state() 한 곳에서만 처리한다.

관성 처리:
  줄(크랭크)은 geom 질량으로 MuJoCo 가 관성을 계산 → calc_rope_inertia() 와
  교차검증 가능(막대근사 대 실기하). 로봇 몸체는 <inertial> 명시로
  params_v19 의 m, ℓ, I_cm 을 정확히 강제(geom 은 시각용).
"""

FRAME_X, FRAME_Y, FRAME_H = 0.400, 0.600, 1.220   # 알루 프레임
AXIS_Z = 1.245                                     # 피벗축 높이 (mount_rise=25)
PIVOT_HALF = 0.300                                 # 상단 피벗 이격 60cm 의 절반
BAR_HALF = 0.050                                   # 발축 바 10cm 의 절반
# 시각 치수 — CAD frame_v4 v7 확정값 (관성은 <inertial> 명시라 물리 무관)
W_LOWER = 0.080                                    # 하체 폭 (발목 베어링 외측 간격 80)
W_UPPER = 0.090                                    # 상체 폭
W_HEAD  = 0.098                                    # 머리 = C 파트(battery lid) 폭, 플레어
H_HEAD  = 0.046                                    # C 파트 높이
BODY_D  = 0.038                                    # 몸 두께 (CAD bbox 38)


def generate(p, real, plant_scale=None):
    """p: params_v19.flat() 값(공칭), real: 현실화 파라미터 dict,
       plant_scale: {'m1':1.05, ...} 플랜트 편차 배율(제어기는 모름).
       반환 MJCF XML 문자열."""
    ps = dict(m1=1.0, m2=1.0, L1=1.0, L2=1.0, R=1.0)
    if plant_scale:
        ps.update(plant_scale)

    # ---- 플랜트(실제) 측 값: 편차 적용 ----
    m1 = p["m1"] * ps["m1"];  m2 = p["m2"] * ps["m2"]
    L1 = p["L1"] * ps["L1"];  L2 = p["L2"] * ps["L2"]
    R  = p["R"]  * ps["R"]
    ell1 = p["ell1"] * ps["L1"];  ell2 = p["ell2"] * ps["L2"]   # CoM 은 길이에 비례 이동
    I1 = p["I1_cm"] * ps["m1"] * ps["L1"]**2
    I2 = p["I2_cm"] * ps["m2"] * ps["L2"]**2

    mc = p["crank_m_each"]; mb = p["bottom_m"]
    msock = p["top_socket_m"]; lsock = p["top_socket_ell"]

    # 크랭크 기하: 상단 피벗 (0, ±0.30, 0)_pivot → 발축 바 끝 (0, ±0.05, −R)
    yp, yb = PIVOT_HALF, BAR_HALF

    # 관절 물성
    c_phi   = p["c_phi"] + real.get("c_phi_extra", 0.0)
    f_phi   = real.get("f_phi", 0.0)          # 상단 베어링 쿨롱 [N·m]
    f_ank   = real.get("f_ankle", 0.0)        # 발목 베어링 쿨롱
    b_ank   = real.get("b_ankle", 0.0)
    arm     = real.get("armature", 0.0)       # 감속기어 반영 관성 [kg·m²]
    b_hip   = p["B_THETA"] + real.get("b_gear", 0.0)
    f_hip   = real.get("f_hip", 0.0)          # 기어 쿨롱 마찰

    # 몸체 얇은 축 관성(수치 안정용 최소값)
    I1z = max(1e-6, m1 * W_LOWER**2 / 12)
    I2z = max(1e-6, m2 * W_UPPER**2 / 12)

    xml = f"""<?xml version="1.0"?>
<mujoco model="slackline_v19">
  <option gravity="0 0 -9.81" timestep="{real.get('mj_dt', 5e-4)}"
          integrator="implicitfast">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0"/>
  </default>

  <worldbody>
    <geom type="plane" size="4 4 0.01" rgba="0.22 0.24 0.28 1" contype="1" conaffinity="1"/>
    <light pos="2 1 4" dir="-0.5 -0.2 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="-2 -1 4" dir="0.5 0.2 -1" diffuse="0.45 0.45 0.45"/>

    <!-- 알루 프레임 (시각) -->
    <geom type="box" pos="{FRAME_X/2} {FRAME_Y/2} {FRAME_H/2}" size="0.01 0.01 {FRAME_H/2}" rgba="0.55 0.57 0.60 1"/>
    <geom type="box" pos="{FRAME_X/2} {-FRAME_Y/2} {FRAME_H/2}" size="0.01 0.01 {FRAME_H/2}" rgba="0.55 0.57 0.60 1"/>
    <geom type="box" pos="{-FRAME_X/2} {FRAME_Y/2} {FRAME_H/2}" size="0.01 0.01 {FRAME_H/2}" rgba="0.55 0.57 0.60 1"/>
    <geom type="box" pos="{-FRAME_X/2} {-FRAME_Y/2} {FRAME_H/2}" size="0.01 0.01 {FRAME_H/2}" rgba="0.55 0.57 0.60 1"/>
    <geom type="box" pos="0 {FRAME_Y/2} {FRAME_H}" size="{FRAME_X/2} 0.01 0.01" rgba="0.55 0.57 0.60 1"/>
    <geom type="box" pos="0 {-FRAME_Y/2} {FRAME_H}" size="{FRAME_X/2} 0.01 0.01" rgba="0.55 0.57 0.60 1"/>
    <geom type="box" pos="0 {FRAME_Y/2} 0.13" size="{FRAME_X/2} 0.01 0.01" rgba="0.55 0.57 0.60 1"/>
    <geom type="box" pos="0 {-FRAME_Y/2} 0.13" size="{FRAME_X/2} 0.01 0.01" rgba="0.55 0.57 0.60 1"/>
    <!-- 피벗 마운트 -->
    <geom type="box" pos="0 {yp} {AXIS_Z}" size="0.02 0.015 0.02" rgba="0.8 0.8 0.85 1"/>
    <geom type="box" pos="0 {-yp} {AXIS_Z}" size="0.02 0.015 0.02" rgba="0.8 0.8 0.85 1"/>

    <!-- ==== 크랭크(사다리꼴, 강체) : 힌지 φ (상단 베어링+엔코더 축) ==== -->
    <body name="crank" pos="0 0 {AXIS_Z}">
      <joint name="j_phi" type="hinge" axis="0 1 0"
             damping="{c_phi}" frictionloss="{f_phi}"/>
      <geom name="crank_rod_a" type="capsule" size="0.004"
            fromto="0 {yp} 0  0 {yb} {-R}" mass="{mc}" rgba="0.15 0.15 0.17 1"/>
      <geom name="crank_rod_b" type="capsule" size="0.004"
            fromto="0 {-yp} 0  0 {-yb} {-R}" mass="{mc}" rgba="0.15 0.15 0.17 1"/>
      <geom name="foot_bar" type="capsule" size="0.005"
            fromto="0 {-yb} {-R}  0 {yb} {-R}" mass="{mb}" rgba="0.9 0.65 0.1 1"/>
      <geom name="sock_a" type="sphere" pos="0 {yp} {-lsock}" size="0.012" mass="{msock/2}" rgba="0.3 0.6 0.9 1"/>
      <geom name="sock_b" type="sphere" pos="0 {-yp} {-lsock}" size="0.012" mass="{msock/2}" rgba="0.3 0.6 0.9 1"/>

      <!-- ==== 하체 : 발목 베어링 α (발축 바 위) ==== -->
      <body name="lower" pos="0 0 {-R}">
        <joint name="j_ankle" type="hinge" axis="0 1 0"
               damping="{b_ank}" frictionloss="{f_ank}"/>
        <inertial pos="0 0 {ell1}" mass="{m1}" diaginertia="{I1} {I1} {I1z}"/>
        <geom name="lower_geom" type="box" size="{BODY_D/2} {W_LOWER/2} {L1/2}"
              pos="0 0 {L1/2}" mass="0.001" rgba="0.05 0.82 0.62 0.85"/>
        <geom name="hip_cyl" type="cylinder" size="0.014 {W_UPPER/2+0.008}"
              pos="0 0 {L1}" euler="90 0 0" mass="0.001" rgba="1 0.76 0.05 1"/>

        <!-- ==== 상체 : 힙 XM430 δ ==== -->
        <body name="upper" pos="0 0 {L1}">
          <joint name="j_hip" type="hinge" axis="0 1 0"
                 damping="{b_hip}" frictionloss="{f_hip}" armature="{arm}"/>
          <inertial pos="0 0 {ell2}" mass="{m2}" diaginertia="{I2} {I2} {I2z}"/>
          <geom name="upper_geom" type="box" size="{BODY_D/2} {W_UPPER/2} {L2/2}"
                pos="0 0 {L2/2}" mass="0.001" rgba="0.52 0.24 0.92 0.85"/>
          <!-- 머리 = C 파트(battery lid): 98폭 박스, 상체 위 플레어 -->
          <geom name="head_geom" type="box" size="{BODY_D/2} {W_HEAD/2} {H_HEAD/2}"
                pos="0 0 {L2+H_HEAD/2}" mass="0.001" rgba="0.95 0.8 0.35 0.95"/>
          <site name="imu" pos="0 0 {p['ELL_IMU']}" size="0.008" rgba="1 0.2 0.2 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <sensor>
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
  </sensor>

  <actuator>
    <motor name="hip_motor" joint="j_hip" gear="1"
           ctrlrange="-{p['TAU_MAX']*2} {p['TAU_MAX']*2}" ctrllimited="true"/>
  </actuator>
</mujoco>"""
    return xml

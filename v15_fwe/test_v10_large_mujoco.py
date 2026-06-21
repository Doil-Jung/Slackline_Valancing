# -*- coding: utf-8 -*-
"""
V15 MuJoCo Slackline Balancing — Large-scale v10 Robot (M=75kg)
==============================================================
FWE 4-Phase control verification for the large model.
"""
import os, sys, time
import numpy as np
import mujoco
import mujoco.viewer

# Add current dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fwe_controller import FWEController, Phase

# ============================================================
#  Robot Parameters (large-scale robot)
# ============================================================
M1 = 30.0         # lower body mass [kg]
M2 = 45.0         # upper body mass [kg]
L1 = 0.85         # lower body length [m]
L2 = 0.85         # upper body length [m]
BD = 0.40         # body depth (between ankles) [m]
GRAV = 9.81
DT = 0.001        # simulation timestep

# Rope / pole geometry
L_POLE = 3.0; POLE_H = 3.0
R_TARGET = 1.0    # rope length [m]

# Servo PD gains (stiff position tracking for high mass)
KP_SERVO = 35000.0
KD_SERVO = 1000.0
TAU_MAX = 50000.0

# Joint damping
B_THETA = 0.5     # hip joint damping

# Initial condition: beta0 ≈ 5°
ALPHA0 = np.radians(5.0)   # lower body tilt
THETA0 = np.radians(5.0)   # upper body tilt (same = uniform tilt)

# FWE parameters (from large model sweep)
FWE_EPS = 4.0
FWE_TF  = 0.150   # 150ms fold/extend time
FWE_TW1 = 0.200   # 200ms wait time
FWE_BETA_THRESH = np.radians(0.5)  # trigger threshold

# Derived
LC1 = L1/2; LC2 = L2/2
Mt = M1 + M2
I1 = M1*L1**2/12; I2 = M2*L2**2/12
p1 = M1*LC1 + M2*L1;  p2 = M2*LC2;  ps = p1 + p2
h_com = (M1*LC1 + M2*(L1+LC2)) / Mt
c_foot = p2 / ps

HALF_POLE = L_POLE / 2.0
ROPE_HORIZ = HALF_POLE - BD / 2.0
ROPE_VERT  = R_TARGET
FOOT_Z = POLE_H - ROPE_VERT
ROPE_MASS = 0.5  # Heavy rope for large-scale stability

# Geometry sizes
LOWER_W = L2/4; UPPER_W = L2/4
POLE_RADIUS = 0.05
POLE_TOP_R  = 0.08
ROPE_RADIUS = 0.015
HIP_R       = 0.08
HIP_H       = 0.05


# ============================================================
#  MJCF Generation
# ============================================================
def generate_mjcf():
    rh = ROPE_HORIZ; rv = ROPE_VERT; hp = HALF_POLE
    lw2 = LOWER_W/2; uw2 = UPPER_W/2; bd2 = BD/2

    xml = f"""<?xml version="1.0"?>
<mujoco model="slackline_v15_large_fwe">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>

  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0.01"/>
  </default>

  <worldbody>
    <!-- Ground -->
    <geom type="plane" size="6 6 0.01" rgba="0.3 0.3 0.35 1"
          contype="1" conaffinity="1"/>
    <light pos="4 0 8" dir="-0.3 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-4 0 8" dir="0.3 0 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Pole A -->
    <geom name="pole_a_cyl" type="cylinder" pos="0 {hp} {POLE_H/2}"
          size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom name="pole_a_top" type="sphere" pos="0 {hp} {POLE_H}"
          size="{POLE_TOP_R}" rgba="0.9 0.9 0.9 1"/>

    <!-- Pole B -->
    <geom name="pole_b_cyl" type="cylinder" pos="0 {-hp} {POLE_H/2}"
          size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom name="pole_b_top" type="sphere" pos="0 {-hp} {POLE_H}"
          size="{POLE_TOP_R}" rgba="0.9 0.9 0.9 1"/>

    <!-- ============ ROPE A CHAIN ============ -->
    <body name="rope_a_mount" pos="0 {hp} {POLE_H}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_a_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_a" type="capsule" size="{ROPE_RADIUS}"
            fromto="0 0 0 0 {-rh} {-rv}"
            rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>

      <!-- Lower body -->
      <body name="lower_body" pos="0 {-rh} {-rv}">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom name="lower_geom" type="box"
              size="{lw2} {bd2} {L1/2}"
              pos="0 {-bd2} {L1/2}"
              mass="{M1}" rgba="0.02 0.84 0.63 0.8"/>

        <body name="ankle_b_target" pos="0 {-BD} 0"/>

        <!-- Hip cylinder visual -->
        <geom name="hip_cyl" type="cylinder" size="{HIP_R} {HIP_H}"
              pos="0 {-bd2} {L1}" euler="90 0 0"
              rgba="1 0.75 0.04 1" mass="0.01"/>

        <!-- Upper body -->
        <body name="upper_body" pos="0 {-bd2} {L1}">
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}"/>
          <geom name="upper_geom" type="box"
                size="{uw2} {bd2} {L2/2}"
                pos="0 0 {L2/2}"
                mass="{M2}" rgba="0.51 0.22 0.93 0.8"/>
        </body>
      </body>
    </body>

    <!-- ============ ROPE B CHAIN ============ -->
    <body name="rope_b_mount" pos="0 {-hp} {POLE_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_b_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_b" type="capsule" size="{ROPE_RADIUS}"
            fromto="0 0 0 0 {rh} {-rv}"
            rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="rope_b_end" pos="0 {rh} {-rv}"/>
    </body>
  </worldbody>

  <!-- Closed loop constraint -->
  <equality>
    <connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0"
             solref="0.001 1" solimp="0.999 0.999 0.0001"/>
  </equality>

  <!-- Position servo actuator (PD control via general actuator) -->
  <actuator>
    <general name="hip_servo" joint="hip" biastype="affine"
             gainprm="{KP_SERVO} 0 0"
             biasprm="0 {-KP_SERVO} {-KD_SERVO}"
             forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"
             ctrlrange="-3.14 3.14" ctrllimited="true"/>
  </actuator>
</mujoco>"""
    return xml


# ============================================================
#  State Extraction
# ============================================================
def get_state(model, data):
    phi = data.joint('phi_Y').qpos[0]
    phi_dot = data.joint('phi_Y').qvel[0]

    lb_id = model.body('lower_body').id
    xmat = data.xmat[lb_id].reshape(3, 3)
    alpha = np.arctan2(-xmat[2, 0], xmat[2, 2])

    ub_id = model.body('upper_body').id
    xmat_u = data.xmat[ub_id].reshape(3, 3)
    theta = np.arctan2(-xmat_u[2, 0], xmat_u[2, 2])

    lb_angvel = data.cvel[lb_id][:3]
    ub_angvel = data.cvel[ub_id][:3]
    alpha_dot = lb_angvel[1]
    theta_dot = ub_angvel[1]

    return np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 60)
    print("  V15 MuJoCo - Large-scale v10 Robot FWE Verification")
    print(f"  m1={M1}kg  m2={M2}kg  Mt={Mt}kg  L1={L1}m  L2={L2}m  R={R_TARGET}m")
    print(f"  h_com={h_com:.4f}m  c_foot={c_foot:.4f}")
    print(f"  FWE parameters: eps={FWE_EPS} Tf={FWE_TF*1000:.0f}ms Tw1={FWE_TW1*1000:.0f}ms")
    print(f"  Servo parameters: kp={KP_SERVO} kd={KD_SERVO} max_torque={TAU_MAX}Nm")
    print("=" * 60)

    # Build model
    xml = generate_mjcf()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Joint addresses
    hip_jid = model.joint('hip').id
    hip_qadr = model.jnt_qposadr[hip_jid]
    ankle_jid = model.joint('ankle').id
    ankle_qadr = model.jnt_qposadr[ankle_jid]

    # Set initial pose
    data.qpos[ankle_qadr] = ALPHA0
    data.qpos[hip_qadr] = THETA0 - ALPHA0
    mujoco.mj_forward(model, data)

    x0 = get_state(model, data)
    beta0 = (p1*x0[1] + p2*x0[2]) / (Mt*h_com)
    print(f"[Init] phi={np.degrees(x0[0]):+.2f}d alpha={np.degrees(x0[1]):+.2f}d "
          f"theta={np.degrees(x0[2]):+.2f}d beta={np.degrees(beta0):+.2f}d")

    # Create FWE controller
    fwe = FWEController(
        eps=FWE_EPS, T_f=FWE_TF, T_w1=FWE_TW1,
        h_com=h_com, c_foot=c_foot,
        p1=p1, p2=p2, Mt=Mt,
        beta_threshold=FWE_BETA_THRESH
    )

    # CSV log
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_large_log.csv')
    log_f = open(log_path, 'w')
    log_f.write('t,phi,alpha,theta,phi_dot,alpha_dot,theta_dot,'
                'beta,delta_cmd,delta_actual,phase\n')

    # Simulation state
    paused = True
    ctrl_on = False
    step_n = 0

    print("[Sim] Space=Pause  R=Reset  C=Toggle_FWE  Q=Quit")
    print("[Sim] FWE OFF (press C to enable)")

    def reset():
        nonlocal step_n
        mujoco.mj_resetData(model, data)
        data.qpos[ankle_qadr] = ALPHA0
        data.qpos[hip_qadr] = THETA0 - ALPHA0
        mujoco.mj_forward(model, data)
        step_n = 0
        fwe.phase = Phase.IDLE
        fwe.t_phase = 0.0
        fwe.delta_target = 0.0
        fwe.cycle_count = 0
        data.ctrl[0] = 0.0
        print("[RESET]")

    def key_callback(keycode):
        nonlocal paused, ctrl_on
        if keycode == 32:  # Space
            paused = not paused
            if paused:
                x = get_state(model, data)
                beta = (p1*x[1] + p2*x[2]) / (Mt*h_com)
                print(f"[PAUSE] t={data.time:.3f}s phi={np.degrees(x[0]):+.2f} "
                      f"a={np.degrees(x[1]):+.2f} th={np.degrees(x[2]):+.2f} "
                      f"beta={np.degrees(beta):+.2f} phase={fwe.phase_name}")
            else:
                print(f"[RUN] FWE={'ON' if ctrl_on else 'OFF'}")
        elif keycode == ord('R') or keycode == ord('r'):
            reset()
            paused = True
        elif keycode == ord('C') or keycode == ord('c'):
            ctrl_on = not ctrl_on
            if not ctrl_on:
                data.ctrl[0] = 0.0
                fwe.phase = Phase.IDLE
                fwe.delta_target = 0.0
            print(f"[FWE] {'ON' if ctrl_on else 'OFF'}")
        elif keycode == ord('Q') or keycode == ord('q'):
            raise SystemExit

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        viewer.cam.lookat[:] = [0, 0, FOOT_Z]

        try:
            while viewer.is_running():
                if not paused:
                    STEPS_PER_FRAME = 24
                    for _ in range(STEPS_PER_FRAME):
                        x = get_state(model, data)

                        if np.any(np.isnan(x)):
                            print(f"[!] NaN at t={data.time:.3f}")
                            paused = True
                            break

                        phi = x[0]
                        alpha = x[1]
                        theta = x[2]
                        beta = (p1*alpha + p2*theta) / (Mt*h_com)
                        delta_actual = data.joint('hip').qpos[0]

                        if ctrl_on:
                            delta_cmd = fwe.step(DT, phi, alpha, theta)
                            data.ctrl[0] = delta_cmd
                        else:
                            data.ctrl[0] = 0.0

                        mujoco.mj_step(model, data)
                        step_n += 1

                        # Log
                        if step_n < 2000 or step_n % 10 == 0:
                            phase_id = list(Phase).index(fwe.phase)
                            log_f.write(
                                f'{data.time:.4f},{x[0]:.6f},{x[1]:.6f},{x[2]:.6f},'
                                f'{x[3]:.4f},{x[4]:.4f},{x[5]:.4f},'
                                f'{beta:.6f},{fwe.delta_target:.6f},'
                                f'{delta_actual:.6f},{phase_id}\n')

                    # Status every 500 steps
                    if step_n % 500 == 0 and step_n > 0:
                        x = get_state(model, data)
                        beta = (p1*x[1] + p2*x[2]) / (Mt*h_com)
                        print(f"  t={data.time:.2f}s phi={np.degrees(x[0]):+.1f} "
                              f"beta={np.degrees(beta):+.1f} "
                              f"delta={np.degrees(fwe.delta_target):+.1f} "
                              f"phase={fwe.phase_name}")

                viewer.sync()
                time.sleep(1/60)

        except (SystemExit, KeyboardInterrupt):
            pass
        finally:
            log_f.close()
            print(f"[LOG] {log_path}")
            print("[Done]")


if __name__ == '__main__':
    main()

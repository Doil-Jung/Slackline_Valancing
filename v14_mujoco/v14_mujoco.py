"""
V14 MuJoCo Slackline Balancing
==============================
Closed kinematic loop via MuJoCo equality constraint.
rope_a: kinematic chain (world → rope → ankle_a ball → body → hip hinge → upper)
rope_b: separate chain (world → rope → end) + equality connect → ankle_b on body
"""
import os, sys, time
import numpy as np
import mujoco
import mujoco.viewer

# ============================================================
#  Parameters — overridable via V14_* environment variables
# ============================================================
def _env(key, default):
    return float(os.environ.get(key, default))

M1 = _env('V14_M1', 30.0); M2 = _env('V14_M2', 40.0)
L1 = _env('V14_L1', 0.85); L2 = _env('V14_L2', 0.80)
BD = _env('V14_BD', 0.50)
LOWER_W = L2 / 4; UPPER_W = L2 / 4   # body thickness ∝ L2
B_THETA = _env('V14_BTHT', 3.0); TAU_MAX = _env('V14_TAUMAX', 1500.0); GRAV = 9.81

# Head (fixed on top of upper body)
M_HEAD = _env('V14_MHEAD', 5.0)
HEAD_R = L2 / 4   # head radius ∝ L2

L_POLE = _env('V14_LPOLE', 4.0); POLE_H = _env('V14_POLEH', 4.5)
R_TARGET = _env('V14_RTGT', 1.5)

# Derived
HALF_POLE = L_POLE / 2.0
ROPE_HORIZ = HALF_POLE - BD / 2.0
ROPE_VERT  = R_TARGET
FOOT_Z = POLE_H - ROPE_VERT

LC1 = L1/2
DT = 0.001
ALPHA0 = _env('V14_ALPHA0', 0.0); THETA0 = _env('V14_THETA0', 0.15)

Q_DIAG = [_env('V14_Q0', 1), _env('V14_Q1', 50), _env('V14_Q2', 50),
          _env('V14_Q3', 0.1), _env('V14_Q4', 5), _env('V14_Q5', 5)]
R_COST = _env('V14_RCOST', 0.001)
ROPE_MASS = 0.01

# Geometry sizes proportional to body scale
POLE_RADIUS = max(0.005, L2 * 0.05)     # pole cylinder radius
POLE_TOP_R  = max(0.008, L2 * 0.075)    # pole top sphere
ROPE_RADIUS = max(0.003, L2 * 0.015)    # rope capsule radius
HIP_R       = max(0.008, L2 * 0.08)     # hip cylinder radius
HIP_H       = max(0.005, BD * 0.15)     # hip cylinder half-height

# Effective upper body (body + head) for LQR
def compute_effective_upper():
    """Compute combined M2_eff, LC2_eff, I2_eff for upper body + head."""
    m2 = M2; m_h = M_HEAD
    d2 = L2/2; d_h = L2 + HEAD_R
    m_tot = m2 + m_h
    if m_tot < 0.01: m_tot = 0.01
    lc2 = (m2*d2 + m_h*d_h) / m_tot
    i2_body = m2*L2**2/12 + m2*(d2 - lc2)**2
    i2_head = 2/5*m_h*HEAD_R**2 + m_h*(d_h - lc2)**2
    return m_tot, lc2, i2_body + i2_head

M2_EFF, LC2_EFF, I2_EFF = compute_effective_upper()
I1 = M1*L1**2/12


def generate_mjcf(pole_b_dh=0.0):
    """Generate MJCF XML. pole_b_dh: pole_b height offset (negative = shorter)."""
    rh = ROPE_HORIZ; rv = ROPE_VERT; hp = HALF_POLE
    lw2 = LOWER_W/2; uw2 = UPPER_W/2; bd2 = BD/2
    POLE_B_H = POLE_H + pole_b_dh  # pole_b top height

    xml = f"""<?xml version="1.0"?>
<mujoco model="slackline_v14">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>

  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0.001"/>
  </default>

  <worldbody>
    <!-- Ground -->
    <geom type="plane" size="5 5 0.01" rgba="0.3 0.3 0.35 1"
          contype="1" conaffinity="1"/>
    <light pos="3 0 6" dir="-0.5 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="-3 0 6" dir="0.5 0 -1" diffuse="0.4 0.4 0.4"/>

    <!-- Pole A (static geoms) -->
    <geom name="pole_a_cyl" type="cylinder" pos="0 {hp} {POLE_H/2}"
          size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom name="pole_a_top" type="sphere" pos="0 {hp} {POLE_H}"
          size="{POLE_TOP_R}" rgba="0.9 0.9 0.9 1"/>

    <!-- Pole B (static geoms) -->
    <geom name="pole_b_cyl" type="cylinder" pos="0 {-hp} {POLE_B_H/2}"
          size="{POLE_RADIUS} {POLE_B_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom name="pole_b_top" type="sphere" pos="0 {-hp} {POLE_B_H}"
          size="{POLE_TOP_R}" rgba="0.9 0.9 0.9 1"/>

    <!-- ============ ROPE A CHAIN ============ -->
    <body name="rope_a_mount" pos="0 {hp} {POLE_H}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_a_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_a" type="capsule" size="{ROPE_RADIUS}"
            fromto="0 0 0 0 {-rh} {-rv}"
            rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>

      <!-- Lower body at ankle_a (rope_a endpoint) -->
      <body name="lower_body" pos="0 {-rh} {-rv}">
        <joint name="ankle" type="hinge" axis="0 1 0"/>  <!-- rigid 4-bar linkage: Y-only -->
        <geom name="lower_geom" type="box"
              size="{lw2} {bd2} {L1/2}"
              pos="0 {-bd2} {L1/2}"
              mass="{M1}" rgba="0.02 0.84 0.63 0.8"/>

        <!-- Ankle B target (for equality constraint) -->
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
          <!-- Head (fixed on upper body, no joint) -->
          <geom name="head" type="sphere"
                size="{HEAD_R}"
                pos="0 0 {L2 + HEAD_R}"
                mass="{M_HEAD}" rgba="1.0 0.85 0.7 0.9"/>
        </body>
      </body>
    </body>

    <!-- ============ ROPE B CHAIN ============ -->
    <body name="rope_b_mount" pos="0 {-hp} {POLE_B_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_b_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_b" type="capsule" size="{ROPE_RADIUS}"
            fromto="0 0 0 0 {rh} {-(POLE_B_H - FOOT_Z)}"
            rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>

      <!-- Rope B endpoint (for equality constraint) -->
      <body name="rope_b_end" pos="0 {rh} {-(POLE_B_H - FOOT_Z)}"/>
    </body>
  </worldbody>

  <!-- ============ CLOSED LOOP CONSTRAINT ============ -->
  <equality>
    <connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0"
             solref="0.001 1" solimp="0.999 0.999 0.0001"/>
  </equality>

  <!-- ============ ACTUATOR ============ -->
  <actuator>
    <motor name="hip_motor" joint="hip" ctrlrange="{-TAU_MAX} {TAU_MAX}"
           ctrllimited="true" gear="1"/>
  </actuator>
</mujoco>"""
    return xml


# ============================================================
#  LQR (same as V10/V12/V13)
# ============================================================
def compute_lqr():
    from scipy.linalg import solve_discrete_are, expm
    R = R_TARGET
    # Use effective upper body (body + head)
    m2e = M2_EFF; lc2e = LC2_EFF; i2e = I2_EFF
    p1 = M1*LC1 + m2e*L1; p2 = m2e*lc2e; p3 = m2e*L1*lc2e; Mt = M1+m2e
    print(f"[LQR] M2_eff={m2e:.1f}kg  LC2_eff={lc2e:.3f}m  I2_eff={i2e:.3f}")
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
    Q = np.diag(Q_DIAG); Rc = np.array([[R_COST]])
    P = solve_discrete_are(Ad, Bd, Q, Rc)
    K = np.linalg.inv(Rc + Bd.T@P@Bd) @ (Bd.T@P@Ad)
    K[0,0] *= -1; K[0,3] *= -1  # phi convention
    eigs_c = sorted(abs(e) for e in np.linalg.eigvals(Ad - Bd@K))
    print(f"[LQR] K = [{', '.join(f'{k:.1f}' for k in K[0])}]")
    print(f"[LQR] Close |eig|: {[f'{e:.4f}' for e in eigs_c]}")
    if max(eigs_c) > 1.0:
        print(f"[WARNING] Unstable! max|eig|={max(eigs_c):.4f} > 1.0 -- control will FAIL")
    return K[0]


# ============================================================
#  State extraction
# ============================================================
def get_state(model, data):
    """Extract [phi, alpha, theta, phi_dot, alpha_dot, theta_dot]."""
    phi = data.joint('phi_Y').qpos[0]
    phi_dot = data.joint('phi_Y').qvel[0]

    # Lower body world orientation via xmat
    lb_id = model.body('lower_body').id
    xmat = data.xmat[lb_id].reshape(3,3)
    # alpha = pitch (Y rotation): sin(alpha) = -xmat[2,0] for small angles
    # More robust: use atan2
    alpha = np.arctan2(-xmat[2,0], xmat[2,2])

    ub_id = model.body('upper_body').id
    xmat_u = data.xmat[ub_id].reshape(3,3)
    theta = np.arctan2(-xmat_u[2,0], xmat_u[2,2])

    # Angular velocities from cvel (6D: [angular(3), linear(3)])
    lb_angvel = data.cvel[lb_id][:3]  # first 3 = angular velocity (world frame)
    ub_angvel = data.cvel[ub_id][:3]
    alpha_dot = lb_angvel[1]  # Y component
    theta_dot = ub_angvel[1]

    return np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])


def get_x_roll(model, data):
    """Get X-axis roll of lower body (for diagnostics)."""
    lb_id = model.body('lower_body').id
    xmat = data.xmat[lb_id].reshape(3,3)
    return np.arctan2(xmat[2,1], xmat[2,2])


# ============================================================
#  Main
# ============================================================
def main():
    global M2_EFF, LC2_EFF, I2_EFF, I1, LC1

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dh', type=float, default=0.0,
                        help='Pole B height offset (m). e.g. --dh -0.3')
    args = parser.parse_args()

    pole_b_dh = args.dh
    # Recalculate ALL derived params (env vars loaded at module level)
    LC1 = L1 / 2
    I1 = M1 * L1**2 / 12
    M2_EFF, LC2_EFF, I2_EFF = compute_effective_upper()

    print("="*60)
    print("  V14 MuJoCo Slackline Balancing")
    print(f"  m1={M1} m2={M2}  L1={L1} L2={L2}  R={R_TARGET}")
    print(f"  Head: {M_HEAD}kg  r={HEAD_R}m")
    print(f"  L_pole={L_POLE} foot_z={FOOT_Z:.1f}  BD={BD}")
    if pole_b_dh != 0:
        print(f"  *** Pole B height: {POLE_H + pole_b_dh:.2f}m (dh={pole_b_dh:+.2f}m) ***")
    print(f"  Q={Q_DIAG}  R={R_COST}  dt={DT}")
    print("="*60)

    K_lqr = compute_lqr()

    # Build model
    xml = generate_mjcf(pole_b_dh=pole_b_dh)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # Set initial pose
    hip_jid = model.joint('hip').id
    hip_qadr = model.jnt_qposadr[hip_jid]
    ankle_jid = model.joint('ankle').id
    ankle_qadr = model.jnt_qposadr[ankle_jid]

    data.qpos[hip_qadr] = THETA0 - ALPHA0  # hip = relative angle
    data.qpos[ankle_qadr] = ALPHA0

    mujoco.mj_forward(model, data)

    x0 = get_state(model, data)
    print(f"[Init] phi={np.degrees(x0[0]):+.1f} alpha={np.degrees(x0[1]):+.1f} "
          f"theta={np.degrees(x0[2]):+.1f}")

    # CSV log
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v14_log.csv')
    log_f = open(log_path, 'w')
    log_f.write('t,phi,alpha,theta,phi_dot,alpha_dot,theta_dot,tau,x_roll\n')

    # Simulation state
    paused = True
    ctrl_on = False
    step_n = 0

    print("[Sim] Space=Pause  R=Reset  C=Toggle_Ctrl  Q=Quit")
    print(f"[Sim] Controller OFF (press C)")


    def reset():
        nonlocal step_n
        mujoco.mj_resetData(model, data)
        data.qpos[hip_qadr] = THETA0 - ALPHA0
        data.qpos[ankle_qadr] = ALPHA0
        mujoco.mj_forward(model, data)
        step_n = 0
        xr = np.degrees(get_x_roll(model, data))
        err = np.linalg.norm(data.xpos[model.body('rope_b_end').id] -
                             data.xpos[model.body('ankle_b_target').id])
        print(f"[RESET] theta0={np.degrees(THETA0):.1f}° X-roll={xr:+.1f}° err={err:.4f}m")

    def key_callback(keycode):
        nonlocal paused, ctrl_on
        if keycode == 32:  # Space
            paused = not paused
            if paused:
                x = get_state(model, data)
                xr = np.degrees(get_x_roll(model, data))
                print(f"[PAUSE] t={data.time:.2f}s phi={np.degrees(x[0]):+.1f} "
                      f"a={np.degrees(x[1]):+.1f} th={np.degrees(x[2]):+.1f} "
                      f"Xroll={xr:+.1f}")
            else:
                tag = "LQR" if ctrl_on else "FREE"
                print(f"[RUN] {tag}")
        elif keycode == ord('R') or keycode == ord('r'):
            reset()
            paused = True
        elif keycode == ord('C') or keycode == ord('c'):
            ctrl_on = not ctrl_on
            if not ctrl_on:
                data.ctrl[0] = 0
            print(f"[CTRL] {'ON' if ctrl_on else 'OFF'}")
        elif keycode == ord('Q') or keycode == ord('q'):
            raise SystemExit

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.distance = 6.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        viewer.cam.lookat[:] = [0, 0, FOOT_Z]

        try:
            while viewer.is_running():
                if not paused:
                    # Steps per frame
                    STEPS_PER_FRAME = 24
                    for _ in range(STEPS_PER_FRAME):
                        x = get_state(model, data)
                        x_roll = get_x_roll(model, data)

                        if np.any(np.isnan(x)):
                            print(f"[!] NaN at t={data.time:.3f}")
                            paused = True
                            break

                        tau = 0.0
                        if ctrl_on:
                            if abs(x[1]) < np.radians(60) and abs(x[2]) < np.radians(60):
                                tau = -float(K_lqr @ x)
                                tau = np.clip(tau, -TAU_MAX, TAU_MAX)
                        data.ctrl[0] = tau

                        mujoco.mj_step(model, data)
                        step_n += 1

                        # Log
                        if step_n < 1000 or step_n % 10 == 0:
                            t = data.time
                            log_f.write(f'{t:.4f},{x[0]:.6f},{x[1]:.6f},{x[2]:.6f},'
                                        f'{x[3]:.4f},{x[4]:.4f},{x[5]:.4f},'
                                        f'{tau:.2f},{x_roll:.6f}\n')

                    # Print status periodically
                    if step_n % 1000 == 0 and step_n > 0:
                        x = get_state(model, data)
                        xr = np.degrees(get_x_roll(model, data))
                        print(f"  t={data.time:.1f}s phi={np.degrees(x[0]):+.1f} "
                              f"a={np.degrees(x[1]):+.1f} th={np.degrees(x[2]):+.1f} "
                              f"tau={data.ctrl[0]:+.0f} Xroll={xr:+.1f}")

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

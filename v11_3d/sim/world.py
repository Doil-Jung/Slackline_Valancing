"""
Slackline V11 - MuJoCo World (v3: simplified, no platform)

Body tree:
  worldbody
    +-- pole_a (static, origin at top)
    +-- pole_b (static, origin at top)
    +-- rope_a (hinge at pole_a_top, capsule down to lower_body's +y edge)
          +-- lower_body (hinge at rope_a tip = foot's +y edge)
                +-- upper_body (hip hinge at lower_body top center)
                +-- rope_b (hinge at lower_body's -y edge, capsule up)
                      ~~ connect constraint to pole_b_top ~~

No platform needed: lower_body edges ARE the rope attachment points.
"""
import mujoco
import numpy as np
import os
from config import V11Config


def _build_detail_upper(L2, bd):
    """Generate detailed upper body hardware geoms (OpenSCAD V11 layout).

    Coordinate frame: origin at hip joint, z-up.
    Hardware dimensions from OpenSCAD model (mm → m).
    Positions scale with L2 relative to real 200mm upper body.
    """
    # Motor
    mx, my, mz = 0.0247, 0.035, 0.0452
    mzup, mzdn = 0.0337, 0.0115
    pt = 0.004
    ux = mx + pt * 2   # 0.0327 frame width
    cy = 0.060          # column depth (y)
    hd, hh = 0.020, 0.003  # horn

    su = L2 / 0.200  # scale factor
    zc = (mzup - mzdn) / 2  # motor z center = 0.0111
    col_bot = (mzup + 0.020) * su
    col_cz = (col_bot + L2) / 2
    col_hz = (L2 - col_bot) / 2

    g = []
    # Motor body
    g.append(f'<geom name="motor" type="box" size="{mx/2:.5f} {my/2:.5f} {mz/2:.5f}" '
             f'pos="0 0 {zc*su:.5f}" rgba="0.18 0.20 0.21 1" contype="0" conaffinity="0"/>')
    # Motor horns
    for s, n in [(1, 'horn_f'), (-1, 'horn_b')]:
        y = s * (my / 2 + hh / 2 + 0.001)
        g.append(f'<geom name="{n}" type="cylinder" size="{hd/2:.5f} {hh/2:.5f}" '
                 f'pos="0 {y:.5f} 0" euler="1.5708 0 0" '
                 f'rgba="1.0 0.84 0.0 1" contype="0" conaffinity="0"/>')
    # Frame side plates
    for s, n in [(1, 'l'), (-1, 'r')]:
        x = s * (mx / 2 + pt / 2)
        g.append(f'<geom name="uf_{n}" type="box" '
                 f'size="{pt/2:.5f} {(my+0.004)/2:.5f} {mz/2:.5f}" '
                 f'pos="{x:.5f} 0 {zc*su:.5f}" '
                 f'rgba="0.68 0.85 0.90 0.4" contype="0" conaffinity="0"/>')
    # Frame top plate
    g.append(f'<geom name="uf_top" type="box" '
             f'size="{ux/2:.5f} {(my+0.004)/2:.5f} {pt/2:.5f}" '
             f'pos="0 0 {(mzup+pt/2)*su:.5f}" '
             f'rgba="0.68 0.85 0.90 0.4" contype="0" conaffinity="0"/>')
    # Frame column
    if col_hz > 0:
        g.append(f'<geom name="uf_col" type="box" '
                 f'size="{ux/2:.5f} {cy/2:.5f} {col_hz:.5f}" '
                 f'pos="0 0 {col_cz:.5f}" '
                 f'rgba="0.68 0.85 0.90 0.35" contype="0" conaffinity="0"/>')
    # ESP32 (+x face, upper)
    ed, ew, eh = 0.012, 0.050, 0.065
    g.append(f'<geom name="esp32" type="box" '
             f'size="{ed/2:.5f} {ew/2:.5f} {eh/2:.5f}" '
             f'pos="{ux/2+ed/2:.5f} 0 {0.160*su:.5f}" '
             f'rgba="0.13 0.55 0.13 1" contype="0" conaffinity="0"/>')
    # URT1 (+x face, lower)
    ud, uw, uh = 0.012, 0.037, 0.056
    g.append(f'<geom name="urt1" type="box" '
             f'size="{ud/2:.5f} {uw/2:.5f} {uh/2:.5f}" '
             f'pos="{ux/2+ud/2:.5f} 0 {0.090*su:.5f}" '
             f'rgba="0.85 0.65 0.13 1" contype="0" conaffinity="0"/>')
    # Power bank (-x, cylinder)
    pd, ph = 0.022, 0.092
    g.append(f'<geom name="powerbank" type="cylinder" '
             f'size="{pd/2:.5f} {ph/2:.5f}" '
             f'pos="{-ux/2-pd/2:.5f} 0 {0.155*su:.5f}" '
             f'rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0"/>')
    # LiPo battery (-x)
    btd, btw, bth = 0.015, 0.030, 0.061
    g.append(f'<geom name="lipo" type="box" '
             f'size="{btd/2:.5f} {btw/2:.5f} {bth/2:.5f}" '
             f'pos="{-ux/2-btd/2:.5f} 0 {0.065*su:.5f}" '
             f'rgba="0.25 0.41 0.88 1" contype="0" conaffinity="0"/>')
    # Upper MPU6050 (+y face)
    mw, mhh, md = 0.016, 0.021, 0.003
    g.append(f'<geom name="mpu_up" type="box" '
             f'size="{mw/2:.5f} {md/2:.5f} {mhh/2:.5f}" '
             f'pos="0 {cy/2+md/2:.5f} {0.060*su:.5f}" '
             f'rgba="0.86 0.08 0.24 1" contype="0" conaffinity="0"/>')

    return ('\n          ').join(g)


def _build_detail_lower(L1, bd):
    """Generate detailed lower body hardware geoms (OpenSCAD V11 layout).

    Coordinate frame: origin at ankle (rope contact = +y edge bottom), z-up.
    Hip is at z=L1 in this frame.
    """
    bw = 0.024          # bracket/frame width (x)
    bi, bt, ba = 0.042, 0.002, 0.030  # bracket inner_y, thick, arm_l
    cy = 0.060          # column depth (y)
    pt = 0.004

    R2H = 0.230   # rope-to-hip distance in real model
    sl = L1 / R2H
    yc = -bd / 2  # body center y in MuJoCo frame

    def zof(zh):
        """Convert z from hip (OpenSCAD) to MuJoCo lower_body frame."""
        return L1 + zh * sl

    g = []
    bbot = -ba - bt  # bracket bottom z from hip = -0.032

    # U-bracket base plate
    g.append(f'<geom name="bkt_base" type="box" '
             f'size="{bw/2:.5f} {(bi+2*bt)/2:.5f} {bt/2:.5f}" '
             f'pos="0 {yc:.5f} {zof(bbot+bt/2):.5f}" '
             f'rgba="0.75 0.75 0.75 1" contype="0" conaffinity="0"/>')
    # U-bracket arms
    for s, n in [(1, 'f'), (-1, 'b')]:
        g.append(f'<geom name="bkt_{n}" type="box" '
                 f'size="{bw/2:.5f} {bt/2:.5f} {ba/2:.5f}" '
                 f'pos="0 {yc + s*(bi/2+bt/2):.5f} {zof(-ba/2):.5f}" '
                 f'rgba="0.75 0.75 0.75 1" contype="0" conaffinity="0"/>')
    # Lower frame top plate
    g.append(f'<geom name="lf_top" type="box" '
             f'size="{bw/2:.5f} {(bi+2*bt)/2:.5f} {pt/2:.5f}" '
             f'pos="0 {yc:.5f} {zof(bbot - pt):.5f}" '
             f'rgba="0.47 0.47 0.47 0.4" contype="0" conaffinity="0"/>')
    # Lower frame column
    ct_z = bbot - 0.024   # column top: -0.056m from hip
    cb_z = -R2H + 0.005   # column bottom: near rope
    zt, zb = zof(ct_z), zof(cb_z)
    if zt > zb:
        g.append(f'<geom name="lf_col" type="box" '
                 f'size="{bw/2:.5f} {cy/2:.5f} {(zt-zb)/2:.5f}" '
                 f'pos="0 {yc:.5f} {(zt+zb)/2:.5f}" '
                 f'rgba="0.47 0.47 0.47 0.35" contype="0" conaffinity="0"/>')
    # Lower MPU6050 (+y face)
    mw, mhh, md = 0.016, 0.021, 0.003
    g.append(f'<geom name="mpu_lo" type="box" '
             f'size="{mw/2:.5f} {md/2:.5f} {mhh/2:.5f}" '
             f'pos="0 {yc + cy/2 + md/2:.5f} {zof(-0.072):.5f}" '
             f'rgba="0.86 0.08 0.24 1" contype="0" conaffinity="0"/>')

    return ('\n        ').join(g)


def build_mjcf(config: V11Config) -> str:
    rp = config.rope
    rb = config.robot
    sim = config.sim

    half_pole = rp.L_pole / 2.0
    bd = rb.body_depth  # distance between two foot edges (y direction)
    half_rope = rp.L0 / 2.0  # each rope segment length

    # Rope geometry: pole_a_top -> foot_front (+y edge of lower_body bottom)
    # Horizontal distance (y): from pole to foot edge
    rope_horiz = half_pole - bd / 2.0  # 2.0 - 0.125 = 1.875
    # Vertical drop: from Pythagoras with rope length
    rope_vert = np.sqrt(max(half_rope**2 - rope_horiz**2, 0.01))  # sqrt(6.25 - 3.516) = 1.654
    rope_len = half_rope  # 2.5m each

    # Foot height (absolute z)
    foot_z = rp.pole_height - rope_vert

    # Rope direction vectors (in rope_a's local frame, joint at pole_a_top = origin)
    # rope_a goes: (0,0,0) -> (0, -rope_horiz, -rope_vert)
    ra_end = [0, -rope_horiz, -rope_vert]

    # rope_b goes: from lower_body's -y edge (0, -bd, 0) relative to ankle
    # to pole_b_top (absolute: 0, -half_pole, pole_height)
    # pole_b in foot_back frame: (0, -(half_pole - bd/2), +rope_vert)
    rb_end = [0, -rope_horiz, rope_vert]

    rope_radius = 0.012
    rope_mass = max(rp.rope_mass / 2.0, 0.01)

    # Joint xml based on mode
    def make_joints(name):
        if sim.mode == 'A':
            return f"""<joint name="{name}_x" type="hinge" axis="1 0 0" damping="0.05"/>
            <joint name="{name}_y" type="hinge" axis="0 1 0" damping="0.05"/>
            <joint name="{name}_z" type="hinge" axis="0 0 1" damping="0.05"/>"""
        else:
            return f'<joint name="{name}" type="hinge" axis="0 1 0" damping="0.05"/>'

    def box_diag(mass, wx, wy, wz):
        ixx = mass * (wy**2 + wz**2) / 12.0
        iyy = mass * (wx**2 + wz**2) / 12.0
        izz = mass * (wx**2 + wy**2) / 12.0
        return f"{ixx:.6f} {iyy:.6f} {izz:.6f}"

    # Lower body center offset from ankle joint (which is at +y edge bottom)
    # The ankle joint is at the +y bottom edge of lower_body
    # lower_body center: shifted -y by bd/2 (to body center), up by L1/2
    lb_cx, lb_cy, lb_cz = 0, -bd / 2, rb.L1 / 2

    # Hip position relative to ankle: at top-center of lower_body
    hip_dx, hip_dy, hip_dz = 0, -bd / 2, rb.L1

    # Upper body center offset from hip joint
    ub_cx, ub_cy, ub_cz = 0, 0, rb.L2 / 2

    # Rope_b joint position relative to ankle: at -y bottom edge
    rb_dx, rb_dy, rb_dz = 0, -bd, 0

    # Build detailed hardware visual geometry (from OpenSCAD V11 model)
    upper_detail_geoms = _build_detail_upper(rb.L2, bd)
    lower_detail_geoms = _build_detail_lower(rb.L1, bd)

    xml = f"""<mujoco model="slackline_v11">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -{sim.gravity}" integrator="Euler"
          solver="Newton" iterations="50" tolerance="1e-10"/>
  <default>
    <joint armature="0.001"/>
    <geom condim="3" friction="1.0 0.5 0.1"/>
  </default>

  <worldbody>
    <!-- Ground -->
    <geom name="ground" type="plane" size="10 10 0.1" rgba="0.12 0.12 0.22 1"/>

    <!-- Pole A (origin = top, where rope ties) -->
    <body name="pole_a" pos="0 {half_pole} {rp.pole_height}">
      <geom name="pole_a_cyl" type="cylinder" size="0.04 {rp.pole_height/2}"
            pos="0 0 {-rp.pole_height/2}" rgba="0.5 0.5 0.5 1"
            contype="0" conaffinity="0"/>
      <geom name="pole_a_cap" type="sphere" size="0.06"
            rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
    </body>

    <!-- Pole B (origin = top, where rope ties) -->
    <body name="pole_b" pos="0 {-half_pole} {rp.pole_height}">
      <geom name="pole_b_cyl" type="cylinder" size="0.04 {rp.pole_height/2}"
            pos="0 0 {-rp.pole_height/2}" rgba="0.5 0.5 0.5 1"
            contype="0" conaffinity="0"/>
      <geom name="pole_b_cap" type="sphere" size="0.06"
            rgba="0.9 0.9 0.9 1" contype="0" conaffinity="0"/>
    </body>

    <!-- ===== Kinematic chain: rope_a -> lower -> upper + rope_b ===== -->

    <!-- Rope A: joint at pole_a_top, capsule hangs down to lower_body -->
    <body name="rope_a" pos="0 {half_pole} {rp.pole_height}">
      {make_joints("ra")}
      <geom name="rope_a_geom" type="capsule" size="{rope_radius}"
            fromto="0 0 0  {ra_end[0]} {ra_end[1]} {ra_end[2]}"
            rgba="1.0 0.8 0.0 1" mass="{rope_mass}"
            contype="0" conaffinity="0"/>
      <inertial pos="0 {ra_end[1]/2} {ra_end[2]/2}" mass="{rope_mass}"
                diaginertia="{rope_mass*rope_len**2/12:.8f} {rope_mass*rope_len**2/12:.8f} {rope_mass*rope_radius**2/2:.8f}"/>

      <!-- Lower body: joint at rope_a endpoint (= foot's +y edge) -->
      <body name="lower_body" pos="{ra_end[0]} {ra_end[1]} {ra_end[2]}">
        <joint name="ankle" type="hinge" axis="0 1 0" damping="{rb.b_alpha}"/>
        {lower_detail_geoms}
        <inertial pos="{lb_cx} {lb_cy} {lb_cz}" mass="{rb.m1}"
                  diaginertia="{box_diag(rb.m1, rb.lower_width, bd, rb.L1)}"/>

        <!-- Upper body: hip joint at top-center of lower body -->
        <body name="upper_body" pos="{hip_dx} {hip_dy} {hip_dz}">
          <joint name="hip" type="hinge" axis="0 1 0" damping="{rb.b_theta}"/>
          {upper_detail_geoms}
          <inertial pos="{ub_cx} {ub_cy} {ub_cz}" mass="{rb.m2}"
                    diaginertia="{box_diag(rb.m2, rb.upper_width, bd, rb.L2)}"/>
          <!-- Hip visual marker -->
          <geom name="hip_marker" type="cylinder"
                size="0.03 {bd*0.7}" euler="1.5708 0 0"
                rgba="1.0 0.75 0.04 1" contype="0" conaffinity="0"/>
        </body>

        <!-- Rope B: joint at lower_body's -y edge (= foot's -y edge) -->
        <body name="rope_b" pos="{rb_dx} {rb_dy} {rb_dz}">
          {make_joints("rb")}
          <geom name="rope_b_geom" type="capsule" size="{rope_radius}"
                fromto="0 0 0  {rb_end[0]} {rb_end[1]} {rb_end[2]}"
                rgba="1.0 0.8 0.0 1" mass="{rope_mass}"
                contype="0" conaffinity="0"/>
          <inertial pos="0 {rb_end[1]/2} {rb_end[2]/2}" mass="{rope_mass}"
                    diaginertia="{rope_mass*rope_len**2/12:.8f} {rope_mass*rope_len**2/12:.8f} {rope_mass*rope_radius**2/2:.8f}"/>
        </body>
      </body>
    </body>
  </worldbody>

  <!-- Rope B tip connects to Pole B top -->
  <equality>
    <connect body1="rope_b" anchor="{rb_end[0]} {rb_end[1]} {rb_end[2]}"
             body2="pole_b"
             solref="0.005 1"/>
  </equality>

  <!-- Hip torque actuator -->
  <actuator>
    <motor name="hip_torque" joint="hip" gear="1"
           ctrlrange="{-rb.tau_max} {rb.tau_max}"/>
  </actuator>

  <!-- Sensors -->
  <sensor>
    <jointpos name="s_ankle_pos" joint="ankle"/>
    <jointvel name="s_ankle_vel" joint="ankle"/>
    <jointpos name="s_hip_pos" joint="hip"/>
    <jointvel name="s_hip_vel" joint="hip"/>
    <framepos name="s_foot_pos" objtype="body" objname="lower_body"/>
    <framequat name="s_foot_quat" objtype="body" objname="lower_body"/>
    <frameangvel name="s_foot_angvel" objtype="body" objname="lower_body"/>
    <subtreecom name="s_robot_com" body="lower_body"/>
  </sensor>

  <!-- Keyframe: initial pose for Backspace reset -->
  <!-- qpos: rope_a joints, ankle, hip, rope_b joints -->
  <keyframe>
    <key name="init" qpos="{_build_keyframe_qpos(sim.mode, rb.alpha0, rb.theta0)}"/>
  </keyframe>
</mujoco>"""
    return xml


def _build_keyframe_qpos(mode, alpha0, theta0):
    """Build qpos string for keyframe"""
    hip_rel = theta0 - alpha0
    if mode == 'A':
        # 3 rope_a joints + ankle + hip + 3 rope_b joints = 8
        vals = [0, 0, 0, alpha0, hip_rel, 0, 0, 0]
    else:
        # 1 rope_a joint + ankle + hip + 1 rope_b joint = 4
        vals = [0, alpha0, hip_rel, 0]
    return ' '.join(f'{v:.6f}' for v in vals)


class SlacklineWorld:
    """MuJoCo slackline world"""

    def __init__(self, config: V11Config):
        self.config = config
        self.model = None
        self.data = None
        self._viewer = None
        self.ankle_jnt_id = -1
        self.hip_jnt_id = -1
        self.hip_act_id = -1

    def initialize(self):
        xml_str = build_mjcf(self.config)

        # Save XML for inspection
        xml_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
        os.makedirs(xml_dir, exist_ok=True)
        xml_path = os.path.join(xml_dir, 'generated_scene.xml')
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        print(f"[World] XML saved: {os.path.abspath(xml_path)}")

        try:
            self.model = mujoco.MjModel.from_xml_string(xml_str)
        except Exception as e:
            print(f"[World] XML error: {e}")
            for i, line in enumerate(xml_str.split('\n'), 1):
                print(f"  {i:3d}: {line}")
            raise

        self.data = mujoco.MjData(self.model)

        self.ankle_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'ankle')
        self.hip_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'hip')
        self.hip_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'hip_torque')

        # Set initial angles in both qpos0 (for Backspace reset) and data.qpos
        rb = self.config.robot
        ankle_qi = self.model.jnt_qposadr[self.ankle_jnt_id]
        hip_qi = self.model.jnt_qposadr[self.hip_jnt_id]
        hip_rel = rb.theta0 - rb.alpha0

        # qpos0 = default pose used by Backspace reset (mj_resetData)
        self.model.qpos0[ankle_qi] = rb.alpha0
        self.model.qpos0[hip_qi] = hip_rel

        # Also set current data
        self.data.qpos[ankle_qi] = rb.alpha0
        self.data.qpos[hip_qi] = hip_rel

        mujoco.mj_forward(self.model, self.data)

        print(f"[World] OK: nq={self.model.nq}, nv={self.model.nv}, "
              f"njnt={self.model.njnt}, nu={self.model.nu}, neq={self.model.neq}")
        print(f"[World] Mode: {'A (3D free)' if self.config.sim.mode == 'A' else 'B (rolling only)'}")
        return self.model, self.data

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def apply_hip_torque(self, tau):
        tau = np.clip(tau, -self.config.robot.tau_max, self.config.robot.tau_max)
        self.data.ctrl[self.hip_act_id] = tau

    def get_state(self):
        aq = self.model.jnt_qposadr[self.ankle_jnt_id]
        av = self.model.jnt_dofadr[self.ankle_jnt_id]
        hq = self.model.jnt_qposadr[self.hip_jnt_id]
        hv = self.model.jnt_dofadr[self.hip_jnt_id]

        alpha = float(self.data.qpos[aq])
        alpha_dot = float(self.data.qvel[av])
        hip_rel = float(self.data.qpos[hq])
        hip_rel_dot = float(self.data.qvel[hv])
        theta = alpha + hip_rel
        theta_dot = alpha_dot + hip_rel_dot

        # Lower body (foot reference) body state
        lb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'lower_body')
        ub_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'upper_body')
        foot_pos = self.data.xpos[lb_id].copy()
        foot_quat = self.data.xquat[lb_id].copy()
        euler = quat_to_euler(foot_quat)

        phi = euler[1]
        pitch = euler[0]
        yaw = euler[2]

        # Angular velocity from sensor
        phi_dot = pitch_dot = yaw_dot = 0.0
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 's_foot_angvel')
        if sid >= 0:
            adr = self.model.sensor_adr[sid]
            av3 = self.data.sensordata[adr:adr+3]
            pitch_dot, phi_dot, yaw_dot = float(av3[0]), float(av3[1]), float(av3[2])

        m1, m2 = self.config.robot.m1, self.config.robot.m2
        com_pos = (m1 * self.data.xpos[lb_id] + m2 * self.data.xpos[ub_id]) / (m1 + m2)

        return {
            'phi': phi, 'phi_dot': phi_dot,
            'alpha': alpha, 'alpha_dot': alpha_dot,
            'theta': theta, 'theta_dot': theta_dot,
            'hip_angle': hip_rel,
            'pitch': pitch, 'pitch_dot': pitch_dot,
            'yaw': yaw, 'yaw_dot': yaw_dot,
            'foot_pos': foot_pos,
            'com_pos': com_pos.copy(),
            'hip_pos': self.data.xpos[lb_id].copy(),
            'head_pos': self.data.xpos[ub_id].copy(),
            'time': self.data.time,
        }

    def apply_perturbation(self, impulse, axis='theta'):
        if axis == 'theta':
            vi = self.model.jnt_dofadr[self.hip_jnt_id]
            self.data.qvel[vi] += impulse
        elif axis == 'pitch':
            uid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'upper_body')
            self.data.xfrc_applied[uid, 3] += impulse * 10
        elif axis == 'yaw':
            uid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'upper_body')
            self.data.xfrc_applied[uid, 5] += impulse * 10

    def launch_viewer(self):
        try:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return True
        except Exception as e:
            print(f"[World] Viewer failed: {e}")
            return False

    def sync_viewer(self):
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            return True
        return False

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except:
                pass
            self._viewer = None


def quat_to_euler(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    sinp = np.clip(2*(w*y - z*x), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return [roll, pitch, yaw]

"""
V11 Parameter Launcher - PyBullet Backend (STL + 6-DOF LQR Control)

Ported from v10's 6-state LQR:
  State: x = [phi, alpha, theta, phi_dot, alpha_dot, theta_dot]
     phi:   foot position angle (rope swing)
     alpha: lower body absolute tilt from vertical
     theta: upper body absolute tilt from vertical
  Control: u = hip_torque (1 input)

STL convention:
  - Both UpperBlock/LowerBlock have origin at hip (motor axis center)
  - Upper extends Z+, Lower extends Z-
  - Rotation axis = Y

Kinematic chain:
  base (pole_a_top, fixed)
    link 0: rope_a (hinge Y)  -> phi
      link 1: lower_body (hinge Y = ankle)  -> alpha_rel
        link 2: upper_body (hinge Y = hip)  -> theta_rel
        link 3: rope_b (hinge Y)
          -> P2P constraint -> pole_b_top
"""
import sys
import os
import json
import shutil
import struct
import tempfile

# ── Conda DLL paths ──
_conda_env = os.path.dirname(sys.executable)
_conda_lib_bin = os.path.join(_conda_env, "Library", "bin")
if os.path.isdir(_conda_lib_bin) and _conda_lib_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _conda_lib_bin + os.pathsep + \
                          os.path.join(_conda_env, "Library", "lib") + os.pathsep + \
                          _conda_env + os.pathsep + \
                          os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_conda_lib_bin)
        os.add_dll_directory(os.path.join(_conda_env, "Library", "lib"))

import tkinter as tk
from tkinter import ttk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_TEMP_DIR = os.path.join(tempfile.gettempdir(), "slackline_sim")


# ================================================================
#  STL / Model utilities
# ================================================================

def read_stl_bounds(filepath):
    verts = []
    with open(filepath, 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        for _ in range(n):
            f.read(12)
            for _ in range(3):
                v = struct.unpack('<fff', f.read(12))
                verts.append(v)
            f.read(2)
    arr = np.array(verts)
    return arr.min(axis=0), arr.max(axis=0)


def scan_stl_models(directory):
    models = {}
    for f in os.listdir(directory):
        if not f.lower().endswith('.stl'):
            continue
        if '_UpperBlock' in f:
            prefix = f.split('_UpperBlock')[0]
            models.setdefault(prefix, {})['upper'] = os.path.join(directory, f)
        elif '_LowerBlock' in f:
            prefix = f.split('_LowerBlock')[0]
            models.setdefault(prefix, {})['lower'] = os.path.join(directory, f)
    return {k: v for k, v in models.items() if 'upper' in v and 'lower' in v}


def get_model_dims(upper_path, lower_path):
    u_min, u_max = read_stl_bounds(upper_path)
    l_min, l_max = read_stl_bounds(lower_path)
    return {
        'L1': abs(l_min[2]) / 1000.0,
        'L2': abs(u_max[2]) / 1000.0,
        'body_depth': 2.0 * max(abs(u_min[1]), abs(l_min[1])) / 1000.0,
        'upper_w': (u_max[0] - u_min[0]) / 1000.0,
        'lower_w': (l_max[0] - l_min[0]) / 1000.0,
        'upper_bounds': (u_min / 1000.0, u_max / 1000.0),
        'lower_bounds': (l_min / 1000.0, l_max / 1000.0),
    }


def load_physics_json(prefix):
    path = os.path.join(_SRC_DIR, f"{prefix}_physics.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def copy_stl_to_temp(src_path):
    os.makedirs(_TEMP_DIR, exist_ok=True)
    name = os.path.basename(src_path)
    dst = os.path.join(_TEMP_DIR, name)
    if not os.path.exists(dst) or os.path.getmtime(src_path) > os.path.getmtime(dst):
        shutil.copy2(src_path, dst)
    return dst


def safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', 'replace').decode())


# ================================================================
#  6-DOF LQR (ported from v10 — discrete DARE)
# ================================================================

def compute_lqr_gain_6dof(m1, m2, R_rope, L1, lc1, lc2, I1_y, I2_y,
                           g=9.81, b_phi=0.1, b_alpha=0.1, b_theta=0.1,
                           Q_diag=None, R_val=None, dt=1.0/240.0):
    """Compute 6-state discrete LQR (DARE) for slackline robot.

    State: x = [phi, alpha, theta, phi_dot, alpha_dot, theta_dot]
    Control: u = hip_torque

    Uses discrete-time algebraic Riccati equation (DARE) matching v10.
    R_rope should be the rope curvature radius (NOT rope length).

    Returns: K (1D array, length 6) or None
    """
    try:
        from scipy.linalg import solve_discrete_are, expm
    except ImportError:
        safe_print("[LQR] scipy not available")
        return None

    if Q_diag is None:
        Q_diag = [1, 50, 50, 0.1, 5, 5]
    if R_val is None:
        R_val = 0.001

    # Physical constants (v10 formulation)
    p1 = m1 * lc1 + m2 * L1
    p2 = m2 * lc2
    p3 = m2 * L1 * lc2
    Mt = m1 + m2

    # 3x3 Mass matrix at equilibrium
    M_eq = np.array([
        [Mt * R_rope**2,   R_rope * p1,                   R_rope * p2],
        [R_rope * p1,      m1*lc1**2 + I1_y + m2*L1**2,   p3         ],
        [R_rope * p2,      p3,                             m2*lc2**2 + I2_y],
    ])

    # Gravity (phi: stabilizing, alpha/theta: destabilizing)
    G_mat = np.array([
        [-Mt * g * R_rope, 0,       0      ],
        [0,                 p1 * g,  0      ],
        [0,                 0,       p2 * g ],
    ])

    # Damping
    D_mat = np.diag([-b_phi, -b_alpha, -b_theta])

    # Input (hip torque)
    E_mat = np.array([[0], [-1], [1]])

    # Continuous state-space
    M_inv = np.linalg.inv(M_eq)
    MinvG = M_inv @ G_mat
    MinvD = M_inv @ D_mat
    MinvE = M_inv @ E_mat

    n = 6
    Ac = np.zeros((n, n))
    Bc = np.zeros((n, 1))
    Ac[0:3, 3:6] = np.eye(3)
    Ac[3:6, 0:3] = MinvG
    Ac[3:6, 3:6] = MinvD
    Bc[3:6, :] = MinvE

    # Discretize using matrix exponential (exact)
    Z_top = np.hstack([Ac, Bc]) * dt
    Z_bot = np.zeros((1, n + 1))
    Z = np.vstack([Z_top, Z_bot])
    expZ = expm(Z)
    Ad = expZ[:n, :n]
    Bd = expZ[:n, n:n+1]

    Q = np.diag(Q_diag)
    R_cost = np.array([[R_val]])

    # System info
    eigs_open = np.linalg.eigvals(Ad)
    safe_print(f"[LQR] Open-loop |eigs|: "
               f"{sorted([f'{abs(e):.4f}' for e in eigs_open])}")
    safe_print(f"[LQR] R_rope={R_rope:.4f}m  Mt={Mt:.3f}kg  p1={p1:.4f}  p2={p2:.4f}")
    safe_print(f"[LQR] Mass matrix cond = {np.linalg.cond(M_eq):.1f}")

    # Controllability
    from numpy.linalg import matrix_rank
    C_mat = np.hstack([Bd] + [np.linalg.matrix_power(Ad, i) @ Bd for i in range(1, 6)])
    ctrl_rank = matrix_rank(C_mat)
    safe_print(f"[LQR] Controllability rank = {ctrl_rank}/6")

    try:
        P = solve_discrete_are(Ad, Bd, Q, R_cost)
        K = np.linalg.inv(R_cost + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        eigs_cl = np.linalg.eigvals(Ad - Bd @ K)
        safe_print(f"[LQR] K = [{', '.join(f'{k:.2f}' for k in K[0])}]")
        safe_print(f"[LQR] Closed-loop |eigs|: "
                   f"{sorted([f'{abs(e):.4f}' for e in eigs_cl])}")

        # Sanity: max tau for 10-deg tilt
        x_test = np.array([0, 0, np.radians(10), 0, 0, 0])
        tau_test = -float(K[0] @ x_test)
        safe_print(f"[LQR] tau for 10deg theta = {tau_test:.2f}Nm")

        # Save to file for easy debugging
        import os
        dbg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lqr_debug.txt')
        with open(dbg_path, 'w') as f:
            f.write(f"K = [{', '.join(f'{k:.4f}' for k in K[0])}]\n")
            f.write(f"Q = {Q_diag}\n")
            f.write(f"R = {R_val}\n")
            f.write(f"tau for 10deg theta = {tau_test:.3f} Nm\n")
            f.write(f"tau for 3deg theta  = {-float(K[0] @ np.array([0,0,np.radians(3),0,0,0])):.3f} Nm\n")
            f.write(f"motor_max = 3.0 Nm\n")
            f.write(f"Open-loop  |eigs| = {sorted([f'{abs(e):.4f}' for e in eigs_open])}\n")
            f.write(f"Closed-loop |eigs| = {sorted([f'{abs(e):.4f}' for e in eigs_cl])}\n")
            f.write(f"R_rope={R_rope:.4f} Mt={Mt:.3f} p1={p1:.4f} p2={p2:.4f}\n")
            f.write(f"M_eq cond = {np.linalg.cond(M_eq):.1f}\n")
            f.write(f"Ctrl rank = {ctrl_rank}/6\n")
        safe_print(f"[LQR] Debug saved to lqr_debug.txt")

        return K[0]
    except Exception as e:
        safe_print(f"[LQR] DARE failed: {e}")
        return None


# ================================================================
#  GUI
# ================================================================

class ParamLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Slackline V11 - Parameter Launcher")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("520x700")
        self.root.resizable(False, False)

        self.vars = {}
        self.models = scan_stl_models(_SRC_DIR)
        self.model_dims = None
        self.physics = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'),
                        foreground='#e0e0ff', background='#1a1a2e')
        style.configure('Section.TLabel', font=('Segoe UI', 10, 'bold'),
                        foreground='#7ecfff', background='#1a1a2e')
        style.configure('Param.TLabel', font=('Segoe UI', 9),
                        foreground='#c0c0d0', background='#1a1a2e')
        style.configure('Value.TLabel', font=('Consolas', 9),
                        foreground='#ffd700', background='#1a1a2e', width=8)
        style.configure('Dim.TLabel', font=('Consolas', 9),
                        foreground='#80ffaa', background='#1a1a2e')
        style.configure('TFrame', background='#1a1a2e')

        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=15, pady=10)

        ttk.Label(main, text="Slackline V11 - 3D Simulator (6-DOF LQR)",
                  style='Title.TLabel').pack(pady=(0, 10))

        # --- Model ---
        self._section(main, "STL Model")
        mf = ttk.Frame(main)
        mf.pack(fill='x', pady=2)
        ttk.Label(mf, text="Model:", style='Param.TLabel').pack(side='left')
        names = sorted(self.models.keys()) if self.models else ["(none)"]
        self.model_var = tk.StringVar(value=names[0])
        self.model_combo = ttk.Combobox(mf, textvariable=self.model_var,
                                         values=names, state='readonly', width=35)
        self.model_combo.pack(side='right', padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_changed)

        self.dims_label = ttk.Label(main, text="", style='Dim.TLabel', wraplength=480)
        self.dims_label.pack(pady=(2, 2), anchor='w')

        self.physics_label = ttk.Label(main, text="", style='Dim.TLabel', wraplength=480)
        self.physics_label.pack(pady=(0, 5), anchor='w')

        # --- Rope/Poles ---
        self._section(main, "Rope / Poles")
        self._slider(main, "L_pole", "Pole distance (m)", 0.3, 3.0, 0.80, res=0.05)
        self._slider(main, "L0", "Rope total length (m)", 0.4, 4.0, 1.00, res=0.05)
        self._slider(main, "pole_height", "Pole height (m)", 0.3, 3.0, 1.00, res=0.05)

        # --- Initial Angles ---
        self._section(main, "Initial Angles (deg)")
        self._slider(main, "alpha0_deg", "Lower body tilt (alpha)", -30, 30, 0, res=0.5)
        self._slider(main, "theta0_deg", "Upper body tilt (theta)", -30, 30, 3.0, res=0.5)

        # --- Damping ---
        self._section(main, "Damping coefficients")
        self._slider(main, "b_phi", "Rope-pole friction", 0.0, 0.5, 0.001, res=0.001)
        self._slider(main, "b_alpha", "Foot-rope friction", 0.0, 0.5, 0.001, res=0.001)
        self._slider(main, "b_theta", "Hip motor friction", 0.0, 0.5, 0.05, res=0.005)

        # --- Info ---
        self.info_label = ttk.Label(main, text="", style='Param.TLabel', wraplength=480)
        self.info_label.pack(pady=5)

        self.status_label = ttk.Label(main, text="Select model and click Launch.",
                                       style='Param.TLabel', wraplength=480)
        self.status_label.pack(pady=2)

        # --- Launch ---
        bf = ttk.Frame(main)
        bf.pack(fill='x', pady=10)
        self.launch_btn = tk.Button(bf, text="Launch Viewer",
                               font=('Segoe UI', 13, 'bold'),
                               bg='#06d6a0', fg='white', activebackground='#05b889',
                               activeforeground='white', relief='flat', cursor='hand2',
                               command=self._launch)
        self.launch_btn.pack(fill='x', ipady=8)

        self._on_model_changed()

    def _section(self, parent, text):
        ttk.Label(parent, text=text, style='Section.TLabel').pack(anchor='w', pady=(8, 2))
        ttk.Separator(parent, orient='horizontal').pack(fill='x')

    def _slider(self, parent, key, label, lo, hi, default, res=1.0):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=1)
        var = tk.DoubleVar(value=default)
        self.vars[key] = var
        ttk.Label(frame, text=label, style='Param.TLabel').pack(side='left')
        vl = ttk.Label(frame, text=f"{default:.2f}", style='Value.TLabel')
        vl.pack(side='right')
        tk.Scale(frame, from_=lo, to=hi, resolution=res, orient='horizontal',
                 variable=var, bg='#1a1a2e', fg='#c0c0d0', troughcolor='#2a2a4e',
                 highlightthickness=0, sliderrelief='flat', bd=0, length=200,
                 command=lambda v, vl=vl: (vl.configure(text=f"{float(v):.2f}"),
                                            self._update_info())).pack(side='right', padx=5)

    def _on_model_changed(self, event=None):
        prefix = self.model_var.get()
        if prefix not in self.models:
            self.dims_label.configure(text="No model")
            self.model_dims = None
            self.physics = None
            return

        paths = self.models[prefix]
        try:
            dims = get_model_dims(paths['upper'], paths['lower'])
            self.model_dims = dims
            self.dims_label.configure(
                text=f"  L1={dims['L1']*1000:.1f}mm  L2={dims['L2']*1000:.1f}mm  "
                     f"bd={dims['body_depth']*1000:.0f}mm")
        except Exception as e:
            self.dims_label.configure(text=f"STL error: {e}")
            self.model_dims = None

        self.physics = load_physics_json(prefix)
        if self.physics:
            u = self.physics['upper']
            l = self.physics['lower']
            self.physics_label.configure(
                text=f"  Physics: upper={u['mass_kg']}kg lower={l['mass_kg']}kg  "
                     f"motor={self.physics.get('motor',{}).get('max_torque_nm','?')}Nm")
        else:
            self.physics_label.configure(text="  Physics: no JSON (using defaults)")

        self._update_info()

    def _update_info(self, *args):
        try:
            L0 = self.vars['L0'].get()
            L_pole = self.vars['L_pole'].get()
            ph = self.vars['pole_height'].get()
            bd = self.model_dims['body_depth'] if self.model_dims else 0.060
            half_rope = L0 / 2.0
            horiz = L_pole / 2.0 - bd / 2.0
            if half_rope > horiz:
                sag = np.sqrt(half_rope**2 - horiz**2)
            else:
                sag = 0
            foot_h = ph - sag
            rope_len = np.sqrt(horiz**2 + sag**2)
            R_eff = rope_len
            if foot_h < 0:
                info = "WARNING: Rope too long"
            elif L0 <= L_pole:
                info = "WARNING: Rope shorter than pole distance"
            else:
                info = (f"Sag: {sag:.3f}m | Foot: {foot_h:.3f}m | "
                        f"R_rope: {R_eff:.3f}m")
            self.info_label.configure(text=info)
        except Exception:
            pass

    # ================================================================
    #  LAUNCH SIMULATION
    # ================================================================
    def _launch(self):
        import pybullet as p
        import pybullet_data
        import time

        prefix = self.model_var.get()
        if prefix not in self.models or self.model_dims is None:
            self.status_label.configure(text="No valid model!")
            return

        paths = self.models[prefix]
        dims = self.model_dims
        phys = self.physics

        upper_stl = copy_stl_to_temp(paths['upper'])
        lower_stl = copy_stl_to_temp(paths['lower'])

        # --- Physical parameters ---
        L1 = dims['L1']
        L2 = dims['L2']
        bd = dims['body_depth']
        lower_w = dims['lower_w']
        upper_w = dims['upper_w']

        if phys:
            m1 = phys['lower']['mass_kg']
            m2 = phys['upper']['mass_kg']
            I1 = phys['lower']['inertia_kg_m2']
            I2 = phys['upper']['inertia_kg_m2']
            com1_hip = phys['lower']['com_m']
            com2_hip = phys['upper']['com_m']
            motor_max_torque = phys.get('motor', {}).get('max_torque_nm', 3.0)
            motor_max_speed = phys.get('motor', {}).get('max_speed_rad_s', 5.7)
            lat_friction = phys['lower'].get('lateral_friction', 1.0)
        else:
            m1, m2 = 0.10, 0.32
            I1 = [0.0003, 0.0003, 0.00005]
            I2 = [0.0009, 0.0009, 0.0001]
            com1_hip = [0, 0, -0.050]
            com2_hip = [0, 0, 0.095]
            motor_max_torque = 3.0
            motor_max_speed = 5.7
            lat_friction = 2.0

        # CoM in link frame (ankle = origin for lower body)
        com1_link = [com1_hip[0], com1_hip[1], com1_hip[2] + L1]
        com2_link = com2_hip

        # LQR parameters
        lc1 = com1_link[2]   # CoM distance above ankle
        lc2 = com2_link[2]   # CoM distance above hip
        I1_y = I1[1]
        I2_y = I2[1]

        # Damping from launcher
        b_phi = self.vars['b_phi'].get()
        b_alpha = self.vars['b_alpha'].get()
        b_theta = self.vars['b_theta'].get()

        # --- Rope geometry ---
        L_pole = self.vars['L_pole'].get()
        L0 = self.vars['L0'].get()
        pole_height = self.vars['pole_height'].get()
        alpha0 = np.radians(self.vars['alpha0_deg'].get())
        theta0 = np.radians(self.vars['theta0_deg'].get())

        half_pole = L_pole / 2.0
        half_rope = L0 / 2.0
        if L0 <= L_pole:
            L0 = L_pole + 0.5
            half_rope = L0 / 2.0
        rope_horiz = half_pole - bd / 2.0
        rope_vert = np.sqrt(max(half_rope**2 - rope_horiz**2, 0.01))
        foot_z = pole_height - rope_vert
        if foot_z < 0.1:
            L0 = L_pole + 0.3
            half_rope = L0 / 2.0
            rope_vert = np.sqrt(max(half_rope**2 - rope_horiz**2, 0.01))
            foot_z = pole_height - rope_vert

        ra_end = np.array([0, -rope_horiz, -rope_vert])
        rb_end = np.array([0, -rope_horiz,  rope_vert])

        # R_rope: sag depth = radius of foot's circular arc in XZ plane
        # When rope_A rotates about Y, foot traces circle of radius = rope_vert
        R_rope = rope_vert

        rope_radius = 0.001
        rope_mass = 0.001  # nearly massless rope
        stl_scale = [0.001, 0.001, 0.001]

        # --- Print summary ---
        safe_print("\n" + "=" * 60)
        safe_print(f"  Model: {prefix}")
        safe_print(f"  Dims: L1={L1*1000:.1f}mm L2={L2*1000:.1f}mm bd={bd*1000:.0f}mm")
        safe_print(f"  Mass: m1={m1:.3f}kg m2={m2:.3f}kg total={m1+m2:.3f}kg")
        safe_print(f"  CoM: lc1={lc1*1000:.1f}mm lc2={lc2*1000:.1f}mm")
        safe_print(f"  Inertia(Iyy): I1={I1_y:.6f} I2={I2_y:.6f} kg*m^2")
        safe_print(f"  Motor: max {motor_max_torque}Nm / {motor_max_speed}rad/s")
        safe_print(f"  Rope: L_pole={L_pole:.2f}m L0={L0:.2f}m H={pole_height:.2f}m")
        safe_print(f"  R_rope={R_rope:.4f}m sag={rope_vert:.3f}m foot_z={foot_z:.3f}m")
        safe_print(f"  Damping: b_phi={b_phi:.2f} b_alpha={b_alpha:.2f} b_theta={b_theta:.2f}")
        safe_print(f"  Init: alpha={np.degrees(alpha0):.1f}deg theta={np.degrees(theta0):.1f}deg")
        safe_print("=" * 60)

        # --- Control frequency = physics frequency (240Hz) ---
        ctrl_dt = 1.0 / 240.0

        # --- Compute 6-DOF LQR ---
        K_lqr = compute_lqr_gain_6dof(
            m1=m1, m2=m2, R_rope=R_rope, L1=L1,
            lc1=lc1, lc2=lc2, I1_y=I1_y, I2_y=I2_y,
            b_phi=b_phi, b_alpha=b_alpha, b_theta=b_theta,
            Q_diag=[1, 50, 50, 0.1, 50, 50],  # High velocity penalty for damping
            R_val=0.1, dt=ctrl_dt)

        self.launch_btn.configure(state='disabled', text="Running...", bg='#555555')
        self.status_label.configure(text="Viewer open. Q=Quit")
        self.root.update()

        # === PyBullet Setup ===
        cid = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations=50)

        p.resetDebugVisualizerCamera(1.5, 50, -20, [0, 0, foot_z])
        p.loadURDF("plane.urdf")

        # Poles
        pv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=pole_height,
                                  rgbaColor=[0.5, 0.5, 0.5, 1])
        p.createMultiBody(0, baseVisualShapeIndex=pv,
                          basePosition=[0, half_pole, pole_height/2])
        p.createMultiBody(0, baseVisualShapeIndex=pv,
                          basePosition=[0, -half_pole, pole_height/2])
        cv = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[0.9,0.9,0.9,1])
        p.createMultiBody(0, baseVisualShapeIndex=cv,
                          basePosition=[0, half_pole, pole_height])
        pole_b_id = p.createMultiBody(0, baseVisualShapeIndex=cv,
                                       basePosition=[0, -half_pole, pole_height])

        # Shape helpers
        def capsule_orn(d):
            d = np.array(d, dtype=float); dl = np.linalg.norm(d)
            if dl < 1e-8: return [0,0,0,1]
            d = d/dl; z = np.array([0,0,1.0])
            dot = np.clip(np.dot(z,d),-1,1)
            if dot > 0.9999: return [0,0,0,1]
            if dot < -0.9999: return [1,0,0,0]
            ax = np.cross(z,d); ax = ax/np.linalg.norm(ax)
            ang = np.arccos(dot); s = np.sin(ang/2)
            return [ax[0]*s, ax[1]*s, ax[2]*s, np.cos(ang/2)]

        # Visual/collision shapes
        ra_mid = ra_end / 2.0
        rope_a_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=rope_radius,
            length=np.linalg.norm(ra_end), visualFramePosition=ra_mid.tolist(),
            visualFrameOrientation=capsule_orn(ra_end), rgbaColor=[1,0.8,0,1])

        l_bounds = dims['lower_bounds']
        u_bounds = dims['upper_bounds']
        lower_y_off = l_bounds[0][1]
        upper_y_off = u_bounds[0][1]

        lower_vis = p.createVisualShape(p.GEOM_MESH, fileName=lower_stl,
            meshScale=stl_scale, visualFramePosition=[0, lower_y_off, L1],
            rgbaColor=[0.6, 0.6, 0.65, 0.9])
        lower_col = p.createCollisionShape(p.GEOM_BOX,
            halfExtents=[lower_w/2, bd/2, L1/2],
            collisionFramePosition=[0, 0, L1/2])

        upper_vis = p.createVisualShape(p.GEOM_MESH, fileName=upper_stl,
            meshScale=stl_scale, visualFramePosition=[0, upper_y_off, 0],
            rgbaColor=[0.68, 0.85, 0.90, 0.9])
        upper_col = p.createCollisionShape(p.GEOM_BOX,
            halfExtents=[upper_w/2, bd/2, L2/2],
            collisionFramePosition=[0, 0, L2/2])

        rb_mid = rb_end / 2.0
        rope_b_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=rope_radius,
            length=np.linalg.norm(rb_end), visualFramePosition=rb_mid.tolist(),
            visualFrameOrientation=capsule_orn(rb_end), rgbaColor=[1,0.8,0,1])

        # === Robot multibody ===
        # Physical model:
        #   LOWER_BODY is a single rigid box. Ropes attach at its bottom edges.
        #   ROPE_A → bottom-front edge (Y=+bd/2) → "ANKLE" joint
        #   ROPE_B → bottom-back edge (Y=-bd/2) → offset [0,-bd,0] from ANKLE
        #   HIP → top of lower body [0,0,L1] from ANKLE
        #
        # Closed kinematic chain:
        #   Base(poleA) → ROPE_A → ANKLE(lower) → HIP(upper)
        #                                └→ ROPE_B → tip constrained to poleB

        robot_id = p.createMultiBody(
            baseMass=0,
            basePosition=[0, half_pole, pole_height],
            baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
            linkMasses=[rope_mass, m1, m2, rope_mass],
            linkCollisionShapeIndices=[-1, lower_col, upper_col, -1],
            linkVisualShapeIndices=[rope_a_vis, lower_vis, upper_vis, rope_b_vis],
            linkPositions=[[0,0,0], ra_end.tolist(), [0,0,L1], [0,-bd,0]],
            linkOrientations=[[0,0,0,1]]*4,
            linkInertialFramePositions=[ra_mid.tolist(), com1_link, com2_link,
                                        rb_mid.tolist()],
            linkInertialFrameOrientations=[[0,0,0,1]]*4,
            linkParentIndices=[0, 1, 2, 2],  # parent: 0=base, 2=ANKLE(link1)
            linkJointTypes=[p.JOINT_REVOLUTE]*4,  # All revolute Y
            linkJointAxis=[[0,1,0]]*4,
        )

        ROPE_A, ANKLE, HIP, ROPE_B = 0, 1, 2, 3

        # Apply physics
        p.changeDynamics(robot_id, ANKLE, mass=m1,
                         localInertiaDiagonal=I1,
                         jointDamping=b_alpha, lateralFriction=lat_friction)
        p.changeDynamics(robot_id, HIP, mass=m2,
                         localInertiaDiagonal=I2,
                         jointDamping=b_theta)
        p.changeDynamics(robot_id, ROPE_A, jointDamping=b_phi)
        p.changeDynamics(robot_id, ROPE_B, jointDamping=b_phi)

        # Disable default motors
        for j in range(4):
            p.setJointMotorControl2(robot_id, j, p.VELOCITY_CONTROL, force=0)

        # Disable self-collision and ground collision
        for i in range(-1, 4):
            for j in range(-1, 4):
                if i != j:
                    p.setCollisionFilterPair(robot_id, robot_id, i, j, enableCollision=0)
            p.setCollisionFilterPair(robot_id, 0, i, -1, enableCollision=0)

        # Closed-chain constraint: ROPE_B tip → pole_B top
        # This represents the REAL rope tension (not an artifact)
        rope_cid = p.createConstraint(
            robot_id, ROPE_B, pole_b_id, -1,
            p.JOINT_POINT2POINT, [0,0,0],
            rb_end.tolist(), [0,0,0])
        p.changeConstraint(rope_cid, maxForce=100.0)

        # === State extraction ===
        def get_state_6dof():
            """Get 6-state vector [phi, alpha_abs, theta_abs, phi_dot, alpha_dot_abs, theta_dot_abs]."""
            # phi from ROPE_A joint angle
            # All joints use same Y-axis rotation → same sign convention
            ra_st = p.getJointState(robot_id, ROPE_A)
            phi = ra_st[0]
            phi_dot = ra_st[1]

            # Absolute angles from world-frame link orientation
            # getLinkState with computeForwardKinematics=True, computeLinkVelocity=True
            lower_info = p.getLinkState(robot_id, ANKLE,
                                         computeForwardKinematics=True,
                                         computeLinkVelocity=True)
            upper_info = p.getLinkState(robot_id, HIP,
                                         computeForwardKinematics=True,
                                         computeLinkVelocity=True)

            # World orientation -> Euler (ZYX convention: roll, pitch, yaw)
            # For Y-axis rotation, pitch is what we want
            lower_euler = p.getEulerFromQuaternion(lower_info[5])  # worldLinkFrameOrientation
            upper_euler = p.getEulerFromQuaternion(upper_info[5])

            # Alpha and Theta from world orientation pitch
            alpha_abs = lower_euler[1]   # pitch = Y-axis rotation
            theta_abs = upper_euler[1]

            # World angular velocity (Y component)
            alpha_dot_abs = lower_info[7][1]   # worldLinkAngularVelocity[Y]
            theta_dot_abs = upper_info[7][1]

            return np.array([phi, alpha_abs, theta_abs,
                             phi_dot, alpha_dot_abs, theta_dot_abs])

        # === Initial pose ===
        hip_rel0 = theta0 - alpha0

        def apply_initial_pose():
            p.resetJointState(robot_id, ROPE_A, 0, 0)
            p.resetJointState(robot_id, ANKLE, alpha0, 0)
            p.resetJointState(robot_id, HIP, hip_rel0, 0)
            p.resetJointState(robot_id, ROPE_B, 0, 0)

        apply_initial_pose()

        # Print initial state
        x0 = get_state_6dof()
        safe_print(f"[Init] state = [{', '.join(f'{np.degrees(v):+.1f}deg' for v in x0[:3])}]")

        # === Viewer debug sliders ===
        # Q = state penalty, R = control cost
        q_phi_slider = p.addUserDebugParameter("Q_phi", 0.1, 100, 1)
        q_alpha_slider = p.addUserDebugParameter("Q_alpha", 1, 500, 50)
        q_theta_slider = p.addUserDebugParameter("Q_theta", 1, 500, 50)
        r_slider = p.addUserDebugParameter("R_cost", 0.001, 1.0, 0.01)

        # HUD
        p.addUserDebugText(f"Model: {prefix}  ({motor_max_torque}Nm)",
                            [0,0,pole_height+0.5], textColorRGB=[0.5,1,0.5], textSize=1.2)
        p.addUserDebugText("Space=Pause R=Reset C=Ctrl Q=Quit",
                            [0,0,pole_height+0.3], textColorRGB=[1,1,0.5], textSize=1.0)
        status_id = p.addUserDebugText("PAUSED", [0,0,foot_z-0.3],
                                        textColorRGB=[1,0.3,0.3], textSize=1.5)
        torque_id = p.addUserDebugText("", [0,0,foot_z-0.5],
                                        textColorRGB=[0.5,0.8,1], textSize=1.0)

        # === Simulation loop ===
        paused = True
        step_count = 0
        controller_on = False  # Start with controller OFF
        last_K = K_lqr
        last_Q = [1, 50, 50, 0.1, 5, 5]
        last_R = 0.01

        # CSV log file
        log_path = os.path.join(_SRC_DIR, 'sim_log.csv')
        log_file = open(log_path, 'w')
        log_file.write('t,phi,alpha,theta,phi_dot,alpha_dot,theta_dot,tau\n')
        safe_print(f'[LOG] Writing to {log_path}')

        def update_status(text, color):
            nonlocal status_id
            p.removeUserDebugItem(status_id)
            status_id = p.addUserDebugText(text, [0,0,foot_z-0.3],
                                            textColorRGB=color, textSize=1.5)

        def update_torque_display(tau, x):
            nonlocal torque_id
            p.removeUserDebugItem(torque_id)
            bar_n = int(abs(tau) / motor_max_torque * 20)
            bar = "|" * bar_n
            torque_id = p.addUserDebugText(
                f"tau={tau:+.2f}Nm {bar}  phi={np.degrees(x[0]):+.1f}",
                [0, 0, foot_z-0.5],
                textColorRGB=[0.5, 0.8, 1], textSize=1.0)

        safe_print("[Viewer] Space=Pause R=Reset C=Toggle_Ctrl Q=Quit")
        safe_print(f"[Viewer] Controller: OFF (press C to enable)")

        try:
            while p.isConnected():
                keys = p.getKeyboardEvents()

                # Space = toggle pause
                if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                    paused = not paused
                    if paused:
                        p.setJointMotorControl2(robot_id, HIP,
                                                p.VELOCITY_CONTROL, force=0)
                        x = get_state_6dof()
                        update_status("PAUSED", [1,0.3,0.3])
                        safe_print(f"[PAUSE] t={step_count/240:.2f}s "
                                   f"phi={np.degrees(x[0]):+.1f}deg "
                                   f"alpha={np.degrees(x[1]):+.1f}deg "
                                   f"theta={np.degrees(x[2]):+.1f}deg")
                    else:
                        ctrl_txt = "LQR" if controller_on and last_K is not None else "FREE"
                        update_status(f"RUNNING ({ctrl_txt})", [0.3,1,0.3])
                        safe_print(f"[RUN] Controller: {ctrl_txt}")

                # R = reset
                if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                    p.setJointMotorControl2(robot_id, HIP,
                                            p.VELOCITY_CONTROL, force=0)
                    apply_initial_pose()
                    step_count = 0
                    paused = True
                    update_status("RESET", [1,0.3,0.3])
                    safe_print("[RESET] Initial pose restored")

                # C = toggle controller
                if ord('c') in keys and keys[ord('c')] & p.KEY_WAS_TRIGGERED:
                    controller_on = not controller_on
                    safe_print(f"[CTRL] Controller: {'ON' if controller_on else 'OFF'}")
                    if not paused:
                        ctrl_txt = "LQR" if controller_on and last_K is not None else "FREE"
                        update_status(f"RUNNING ({ctrl_txt})", [0.3,1,0.3])
                    if not controller_on:
                        p.setJointMotorControl2(robot_id, HIP,
                                                p.VELOCITY_CONTROL, force=0)

                # Q = quit
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    break

                if not paused:
                    # Read LQR tuning sliders
                    q_phi_v = p.readUserDebugParameter(q_phi_slider)
                    q_alpha_v = p.readUserDebugParameter(q_alpha_slider)
                    q_theta_v = p.readUserDebugParameter(q_theta_slider)
                    r_v = p.readUserDebugParameter(r_slider)

                    # Recompute LQR if weights changed significantly
                    new_Q = [q_phi_v, q_alpha_v, q_theta_v, 0.1, 1, 1]
                    if (abs(new_Q[0]-last_Q[0]) > 0.5 or
                        abs(new_Q[1]-last_Q[1]) > 1 or
                        abs(new_Q[2]-last_Q[2]) > 1 or
                        abs(r_v-last_R) > 0.001):
                        last_Q = new_Q
                        last_R = r_v
                        new_K = compute_lqr_gain_6dof(
                            m1, m2, R_rope, L1, lc1, lc2, I1_y, I2_y,
                            b_phi=b_phi, b_alpha=b_alpha, b_theta=b_theta,
                            Q_diag=new_Q, R_val=r_v,
                            dt=ctrl_dt)
                        if new_K is not None:
                            last_K = new_K

                    # Read state
                    x = get_state_6dof()

                    # NaN check
                    if np.any(np.isnan(x)):
                        safe_print("[!] NaN detected - auto-pause")
                        paused = True
                        p.setJointMotorControl2(robot_id, HIP,
                                                p.VELOCITY_CONTROL, force=0)
                        update_status("NaN ERROR - press R", [1, 0, 0])
                        continue

                    # Apply control
                    tau = 0.0
                    if controller_on and last_K is not None:
                        if abs(x[1]) < np.radians(60) and abs(x[2]) < np.radians(60):
                            tau = -float(last_K @ x)
                            tau = np.clip(tau, -motor_max_torque, motor_max_torque)

                        p.setJointMotorControl2(
                            robot_id, HIP, p.TORQUE_CONTROL, force=tau)
                    else:
                        p.setJointMotorControl2(
                            robot_id, HIP, p.VELOCITY_CONTROL, force=0)

                    p.stepSimulation()
                    step_count += 1

                    # Log to CSV
                    if step_count < 240 * 3 or step_count % 10 == 0:
                        t = step_count / 240.0
                        log_file.write(f'{t:.4f},{x[0]:.6f},{x[1]:.6f},{x[2]:.6f},'
                                       f'{x[3]:.4f},{x[4]:.4f},{x[5]:.4f},{tau:.4f}\n')

                    # Periodic HUD update
                    if step_count % 120 == 0:
                        update_torque_display(tau, x)

                    # Periodic console output
                    if step_count % 480 == 0 and step_count >= 240 * 5:
                        t = step_count / 240.0
                        safe_print(
                            f"  t={t:.1f}s | phi={np.degrees(x[0]):+.1f} "
                            f"alpha={np.degrees(x[1]):+.1f} "
                            f"theta={np.degrees(x[2]):+.1f} "
                            f"tau={tau:+.2f}Nm")

                time.sleep(1.0 / 240.0)

        except Exception as e:
            safe_print(f"[Error] {e}")
            import traceback
            traceback.print_exc()
        finally:
            log_file.close()
            safe_print(f'[LOG] Saved to {log_path}')
            try:
                p.disconnect()
            except Exception:
                pass
            safe_print("[Viewer] Closed.\n")
            self.launch_btn.configure(state='normal', text="Launch Viewer",
                                       bg='#06d6a0')
            self.status_label.configure(text="Viewer closed. Launch again.")

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = ParamLauncher()
    app.run()

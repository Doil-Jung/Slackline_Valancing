"""
V14 MuJoCo Launcher — Parameter GUI
Adjust robot/structure params → Start simulation
Save/Load presets
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess, sys, os, json

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v14_mujoco.py')
PYTHON = sys.executable
PRESET_DIR = os.path.dirname(os.path.abspath(__file__))


class ParamSlider:
    """Slider + editable entry. Entry accepts ANY value (no clamping)."""
    def __init__(self, parent, row, label, from_, to_, default, key, fmt=".2f"):
        self.fmt = fmt
        self.key = key
        self._updating = False

        ttk.Label(parent, text=label, width=22, anchor='e').grid(
            row=row, column=0, padx=(10,4), pady=2, sticky='e')

        self.var = tk.DoubleVar(value=default)
        self.scale = ttk.Scale(parent, from_=from_, to=to_, variable=self.var,
                               orient='horizontal', length=220, command=self._from_slider)
        self.scale.grid(row=row, column=1, padx=4, pady=2)

        self.entry_var = tk.StringVar(value=f"{default:{fmt}}")
        self.entry = ttk.Entry(parent, textvariable=self.entry_var, width=10, justify='right')
        self.entry.grid(row=row, column=2, padx=(0,10), pady=2)
        self.entry.bind('<Return>', self._from_entry)
        self.entry.bind('<FocusOut>', self._from_entry)

    def _from_slider(self, *_):
        if self._updating: return
        self._updating = True
        self.entry_var.set(f"{self.var.get():{self.fmt}}")
        self._updating = False

    def _from_entry(self, *_):
        if self._updating: return
        self._updating = True
        try:
            val = float(self.entry_var.get())
            # Update slider to closest valid position (visual only)
            self.var.set(val)
            self.entry_var.set(f"{val:{self.fmt}}")
        except ValueError:
            self.entry_var.set(f"{self.var.get():{self.fmt}}")
        self._updating = False

    def get(self):
        try:
            return float(self.entry_var.get())
        except ValueError:
            return self.var.get()

    def set(self, val):
        self._updating = True
        self.var.set(val)
        self.entry_var.set(f"{val:{self.fmt}}")
        self._updating = False


class Launcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("V14 MuJoCo Slackline Launcher")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')

        self.proc = None
        self.sliders = {}  # key → ParamSlider
        self._build_ui()

    def _add_slider(self, parent, row, label, from_, to_, default, key, fmt=".2f"):
        s = ParamSlider(parent, row, label, from_, to_, default, key, fmt)
        self.sliders[key] = s
        return s

    def _build_ui(self):
        root = self.root
        r = 0

        # === Robot Body ===
        ttk.Label(root, text="── Robot Body ──", font=('', 10, 'bold')).grid(
            row=r, column=0, columnspan=3, pady=(10,4)); r+=1

        self._add_slider(root, r, "Lower mass M1 (kg):",   0.1, 80, 0.1,   'm1', ".2f"); r+=1
        self._add_slider(root, r, "Upper mass M2 (kg):",   0.1, 80, 0.25,  'm2', ".2f"); r+=1
        self._add_slider(root, r, "Head mass (kg):",        0,  30, 0.05,  'mh', ".2f"); r+=1
        self._add_slider(root, r, "Lower length L1 (m):",  0.05, 2.0, 0.2, 'l1', ".3f"); r+=1
        self._add_slider(root, r, "Upper length L2 (m):",  0.05, 2.0, 0.2, 'l2', ".3f"); r+=1
        self._add_slider(root, r, "Body width BD (m):",    0.05, 1.0, 0.1, 'bd', ".3f"); r+=1

        # === Structure ===
        ttk.Label(root, text="── Structure ──", font=('', 10, 'bold')).grid(
            row=r, column=0, columnspan=3, pady=(10,4)); r+=1

        self._add_slider(root, r, "Pole spacing L_pole (m):", 0.1, 8.0, 0.8, 'lpole', ".3f"); r+=1
        self._add_slider(root, r, "Pole height (m):",         0.3, 8.0, 1.0, 'poleh', ".3f"); r+=1
        self._add_slider(root, r, "Sag / R_target (m):",      0.05, 4.0, 0.3, 'rtgt', ".3f"); r+=1

        # === Initial Conditions ===
        ttk.Label(root, text="── Initial Conditions ──", font=('', 10, 'bold')).grid(
            row=r, column=0, columnspan=3, pady=(10,4)); r+=1

        self._add_slider(root, r, "Upper tilt θ₀ (deg):",  -30, 30, 8.6, 'theta0', ".1f"); r+=1
        self._add_slider(root, r, "Lower tilt α₀ (deg):",  -30, 30, 0.0, 'alpha0', ".1f"); r+=1

        # === Control ===
        ttk.Label(root, text="── Control ──", font=('', 10, 'bold')).grid(
            row=r, column=0, columnspan=3, pady=(10,4)); r+=1

        self._add_slider(root, r, "Hip damping B_th:",      0, 20,  0.01, 'btht', ".3f"); r+=1
        self._add_slider(root, r, "Max torque (Nm):",      0.1, 3000, 2.2, 'taumax', ".1f"); r+=1
        self._add_slider(root, r, "LQR R_cost:",        0.0001, 1.0, 0.001, 'rcost', ".4f"); r+=1

        # === LQR Q weights ===
        ttk.Label(root, text="── Q weights ──", font=('', 10, 'bold')).grid(
            row=r, column=0, columnspan=3, pady=(10,4)); r+=1

        self._add_slider(root, r, "Q0 phi (rope pos):",    0.01, 100, 1,    'q0', ".2f"); r+=1
        self._add_slider(root, r, "Q1 alpha (lower):",     0.1, 200, 50,   'q1', ".1f"); r+=1
        self._add_slider(root, r, "Q2 theta (upper):",     0.1, 200, 50,   'q2', ".1f"); r+=1
        self._add_slider(root, r, "Q3 phi_dot:",           0.01, 50, 0.1,  'q3', ".2f"); r+=1
        self._add_slider(root, r, "Q4 alpha_dot:",         0.1, 50, 5,     'q4', ".1f"); r+=1
        self._add_slider(root, r, "Q5 theta_dot:",         0.1, 50, 5,     'q5', ".1f"); r+=1

        # === Simulation Buttons ===
        btn_frame = ttk.Frame(root)
        btn_frame.grid(row=r, column=0, columnspan=3, pady=(12,4)); r+=1

        self.start_btn = ttk.Button(btn_frame, text="▶ Start Simulation", command=self._start)
        self.start_btn.pack(side='left', padx=8)

        self.stop_btn = ttk.Button(btn_frame, text="■ Stop", command=self._stop, state='disabled')
        self.stop_btn.pack(side='left', padx=8)

        # === Preset Buttons ===
        preset_frame = ttk.Frame(root)
        preset_frame.grid(row=r, column=0, columnspan=3, pady=(4,4)); r+=1

        ttk.Button(preset_frame, text="💾 Save Preset", command=self._save_preset).pack(side='left', padx=8)
        ttk.Button(preset_frame, text="📂 Load Preset", command=self._load_preset).pack(side='left', padx=8)
        ttk.Button(preset_frame, text="Reset Defaults", command=self._reset_defaults).pack(side='left', padx=8)

        # Status
        self.status = ttk.Label(root, text="Ready", foreground='gray')
        self.status.grid(row=r, column=0, columnspan=3, pady=(0,10))

    # ---------- Preset Save / Load ----------
    def _get_all_values(self):
        return {key: s.get() for key, s in self.sliders.items()}

    def _set_all_values(self, data):
        for key, val in data.items():
            if key in self.sliders:
                self.sliders[key].set(val)

    def _save_preset(self):
        path = filedialog.asksaveasfilename(
            initialdir=PRESET_DIR, defaultextension='.json',
            filetypes=[('JSON Preset', '*.json')],
            title='Save Preset')
        if not path: return
        try:
            with open(path, 'w') as f:
                json.dump(self._get_all_values(), f, indent=2)
            name = os.path.basename(path)
            self.status.config(text=f"Saved: {name}", foreground='blue')
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _load_preset(self):
        path = filedialog.askopenfilename(
            initialdir=PRESET_DIR, defaultextension='.json',
            filetypes=[('JSON Preset', '*.json')],
            title='Load Preset')
        if not path: return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._set_all_values(data)
            name = os.path.basename(path)
            self.status.config(text=f"Loaded: {name}", foreground='blue')
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # ---------- Simulation ----------
    def _build_cmd(self):
        import math
        args = [PYTHON, '-u', SCRIPT]
        v = self._get_all_values()
        env_params = {
            'V14_M1':     f'{v["m1"]:.4f}',
            'V14_M2':     f'{v["m2"]:.4f}',
            'V14_MHEAD':  f'{v["mh"]:.4f}',
            'V14_L1':     f'{v["l1"]:.4f}',
            'V14_L2':     f'{v["l2"]:.4f}',
            'V14_BD':     f'{v["bd"]:.4f}',
            'V14_LPOLE':  f'{v["lpole"]:.4f}',
            'V14_POLEH':  f'{v["poleh"]:.4f}',
            'V14_RTGT':   f'{v["rtgt"]:.4f}',
            'V14_THETA0': f'{math.radians(v["theta0"]):.6f}',
            'V14_ALPHA0': f'{math.radians(v["alpha0"]):.6f}',
            'V14_BTHT':   f'{v["btht"]:.4f}',
            'V14_TAUMAX': f'{v["taumax"]:.4f}',
            'V14_RCOST':  f'{v["rcost"]:.6f}',
            'V14_Q0':     f'{v["q0"]:.4f}',
            'V14_Q1':     f'{v["q1"]:.4f}',
            'V14_Q2':     f'{v["q2"]:.4f}',
            'V14_Q3':     f'{v["q3"]:.4f}',
            'V14_Q4':     f'{v["q4"]:.4f}',
            'V14_Q5':     f'{v["q5"]:.4f}',
        }
        return args, env_params

    def _start(self):
        if self.proc and self.proc.poll() is None:
            self._stop()

        args, env_params = self._build_cmd()
        env = os.environ.copy()
        env.update(env_params)

        try:
            self.proc = subprocess.Popen(args, env=env, cwd=os.path.dirname(SCRIPT))
            self.status.config(text="Simulation running...", foreground='green')
            self.start_btn.config(text="▶ Restart")
            self.stop_btn.config(state='normal')
            self.root.after(500, self._check_proc)
        except Exception as e:
            self.status.config(text=f"Error: {e}", foreground='red')

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        self.status.config(text="Stopped", foreground='gray')
        self.stop_btn.config(state='disabled')
        self.start_btn.config(text="▶ Start Simulation")

    def _check_proc(self):
        if self.proc and self.proc.poll() is not None:
            self.status.config(text="Simulation ended", foreground='gray')
            self.stop_btn.config(state='disabled')
            self.start_btn.config(text="▶ Start Simulation")
        elif self.proc:
            self.root.after(500, self._check_proc)

    def _reset_defaults(self):
        defaults = {
            'm1': 0.1, 'm2': 0.25, 'mh': 0.05,
            'l1': 0.2, 'l2': 0.2, 'bd': 0.1,
            'lpole': 0.8, 'poleh': 1.0, 'rtgt': 0.3,
            'theta0': 8.6, 'alpha0': 0.0,
            'btht': 0.01, 'taumax': 2.2, 'rcost': 0.001,
            'q0': 1, 'q1': 50, 'q2': 50,
            'q3': 0.1, 'q4': 5, 'q5': 5,
        }
        self._set_all_values(defaults)
        self.status.config(text="Defaults restored", foreground='gray')

    def run(self):
        self.root.mainloop()
        self._stop()


if __name__ == '__main__':
    Launcher().run()

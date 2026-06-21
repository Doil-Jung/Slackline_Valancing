"""
V11 Parameter Launcher - Tkinter GUI for adjusting simulation parameters

Sliders for: body sizes, rope length, pole height, initial angles, mode
Click 'Launch' -> rebuilds MJCF model -> opens MuJoCo viewer

Changes:
  - Auto-launches simulation on startup with initial angles applied
  - Launch button is always enabled (explicit state='normal')
  - Initial angles (alpha0, theta0) correctly set before first mj_step
"""
import sys
import os
import tkinter as tk
from tkinter import ttk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ParamLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Slackline V11 - Parameter Launcher")
        self.root.configure(bg='#1a1a2e')
        self.root.geometry("520x780")
        self.root.resizable(False, False)

        self.vars = {}
        self._viewer_thread = None
        self._viewer_running = False
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
        style.configure('Launch.TButton', font=('Segoe UI', 12, 'bold'),
                        foreground='white', background='#06d6a0')
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TScale', background='#1a1a2e')

        main = ttk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=15, pady=10)

        ttk.Label(main, text="Slackline V11 - 3D Simulator",
                  style='Title.TLabel').pack(pady=(0, 10))

        # --- Robot params ---
        self._section(main, "Robot")
        self._slider(main, "m1", "Lower body mass (kg)", 1, 80, 30)
        self._slider(main, "m2", "Upper body mass (kg)", 1, 80, 40)
        self._slider(main, "L1", "Lower body length (m)", 0.1, 2.0, 0.85, res=0.01)
        self._slider(main, "L2", "Upper body length (m)", 0.1, 2.0, 0.80, res=0.01)
        self._slider(main, "body_depth", "Foot gap / depth (m)", 0.05, 0.5, 0.25, res=0.01)

        # --- Rope params ---
        self._section(main, "Rope / Poles")
        self._slider(main, "L_pole", "Pole distance (m)", 1.0, 8.0, 4.0, res=0.1)
        self._slider(main, "L0", "Rope total length (m)", 1.5, 10.0, 5.0, res=0.1)
        self._slider(main, "pole_height", "Pole height (m)", 1.0, 5.0, 2.5, res=0.1)

        # --- Initial conditions ---
        self._section(main, "Initial Angles (deg)")
        self._slider(main, "alpha0_deg", "Lower body tilt", -30, 30, 0, res=0.5)
        self._slider(main, "theta0_deg", "Upper body tilt", -30, 30, 8.6, res=0.5)

        # --- Mode ---
        self._section(main, "Simulation Mode")
        mode_frame = ttk.Frame(main)
        mode_frame.pack(fill='x', pady=2)
        self.mode_var = tk.StringVar(value='A')
        ttk.Radiobutton(mode_frame, text="Mode A: 3D Free (ball joints)",
                        variable=self.mode_var, value='A').pack(anchor='w')
        ttk.Radiobutton(mode_frame, text="Mode B: Rolling only (y-hinge)",
                        variable=self.mode_var, value='B').pack(anchor='w')

        # --- Info ---
        self.info_label = ttk.Label(main, text="", style='Param.TLabel',
                                    wraplength=480)
        self.info_label.pack(pady=5)
        self._update_info()

        # --- Status label ---
        self.status_label = ttk.Label(main, text="", style='Param.TLabel',
                                       wraplength=480)
        self.status_label.pack(pady=2)

        # --- Launch button ---
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill='x', pady=10)
        self.launch_btn = tk.Button(btn_frame, text="▶  Launch Viewer",
                               font=('Segoe UI', 13, 'bold'),
                               bg='#06d6a0', fg='white', activebackground='#05b889',
                               activeforeground='white',
                               relief='flat', cursor='hand2',
                               state='normal',
                               command=self._launch)
        self.launch_btn.pack(fill='x', ipady=8)

    def _section(self, parent, text):
        ttk.Label(parent, text=text, style='Section.TLabel').pack(
            anchor='w', pady=(8, 2))
        sep = ttk.Separator(parent, orient='horizontal')
        sep.pack(fill='x')

    def _slider(self, parent, key, label, lo, hi, default, res=1.0):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=1)

        var = tk.DoubleVar(value=default)
        self.vars[key] = var

        ttk.Label(frame, text=label, style='Param.TLabel').pack(side='left')
        val_label = ttk.Label(frame, text=f"{default:.2f}", style='Value.TLabel')
        val_label.pack(side='right')

        slider = tk.Scale(frame, from_=lo, to=hi, resolution=res,
                          orient='horizontal', variable=var,
                          bg='#1a1a2e', fg='#c0c0d0',
                          troughcolor='#2a2a4e', highlightthickness=0,
                          sliderrelief='flat', bd=0, length=200,
                          command=lambda v, vl=val_label: (
                              vl.configure(text=f"{float(v):.2f}"),
                              self._update_info()))
        slider.pack(side='right', padx=5)

    def _update_info(self, *args):
        try:
            L0 = self.vars['L0'].get()
            L_pole = self.vars['L_pole'].get()
            bd = self.vars['body_depth'].get()
            ph = self.vars['pole_height'].get()

            half_rope = L0 / 2.0
            horiz = L_pole / 2.0 - bd / 2.0
            if half_rope > horiz:
                sag = np.sqrt(half_rope**2 - horiz**2)
            else:
                sag = 0
            foot_h = ph - sag

            if foot_h < 0:
                info = "WARNING: Rope too long! Foot goes below ground."
            elif L0 <= L_pole:
                info = f"WARNING: Rope shorter than pole distance (taut rope, no sag)"
            else:
                info = (f"Sag: {sag:.3f}m | Foot height: {foot_h:.3f}m | "
                        f"Rope each: {half_rope:.2f}m")
            self.info_label.configure(text=info)
        except Exception:
            pass

    def _build_config(self):
        """Build V11Config from current slider values"""
        from config import V11Config
        config = V11Config()

        # Apply slider values
        config.robot.m1 = self.vars['m1'].get()
        config.robot.m2 = self.vars['m2'].get()
        config.robot.L1 = self.vars['L1'].get()
        config.robot.L2 = self.vars['L2'].get()
        config.robot.body_depth = self.vars['body_depth'].get()
        config.robot.alpha0 = np.radians(self.vars['alpha0_deg'].get())
        config.robot.theta0 = np.radians(self.vars['theta0_deg'].get())

        config.rope.L_pole = self.vars['L_pole'].get()
        config.rope.L0 = self.vars['L0'].get()
        config.rope.pole_height = self.vars['pole_height'].get()
        config.rope._body_depth = config.robot.body_depth

        config.sim.mode = self.mode_var.get()

        # Validate
        if config.rope.L0 <= config.rope.L_pole:
            print("[WARN] Rope is taut (L0 <= L_pole), adjusting...")
            config.rope.L0 = config.rope.L_pole + 0.5

        if config.rope.foot_height < 0.1:
            print("[WARN] Foot height too low, adjusting rope length...")
            config.rope.L0 = config.rope.L_pole + 0.3

        return config

    def _launch(self):
        """Launch MuJoCo viewer with full control over initial angles.

        Uses launch_passive so we control data directly (launch() may create
        its own MjData, discarding our initial qpos settings).

        We implement play/pause and reset ourselves:
        - Simulation starts PAUSED so user can verify initial tilt
        - Right-click toggles Play/Pause
        - Backspace resets to initial angles
        """
        from sim.world import SlacklineWorld
        import mujoco
        import mujoco.viewer
        import time

        config = self._build_config()
        alpha0 = config.robot.alpha0
        hip_rel0 = config.robot.theta0 - config.robot.alpha0

        print("\n" + "=" * 50)
        print(f"  Launching with: Mode {config.sim.mode}")
        print(f"  Robot: m1={config.robot.m1:.1f}kg m2={config.robot.m2:.1f}kg "
              f"L1={config.robot.L1:.2f}m L2={config.robot.L2:.2f}m")
        print(f"  Rope: L_pole={config.rope.L_pole:.1f}m L0={config.rope.L0:.1f}m "
              f"H={config.rope.pole_height:.1f}m")
        print(f"  Sag={config.rope.sag:.3f}m FootH={config.rope.foot_height:.3f}m")
        print(f"  Init: alpha={np.degrees(config.robot.alpha0):.1f}deg "
              f"theta={np.degrees(config.robot.theta0):.1f}deg")
        print("=" * 50)

        # Disable button while viewer is open
        self.launch_btn.configure(state='disabled', text="⏳ Viewer Running...",
                                   bg='#555555')
        self.status_label.configure(
            text="MuJoCo viewer open. Close viewer window to return here.")
        self.root.update()

        # Build world
        world = SlacklineWorld(config)
        try:
            model, data = world.initialize()
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.launch_btn.configure(state='normal', text="▶  Launch Viewer",
                                       bg='#06d6a0')
            self.status_label.configure(text=f"ERROR: {e}")
            return

        ankle_qi = model.jnt_qposadr[world.ankle_jnt_id]
        hip_qi = model.jnt_qposadr[world.hip_jnt_id]

        # Force initial angles into data (in case anything reset them)
        data.qpos[ankle_qi] = alpha0
        data.qpos[hip_qi] = hip_rel0
        mujoco.mj_forward(model, data)

        print(f"[Init] ankle qpos = {np.degrees(data.qpos[ankle_qi]):.2f} deg")
        print(f"[Init] hip   qpos = {np.degrees(data.qpos[hip_qi]):.2f} deg")

        # State
        paused = True  # Start paused so user sees initial tilt
        sim_time_at_pause = 0.0

        # Key callback for viewer controls
        def key_callback(keycode):
            nonlocal paused, sim_time_at_pause
            # Space bar (32) = toggle pause
            if keycode == 32:
                paused = not paused
                if paused:
                    sim_time_at_pause = data.time
                    print(f"[Viewer] ⏸  PAUSED at t={data.time:.3f}s")
                else:
                    print(f"[Viewer] ▶  RUNNING")
            # Backspace (259) = reset to initial angles
            elif keycode == 259:
                mujoco.mj_resetData(model, data)
                data.qpos[ankle_qi] = alpha0
                data.qpos[hip_qi] = hip_rel0
                mujoco.mj_forward(model, data)
                paused = True
                print(f"[Viewer] ↺  RESET to initial angles "
                      f"(alpha={np.degrees(alpha0):.1f}°, "
                      f"theta={np.degrees(config.robot.theta0):.1f}°) - PAUSED")

        # Open viewer
        print("[Viewer] Opening... (Space=Play/Pause, Backspace=Reset)")
        print("[Viewer] ⏸  Starting PAUSED — press Space to run simulation")
        try:
            viewer = mujoco.viewer.launch_passive(
                model, data, key_callback=key_callback)

            # Re-apply initial angles after launch_passive
            # (launch_passive might reset data)
            data.qpos[ankle_qi] = alpha0
            data.qpos[hip_qi] = hip_rel0
            mujoco.mj_forward(model, data)
            viewer.sync()

            while viewer.is_running():
                if not paused:
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.002)

            viewer.close()
        except Exception as e:
            print(f"Viewer error: {e}")
            import traceback
            traceback.print_exc()

        print("[Viewer] Closed. Adjust params and launch again.\n")

        # Re-enable button
        self.launch_btn.configure(state='normal', text="▶  Launch Viewer",
                                   bg='#06d6a0')
        self.status_label.configure(text="Viewer closed. Adjust parameters and launch again.")

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = ParamLauncher()
    app.run()

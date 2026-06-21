"""
Slackline Balance Simulator V11 - Main Entry Point (MuJoCo)

Phase 1: Free fall test
  - MuJoCo viewer visualization
  - Perturbation (Rolling/Pitching/Yawing)
  - Real-time state monitoring
"""
import sys
import os
import time
import io
import numpy as np

# Windows cp949 encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import V11Config
from sim.world import SlacklineWorld


def main():
    print("=" * 60)
    print("  Slackline Balance Simulator V11 - 3D (MuJoCo)")
    print("=" * 60)

    # --- Config ---
    config = V11Config()

    if len(sys.argv) > 1 and sys.argv[1].upper() == 'B':
        config.sim.mode = 'B'
        print("  Mode B: Rigid trapezoid (rolling only)")
    else:
        config.sim.mode = 'A'
        print("  Mode A: Hinge rope (3D free)")

    print(f"  Robot: m1={config.robot.m1}kg, m2={config.robot.m2}kg, "
          f"L1={config.robot.L1}m, L2={config.robot.L2}m")
    print(f"  Rope:  L_pole={config.rope.L_pole}m, L0={config.rope.L0}m, "
          f"sag={config.rope.sag:.3f}m")
    print(f"  Init:  a0={np.degrees(config.robot.alpha0):.1f}deg, "
          f"th0={np.degrees(config.robot.theta0):.1f}deg")
    print("=" * 60)

    # --- World init ---
    world = SlacklineWorld(config)
    try:
        model, data = world.initialize()
    except Exception as e:
        print(f"\n[ERROR] World init failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- MuJoCo viewer ---
    viewer_ok = world.launch_viewer()
    if not viewer_ok:
        print("[Main] Running in console mode (no viewer)")

    # --- Simulation loop ---
    frame_count = 0
    print_interval = 240  # ~1sec at 240Hz
    perturb_applied = False

    print("\n[Main] Simulation started (Ctrl+C to stop)")
    print("[Main] Auto perturbation at t=2s\n")

    try:
        while True:
            # Auto perturbation at 2s
            if not perturb_applied and data.time > 2.0:
                theta0 = config.robot.theta0
                if abs(theta0) > 0.01:
                    print(f"[Perturb] th0={np.degrees(theta0):.1f}deg - observing")
                else:
                    world.apply_perturbation(0.5, axis='theta')
                    print("[Perturb] Rolling perturbation 0.5 rad/s applied")
                perturb_applied = True

            # Physics step
            world.step()
            frame_count += 1

            # Viewer sync
            if viewer_ok:
                if not world.sync_viewer():
                    print("[Main] Viewer closed")
                    break

            # State print
            if frame_count % print_interval == 0:
                state = world.get_state()
                phi_d = np.degrees(state['phi'])
                alp_d = np.degrees(state['alpha'])
                the_d = np.degrees(state['theta'])
                pit_d = np.degrees(state['pitch'])
                yaw_d = np.degrees(state['yaw'])
                com_e = (state['com_pos'][0] - state['foot_pos'][0]) * 100

                print(f"t={state['time']:6.2f}s | "
                      f"Roll phi={phi_d:+6.1f} a={alp_d:+6.1f} th={the_d:+6.1f} | "
                      f"Pitch={pit_d:+5.1f} Yaw={yaw_d:+5.1f} | "
                      f"CoM_err={com_e:+5.1f}cm | "
                      f"foot_z={state['foot_pos'][2]:.2f}m")

            # Real-time sync
            if viewer_ok:
                time.sleep(max(0, config.sim.dt - 0.0001))

    except KeyboardInterrupt:
        print("\n[Main] Stopped by user")
    finally:
        world.close()
        print("[Main] Done")


if __name__ == '__main__':
    main()

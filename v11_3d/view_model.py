"""
V11 Model Viewer - MJCF model inspection tool

Run: python view_model.py
  - Opens MuJoCo viewer with the slackline model
  - Physics paused initially -> inspect the model
  - Press SPACE in the viewer to start/pause physics
  - Mouse: rotate, scroll: zoom, Shift+drag: pan

Run: python view_model.py B
  - Mode B (y-axis hinge only)
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import V11Config
from sim.world import SlacklineWorld


def main():
    config = V11Config()

    if len(sys.argv) > 1 and sys.argv[1].upper() == 'B':
        config.sim.mode = 'B'
        mode_name = "B (y-axis hinge only)"
    else:
        config.sim.mode = 'A'
        mode_name = "A (3D ball joints)"

    # Small initial tilt so robot falls when physics starts
    # (if set to 0, robot stays in unstable equilibrium forever)
    config.robot.theta0 = 0.15   # ~8.6 deg upper body tilt
    config.robot.alpha0 = 0.0

    print("=" * 50)
    print("  V11 Model Viewer")
    print("=" * 50)
    print(f"  Mode: {mode_name}")
    print(f"  Rope: L_pole={config.rope.L_pole}m, L0={config.rope.L0}m")
    print(f"  Sag:  {config.rope.sag:.3f}m")
    print(f"  Foot height: {config.rope.foot_height:.3f}m")
    print(f"  Robot: m1={config.robot.m1}kg, m2={config.robot.m2}kg")
    print(f"  Robot: L1={config.robot.L1}m, L2={config.robot.L2}m")
    print("=" * 50)

    # Initialize world
    world = SlacklineWorld(config)
    try:
        model, data = world.initialize()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print joint info
    print("\n--- Joint Info ---")
    for i in range(model.njnt):
        name = model.joint(i).name
        jtype = model.jnt_type[i]
        type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
        print(f"  Joint {i}: '{name}' type={type_names.get(jtype, jtype)}")

    # Print body info
    print("\n--- Body Tree ---")
    for i in range(model.nbody):
        name = model.body(i).name
        parent = model.body_parentid[i]
        parent_name = model.body(parent).name if parent >= 0 else "world"
        pos = data.xpos[i]
        print(f"  Body {i}: '{name}' parent='{parent_name}' pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Print equality constraint info
    print(f"\n--- Equality Constraints: {model.neq} ---")

    # Launch viewer
    print("\n--- Opening Viewer ---")
    print("  Controls:")
    print("    Mouse drag: rotate camera")
    print("    Scroll: zoom")
    print("    Shift+drag: pan")
    print("    SPACE: start/pause physics")
    print("    Backspace: reset")
    print("    ESC or close window: exit")
    print()

    try:
        import mujoco.viewer
        # launch() blocks until viewer is closed (interactive mode)
        mujoco.viewer.launch(model, data)
    except Exception as e:
        print(f"Viewer error: {e}")
        print("\nFallback: rendering to image...")
        try:
            import mujoco
            renderer = mujoco.Renderer(model, 800, 600)
            renderer.update_scene(data)
            pixels = renderer.render()
            # Save as image
            from PIL import Image
            img = Image.fromarray(pixels)
            img_path = os.path.join(os.path.dirname(__file__), 'model_preview.png')
            img.save(img_path)
            print(f"  Saved to: {img_path}")
        except Exception as e2:
            print(f"  Image render also failed: {e2}")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
import mujoco
import numpy as np

xml = """<?xml version="1.0"?>
<mujoco model="test_servo">
  <option gravity="0 0 -9.81" timestep="0.001">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="2.0" armature="0.1"/>
  </default>
  <worldbody>
    <geom type="plane" size="6 6 0.01"/>
    <body name="base" pos="0 0 1.0">
      <joint name="ankle" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.1 0.1 0.4" pos="0 0 0.4" mass="30.0"/>
      <body name="upper" pos="0 0 0.8">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.1 0.1 0.4" pos="0 0 0.4" mass="45.0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <general name="hip_servo" joint="hip" gainprm="150000 0 0" biasprm="0 -150000 -3000" biastype="affine" forcerange="-100000 100000" forcelimited="true"/>
  </actuator>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Set initial hip angle to 5 degrees (0.087 rad)
hip_qadr = model.jnt_qposadr[model.joint('hip').id]
data.qpos[hip_qadr] = np.radians(5.0)
mujoco.mj_forward(model, data)

print(f"Initial: qpos={data.qpos[hip_qadr]:.4f} rad ({np.degrees(data.qpos[hip_qadr]):.2f} deg)")

for step in range(10):
    data.ctrl[0] = 0.0 # Try to control to 0.0
    mujoco.mj_step(model, data)
    qpos = data.qpos[hip_qadr]
    qvel = data.qvel[model.jnt_dofadr[model.joint('hip').id]]
    actuator_force = data.actuator_force[0]
    print(f"Step {step}: qpos={qpos:+.6f} rad, qvel={qvel:+.6f} rad/s, force={actuator_force:+.1f} Nm")

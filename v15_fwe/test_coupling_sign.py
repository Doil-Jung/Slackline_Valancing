# -*- coding: utf-8 -*-
import mujoco
import numpy as np

xml = """<?xml version="1.0"?>
<mujoco model="test_coupling">
  <option gravity="0 0 0" timestep="0.001">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 1.0">
      <joint name="ankle" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.1 0.1 0.4" pos="0 0 0.4" mass="30.0"/>
      <body name="upper" pos="0 0 0.8">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.1 0.1 0.4" pos="0 0 0.4" mass="45.0"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Set initial state to 0
mujoco.mj_forward(model, data)

# Let's apply a torque of +100.0 Nm to the hip joint
# and see the resulting accelerations qacc.
# Since there is no gravity, the accelerations are purely due to the torque.
data.qacc[:] = 0
data.qvel[:] = 0
data.qpos[:] = 0

# Apply torque to hip
hip_dof_id = model.jnt_dofadr[model.joint('hip').id]
ankle_dof_id = model.jnt_dofadr[model.joint('ankle').id]

data.qfrc_applied[hip_dof_id] = 100.0

# In MuJoCo, mj_forward computes qacc
mujoco.mj_forward(model, data)
print(f"Applied hip torque: +100.0 Nm")
print(f"Resulting ankle acceleration (alpha_double_dot): {data.qacc[ankle_dof_id]:+.4f} rad/s^2")
print(f"Resulting hip acceleration (delta_double_dot): {data.qacc[hip_dof_id]:+.4f} rad/s^2")

# Calculate beta double dot
M1, M2 = 30.0, 45.0
L1, L2 = 0.85, 0.85
LC1, LC2 = L1/2, L2/2
Mt = M1 + M2
h_com = (M1*LC1 + M2*(L1+LC2)) / Mt
p1 = M1*LC1 + M2*L1
p2 = M2*LC2
# beta = (p1*alpha + p2*theta) / (Mt*h_com)
# Since theta = alpha + delta:
# beta = (p1*alpha + p2*(alpha + delta)) / (Mt*h_com) = ((p1+p2)*alpha + p2*delta) / (Mt*h_com)
# d2_beta = ((p1+p2)*d2_alpha + p2*d2_delta) / (Mt*h_com)
d2_alpha = data.qacc[ankle_dof_id]
d2_delta = data.qacc[hip_dof_id]
d2_beta = ((p1+p2)*d2_alpha + p2*d2_delta) / (Mt*h_com)
print(f"Resulting beta acceleration (beta_double_dot): {d2_beta:+.4f} rad/s^2")

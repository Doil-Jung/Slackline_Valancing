# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

L_POLE = 3.0; POLE_H = 3.0
R_TARGET = 1.0    # rope length [m]

KP_SERVO = 150000.0
KD_SERVO = 3000.0
TAU_MAX = 100000.0
B_THETA = 0.5     # hip joint damping

ALPHA0 = np.radians(5.0)   # lower body tilt
THETA0 = np.radians(5.0)   # upper body tilt

HALF_POLE = L_POLE / 2.0
ROPE_HORIZ = HALF_POLE - BD / 2.0
ROPE_VERT  = R_TARGET
FOOT_Z = POLE_H - ROPE_VERT
ROPE_MASS = 0.5

LOWER_W = L2/4; UPPER_W = L2/4
POLE_RADIUS = 0.05
POLE_TOP_R  = 0.08
ROPE_RADIUS = 0.015
HIP_R       = 0.08
HIP_H       = 0.05

def generate_mjcf():
    rh = ROPE_HORIZ; rv = ROPE_VERT; hp = HALF_POLE
    lw2 = LOWER_W/2; uw2 = UPPER_W/2; bd2 = BD/2
    return f"""<?xml version="1.0"?>
<mujoco model="slackline_passive">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="2.0" armature="0.1"/>
  </default>
  <worldbody>
    <geom type="plane" size="6 6 0.01" rgba="0.3 0.3 0.35 1" contype="1" conaffinity="1"/>
    <geom name="pole_a_cyl" type="cylinder" pos="0 {hp} {POLE_H/2}" size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom name="pole_b_cyl" type="cylinder" pos="0 {-hp} {POLE_H/2}" size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <body name="rope_a_mount" pos="0 {hp} {POLE_H}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_a_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_a" type="capsule" size="{ROPE_RADIUS}" fromto="0 0 0 0 {-rh} {-rv}" rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="lower_body" pos="0 {-rh} {-rv}">
        <joint name="ankle" type="hinge" axis="0 1 0" damping="10.0"/>
        <geom name="lower_geom" type="box" size="{lw2} {bd2} {L1/2}" pos="0 {-bd2} {L1/2}" mass="{M1}" rgba="0.02 0.84 0.63 0.8"/>
        <body name="ankle_b_target" pos="0 {-BD} 0"/>
        <body name="upper_body" pos="0 {-bd2} {L1}">
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}"/>
          <geom name="upper_geom" type="box" size="{uw2} {bd2} {L2/2}" pos="0 0 {L2/2}" mass="{M2}" rgba="0.51 0.22 0.93 0.8"/>
        </body>
      </body>
    </body>
    <body name="rope_b_mount" pos="0 {-hp} {POLE_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_b_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_b" type="capsule" size="{ROPE_RADIUS}" fromto="0 0 0 0 {rh} {-rv}" rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="rope_b_end" pos="0 {rh} {-rv}"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0" solref="0.001 1" solimp="0.999 0.999 0.0001"/>
  </equality>
  <actuator>
    <general name="hip_servo" joint="hip" gainprm="{KP_SERVO} 0 0" biasprm="0 {-KP_SERVO} {-KD_SERVO}" forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/>
  </actuator>
</mujoco>"""

def get_state(model, data):
    phi = data.joint('phi_Y').qpos[0]
    lb_id = model.body('lower_body').id
    xmat = data.xmat[lb_id].reshape(3, 3)
    alpha = np.arctan2(-xmat[2, 0], xmat[2, 2])
    ub_id = model.body('upper_body').id
    xmat_u = data.xmat[ub_id].reshape(3, 3)
    theta = np.arctan2(-xmat_u[2, 0], xmat_u[2, 2])
    return phi, alpha, theta

def main():
    xml = generate_mjcf()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    hip_qadr = model.jnt_qposadr[model.joint('hip').id]
    ankle_qadr = model.jnt_qposadr[model.joint('ankle').id]
    
    data.qpos[ankle_qadr] = ALPHA0
    data.qpos[hip_qadr] = THETA0 - ALPHA0
    mujoco.mj_forward(model, data)
    
    SIM_TIME = 5.0
    N = int(SIM_TIME / DT)
    
    t_arr = np.zeros(N)
    phi_arr = np.zeros(N)
    alpha_arr = np.zeros(N)
    theta_arr = np.zeros(N)
    hip_arr = np.zeros(N)
    
    for i in range(N):
        phi, alpha, theta = get_state(model, data)
        data.ctrl[0] = 0.0  # hold hip = 0
        
        t_arr[i] = data.time
        phi_arr[i] = phi
        alpha_arr[i] = alpha
        theta_arr[i] = theta
        hip_arr[i] = data.joint('hip').qpos[0]
        
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)):
            print(f"NaN at t={data.time:.3f}")
            t_arr = t_arr[:i]
            phi_arr = phi_arr[:i]
            alpha_arr = alpha_arr[:i]
            theta_arr = theta_arr[:i]
            hip_arr = hip_arr[:i]
            break
            
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t_arr, np.degrees(phi_arr), label='phi')
    axes[0].set_ylabel('phi [deg]')
    axes[0].grid(True)
    axes[0].legend()
    
    axes[1].plot(t_arr, np.degrees(alpha_arr), label='alpha')
    axes[1].plot(t_arr, np.degrees(theta_arr), label='theta')
    axes[1].set_ylabel('body angles [deg]')
    axes[1].grid(True)
    axes[1].legend()
    
    axes[2].plot(t_arr, np.degrees(hip_arr), label='hip (delta)')
    axes[2].set_ylabel('hip [deg]')
    axes[2].grid(True)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('passive_stability.png')
    print("Done. Saved passive_stability.png")
    
    # Calculate beta
    p1_val = M1*(L1/2) + M2*L1
    p2_val = M2*(L2/2)
    Mt_val = M1 + M2
    h_com_val = (M1*(L1/2) + M2*(L1+L2/2)) / Mt_val
    beta_arr = (p1_val*alpha_arr + p2_val*theta_arr) / (Mt_val*h_com_val)
    
    print(f"Final t: {t_arr[-1]:.3f}s")
    print(f"Final phi: {np.degrees(phi_arr[-1]):.2f} deg, max |phi|: {np.degrees(np.max(np.abs(phi_arr))):.2f} deg")
    print(f"Final beta: {np.degrees(beta_arr[-1]):.2f} deg, max |beta|: {np.degrees(np.max(np.abs(beta_arr))):.2f} deg")
    print(f"Final hip: {np.degrees(hip_arr[-1]):.2f} deg, max |hip|: {np.degrees(np.max(np.abs(hip_arr))):.2f} deg")

if __name__ == '__main__':
    main()

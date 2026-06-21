"""Sign test: apply constant delta, observe beta direction"""
import numpy as np, mujoco
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

M1=0.1;M2=0.25;L1=0.2;L2=0.2;BD=0.10;GRAV=9.81;DT=0.001
POLE_H=1.0;L_POLE=0.8;R_TARGET=0.3;B_THETA=0.01
LC1=L1/2;LC2=L2/2;Mt=M1+M2
p1=M1*LC1+M2*L1;p2=M2*LC2;ps=p1+p2
h_com=(M1*LC1+M2*(L1+LC2))/Mt;c_foot=p2/ps
HP=L_POLE/2;RH=HP-BD/2;RV=R_TARGET
lw2=L2/8;uw2=L2/8;bd2=BD/2
PR=max(0.005,L2*0.05);RR=max(0.003,L2*0.015);RM=0.01
KP=500;KD=2;TAU_MAX=20.0

xml=f"""<?xml version="1.0"?>
<mujoco model="v15_sign">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>
  <default><geom contype="0" conaffinity="0"/><joint damping="0" armature="0.001"/></default>
  <worldbody>
    <geom type="plane" size="3 3 0.01" rgba="0.3 0.3 0.35 1" contype="1" conaffinity="1"/>
    <geom type="cylinder" pos="0 {HP} {POLE_H/2}" size="{PR} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom type="cylinder" pos="0 {-HP} {POLE_H/2}" size="{PR} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <body name="rope_a_mount" pos="0 {HP} {POLE_H}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_a_X" type="hinge" axis="1 0 0"/>
      <geom type="capsule" size="{RR}" fromto="0 0 0 0 {-RH} {-RV}" rgba="1 0.8 0 1" mass="{RM}"/>
      <body name="lower_body" pos="0 {-RH} {-RV}">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom type="box" size="{lw2} {bd2} {L1/2}" pos="0 {-bd2} {L1/2}" mass="{M1}" rgba="0.02 0.84 0.63 0.8"/>
        <body name="ankle_b_target" pos="0 {-BD} 0"/>
        <body name="upper_body" pos="0 {-bd2} {L1}">
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}"
                 range="-1.57 1.57" limited="true"/>
          <geom type="box" size="{uw2} {bd2} {L2/2}" pos="0 0 {L2/2}" mass="{M2}" rgba="0.51 0.22 0.93 0.8"/>
        </body>
      </body>
    </body>
    <body name="rope_b_mount" pos="0 {-HP} {POLE_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_b_X" type="hinge" axis="1 0 0"/>
      <geom type="capsule" size="{RR}" fromto="0 0 0 0 {RH} {-RV}" rgba="1 0.8 0 1" mass="{RM}"/>
      <body name="rope_b_end" pos="0 {RH} {-RV}"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0"
             solref="0.001 1" solimp="0.999 0.999 0.0001"/>
  </equality>
  <actuator>
    <general name="hip_servo" joint="hip"
             gainprm="{KP} 0 0" biasprm="0 {-KP} {-KD}"
             ctrlrange="-1.57 1.57" ctrllimited="true"
             forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/>
  </actuator>
</mujoco>"""

fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharex='col')

for col, delta_deg in enumerate([-30, 0, +30]):
    delta_rad = np.radians(delta_deg)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    hip_q = model.jnt_qposadr[model.joint('hip').id]
    ank_q = model.jnt_qposadr[model.joint('ankle').id]

    # Start with beta0=5d (uniform tilt)
    data.qpos[ank_q] = np.radians(5)
    data.qpos[hip_q] = 0
    mujoco.mj_forward(model, data)

    N = int(0.5/DT)
    ts=np.zeros(N);phis=np.zeros(N);betas=np.zeros(N);dact=np.zeros(N)

    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        lb=model.body('lower_body').id;xm=data.xmat[lb].reshape(3,3)
        alpha=np.arctan2(-xm[2,0],xm[2,2])
        ub=model.body('upper_body').id;xmu=data.xmat[ub].reshape(3,3)
        theta=np.arctan2(-xmu[2,0],xmu[2,2])
        beta=(p1*alpha+p2*theta)/(Mt*h_com)

        data.ctrl[0] = delta_rad
        ts[i]=data.time;phis[i]=phi;betas[i]=beta
        dact[i]=data.joint('hip').qpos[0]
        mujoco.mj_step(model,data)

    axes[0,col].plot(ts*1000,np.degrees(phis),'b-',lw=2)
    axes[0,col].axhline(0,color='red',ls='--')
    axes[0,col].set_ylabel('phi [d]')
    axes[0,col].set_title(f'delta_cmd = {delta_deg:+d}°')

    axes[1,col].plot(ts*1000,np.degrees(betas),'r-',lw=2)
    axes[1,col].axhline(0,color='black',lw=0.5)
    axes[1,col].axhline(5,color='gray',ls='--',alpha=0.3)
    axes[1,col].set_ylabel('beta [d]')

    axes[2,col].plot(ts*1000,np.degrees(dact),'g-',lw=2)
    axes[2,col].axhline(delta_deg,color='gray',ls='--')
    axes[2,col].set_ylabel('delta actual [d]')
    axes[2,col].set_xlabel('ms')

    print(f"delta={delta_deg:+d}d: beta@50ms={np.degrees(betas[50]):.1f}d  "
          f"beta@200ms={np.degrees(betas[200]):.1f}d  phi@200ms={np.degrees(phis[200]):.1f}d")

plt.suptitle('Sign test: beta0=+5d, constant delta command', fontsize=13)
plt.tight_layout()
import os
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v15_sign_test.png')
plt.savefig(out,dpi=150)
print(f"\nSaved: {out}")
print("\nExpected: one direction of delta should REDUCE beta initially")

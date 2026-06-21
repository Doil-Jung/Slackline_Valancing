"""V15 PD servo single-cycle test"""
import os, sys, numpy as np, mujoco
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fwe_controller import FWEController, Phase

M1=0.1;M2=0.25;L1=0.2;L2=0.2;BD=0.10;GRAV=9.81;DT=0.001
L_POLE=0.8;POLE_H=1.0;R_TARGET=0.3;B_THETA=0.01
LC1=L1/2;LC2=L2/2;Mt=M1+M2
p1=M1*LC1+M2*L1;p2=M2*LC2;ps=p1+p2
h_com=(M1*LC1+M2*(L1+LC2))/Mt;c_foot=p2/ps
HALF_POLE=L_POLE/2;ROPE_HORIZ=HALF_POLE-BD/2;ROPE_VERT=R_TARGET
ROPE_MASS=0.01;LOWER_W=L2/4;UPPER_W=L2/4
POLE_RADIUS=max(0.005,L2*0.05);ROPE_RADIUS=max(0.003,L2*0.015)
HIP_R=max(0.008,L2*0.08);HIP_H=max(0.005,BD*0.15)
rh=ROPE_HORIZ;rv=ROPE_VERT;hp=HALF_POLE
lw2=LOWER_W/2;uw2=UPPER_W/2;bd2=BD/2
KP=500;KD=10

xml = f"""<?xml version="1.0"?>
<mujoco model="v15">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0.001"/>
  </default>
  <worldbody>
    <geom type="plane" size="3 3 0.01" rgba="0.3 0.3 0.35 1" contype="1" conaffinity="1"/>
    <geom type="cylinder" pos="0 {hp} {POLE_H/2}" size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom type="cylinder" pos="0 {-hp} {POLE_H/2}" size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <body name="rope_a_mount" pos="0 {hp} {POLE_H}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_a_X" type="hinge" axis="1 0 0"/>
      <geom type="capsule" size="{ROPE_RADIUS}" fromto="0 0 0 0 {-rh} {-rv}" rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="lower_body" pos="0 {-rh} {-rv}">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom type="box" size="{lw2} {bd2} {L1/2}" pos="0 {-bd2} {L1/2}" mass="{M1}" rgba="0.02 0.84 0.63 0.8"/>
        <body name="ankle_b_target" pos="0 {-BD} 0"/>
        <body name="upper_body" pos="0 {-bd2} {L1}">
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}"/>
          <geom type="box" size="{uw2} {bd2} {L2/2}" pos="0 0 {L2/2}" mass="{M2}" rgba="0.51 0.22 0.93 0.8"/>
        </body>
      </body>
    </body>
    <body name="rope_b_mount" pos="0 {-hp} {POLE_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_b_X" type="hinge" axis="1 0 0"/>
      <geom type="capsule" size="{ROPE_RADIUS}" fromto="0 0 0 0 {rh} {-rv}" rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="rope_b_end" pos="0 {rh} {-rv}"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0"
             solref="0.001 1" solimp="0.999 0.999 0.0001"/>
  </equality>
  <actuator>
    <general name="hip_servo" joint="hip"
             gainprm="{KP} 0 0"
             biasprm="0 {-KP} {-KD}"
             ctrlrange="-3.14 3.14" ctrllimited="true"/>
  </actuator>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
hip_q = model.jnt_qposadr[model.joint('hip').id]
ank_q = model.jnt_qposadr[model.joint('ankle').id]

a0 = np.radians(5)
data.qpos[ank_q] = a0; data.qpos[hip_q] = 0
mujoco.mj_forward(model, data)

fwe = FWEController(eps=12, T_f=0.02, T_w1=0.26, h_com=h_com, c_foot=c_foot,
                    p1=p1, p2=p2, Mt=Mt, beta_threshold=np.radians(1.0),
                    max_delta=np.radians(60))

N = int(0.8/DT)
ts=[]; phis=[]; betas=[]; dcmd=[]; dact=[]

for i in range(N):
    phi = data.joint('phi_Y').qpos[0]
    lb = model.body('lower_body').id
    xm = data.xmat[lb].reshape(3,3); alpha = np.arctan2(-xm[2,0], xm[2,2])
    ub = model.body('upper_body').id
    xmu = data.xmat[ub].reshape(3,3); theta = np.arctan2(-xmu[2,0], xmu[2,2])
    beta = (p1*alpha + p2*theta) / (Mt*h_com)

    if fwe.cycle_count >= 1 and fwe.phase == Phase.IDLE:
        data.ctrl[0] = 0; d = 0
    else:
        d = fwe.step(DT, phi, alpha, theta)
        data.ctrl[0] = d

    ts.append(data.time); phis.append(phi); betas.append(beta)
    dcmd.append(d); dact.append(data.joint('hip').qpos[0])
    mujoco.mj_step(model, data)

ts=np.array(ts); phis=np.array(phis); betas=np.array(betas)
dcmd=np.array(dcmd); dact=np.array(dact)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
axes[0].plot(ts*1000, np.degrees(phis), 'b-', lw=2)
axes[0].axhline(0, color='red', ls='--'); axes[0].set_ylabel('phi [deg]')

axes[1].plot(ts*1000, np.degrees(betas), 'r-', lw=2)
axes[1].axhline(0, color='black', lw=0.5)
axes[1].axhline(5, color='gray', ls='--', alpha=0.3); axes[1].set_ylabel('beta [deg]')

axes[2].plot(ts*1000, np.degrees(dcmd), 'g-', lw=2, label='cmd')
axes[2].plot(ts*1000, np.degrees(dact), 'k--', lw=1, label='actual')
axes[2].set_ylabel('delta [deg]'); axes[2].set_xlabel('ms'); axes[2].legend()

axes[0].set_title(f'V15 PD-servo 1 cycle: kp={KP} kd={KD} eps=12 Tf=20ms Tw1=260ms')
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_pd_1cycle.png')
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
print(f"phi @ 300ms: {np.degrees(phis[min(300,N-1)]):.1f}d")
print(f"beta @ 400ms: {np.degrees(betas[min(400,N-1)]):.1f}d")
print(f"Final: phi={np.degrees(phis[-1]):.1f}d beta={np.degrees(betas[-1]):.1f}d")

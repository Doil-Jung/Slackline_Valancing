"""Detailed sign diagnosis — check alpha, theta, beta, phi conventions"""
import numpy as np, mujoco, os
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
KP=500;KD=2;TAU_MAX=50.0  # Very strong servo

xml=f"""<?xml version="1.0"?>
<mujoco model="v15_diag">
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

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
hip_q = model.jnt_qposadr[model.joint('hip').id]
ank_q = model.jnt_qposadr[model.joint('ankle').id]

# ---- Test 1: Set ankle=+5deg, observe all variables at t=0 ----
data.qpos[ank_q] = np.radians(5)
data.qpos[hip_q] = 0
mujoco.mj_forward(model, data)

phi0 = data.joint('phi_Y').qpos[0]
lb=model.body('lower_body').id; xm=data.xmat[lb].reshape(3,3)
alpha0 = np.arctan2(-xm[2,0], xm[2,2])
ub=model.body('upper_body').id; xmu=data.xmat[ub].reshape(3,3)
theta0 = np.arctan2(-xmu[2,0], xmu[2,2])

# Also read joint qpos directly
ankle_qpos = data.qpos[ank_q]
hip_qpos = data.qpos[hip_q]

print("="*60)
print("Initial state (ankle=+5deg, hip=0)")
print(f"  ankle qpos = {np.degrees(ankle_qpos):.2f}d")
print(f"  hip qpos   = {np.degrees(hip_qpos):.2f}d")
print(f"  phi_Y qpos = {np.degrees(phi0):.4f}d")
print(f"  alpha (xmat)= {np.degrees(alpha0):.2f}d")
print(f"  theta (xmat)= {np.degrees(theta0):.2f}d")
print(f"  beta = (p1*a + p2*t)/(Mt*h) = {np.degrees((p1*alpha0+p2*theta0)/(Mt*h_com)):.2f}d")
print()

# ---- Test 2: Large delta sweeps to see which direction REDUCES beta ----
deltas_deg = [-90, -60, -30, 0, +30, +60, +90]
N = int(0.3/DT)

fig, axes = plt.subplots(4, 1, figsize=(14, 16))

for dd in deltas_deg:
    dr = np.radians(dd)
    
    data2 = mujoco.MjData(model)
    data2.qpos[ank_q] = np.radians(5)
    data2.qpos[hip_q] = 0
    mujoco.mj_forward(model, data2)
    
    ts=[]; phis=[]; alphas=[]; thetas=[]; betas=[]; dacts=[]
    
    for i in range(N):
        phi=data2.joint('phi_Y').qpos[0]
        xm=data2.xmat[model.body('lower_body').id].reshape(3,3)
        a=np.arctan2(-xm[2,0],xm[2,2])
        xmu=data2.xmat[model.body('upper_body').id].reshape(3,3)
        t=np.arctan2(-xmu[2,0],xmu[2,2])
        beta=(p1*a+p2*t)/(Mt*h_com)
        
        data2.ctrl[0] = dr
        ts.append(data2.time);phis.append(phi);alphas.append(a);thetas.append(t)
        betas.append(beta);dacts.append(data2.joint('hip').qpos[0])
        mujoco.mj_step(model,data2)
    
    ts=np.array(ts)*1000;phis=np.degrees(phis);betas=np.degrees(betas)
    alphas=np.degrees(alphas);thetas=np.degrees(thetas);dacts=np.degrees(dacts)
    
    label = f'{dd:+d}d'
    axes[0].plot(ts, phis, lw=1.5, label=label)
    axes[1].plot(ts, betas, lw=1.5, label=label)
    axes[2].plot(ts, alphas, lw=1.5, label=label)
    axes[3].plot(ts, dacts, lw=1.5, label=label)
    
    print(f"delta={dd:+3d}d → beta@10ms={betas[10]:.1f}  beta@50ms={betas[50]:.1f}  "
          f"alpha@50ms={alphas[50]:.1f}  theta@50ms={thetas[50]:.1f}  "
          f"dact@50ms={dacts[50]:.1f}  phi@100ms={phis[100]:.1f}")

for ax, yl in zip(axes, ['phi [d]','beta [d]','alpha [d]','delta_act [d]']):
    ax.set_ylabel(yl); ax.legend(fontsize=8, ncol=4)
    ax.axhline(0, color='black', lw=0.5)

axes[0].set_title('Strong servo sweep: beta0=+5d, various delta_cmd')
axes[-1].set_xlabel('ms')
plt.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v15_sign_sweep.png')
plt.savefig(out,dpi=150)
print(f"\nSaved: {out}")

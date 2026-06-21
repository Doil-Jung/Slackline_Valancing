"""
Detailed diagnosis: K1=4.1, K2=0
Plot each cycle in detail — what exactly happens in cycle 2?
"""
import numpy as np, mujoco, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

M1=0.1;M2=0.25;L1=0.2;L2=0.2;BD=0.10;GRAV=9.81;DT=0.001
POLE_H=1.0;L_POLE=0.8;R_TARGET=0.3;B_THETA=0.01
Mt=M1+M2
HP=L_POLE/2;RH=HP-BD/2;RV=R_TARGET
lw2=L2/8;uw2=L2/8;bd2=BD/2
PR=max(0.005,L2*0.05);RR=max(0.003,L2*0.015);RM=0.01
KP=200;KD_s=10;TAU_MAX=10000

xml=f"""<?xml version="1.0"?>
<mujoco model="v15_fwe">
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
             gainprm="{KP} 0 0" biasprm="0 {-KP} {-KD_s}"
             ctrlrange="-1.57 1.57" ctrllimited="true"
             forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/>
  </actuator>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)

def compute_beta(m,d):
    foot=d.xpos[m.body('lower_body').id]
    cl=d.xipos[m.body('lower_body').id]
    cu=d.xipos[m.body('upper_body').id]
    com=(M1*cl+M2*cu)/Mt
    return np.arctan2(com[0]-foot[0],com[2]-foot[2])

# Run with detailed logging
K1 = 4.1; K2 = 0; T_w1 = 0.10  # best from fixed d=21 result
sim_time = 4.0

data = mujoco.MjData(model)
data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(5)
mujoco.mj_forward(model, data)

IDLE,FOLD,WAIT1,EXTEND,WAIT2 = 0,1,2,3,4
state_names = ['IDLE','FOLD','WAIT1','EXTEND','WAIT2']
state=IDLE;t_ph=0;target=0;fd=0;cyc=0
pp=0;bp=np.radians(5);ath=np.radians(2);md=np.radians(60)

N=int(sim_time/DT)
ts=[];betas=[];phis=[];dcmd=[];dact=[];sts=[];bdots=[];taus=[]

for i in range(N):
    phi=data.joint('phi_Y').qpos[0]
    beta=compute_beta(model,data)
    bd=(beta-bp)/DT
    ha=data.joint('hip').qpos[0]
    tau=data.actuator_force[0]
    t_ph+=DT

    if state==IDLE:
        target=0
        if abs(beta)>np.radians(0.5) and cyc<5:
            fd=np.clip(K1*beta+K2*bd,-md,md)
            target=fd;state=FOLD;t_ph=0;cyc+=1
            print(f"\n[Cycle {cyc}] === FOLD ===")
            print(f"  t={data.time*1000:.0f}ms  beta={np.degrees(beta):+.2f}°  "
                  f"beta_dot={np.degrees(bd):+.1f}°/s  phi={np.degrees(phi):+.1f}°")
            print(f"  delta_cmd = K1*beta = {K1:.1f}*{np.degrees(beta):.2f}° = {np.degrees(fd):.1f}°")
    elif state==FOLD:
        target=fd
        if abs(ha-fd)<ath or t_ph>0.2:
            print(f"  FOLD→WAIT1 @t={data.time*1000:.0f}ms  hip={np.degrees(ha):.1f}°  "
                  f"beta={np.degrees(beta):+.2f}°  t_fold={t_ph*1000:.0f}ms")
            state=WAIT1;t_ph=0
    elif state==WAIT1:
        target=fd
        if t_ph>=T_w1:
            print(f"  WAIT1→EXTEND @t={data.time*1000:.0f}ms  beta={np.degrees(beta):+.2f}°  "
                  f"phi={np.degrees(phi):+.1f}°  t_w1={t_ph*1000:.0f}ms")
            target=0;state=EXTEND;t_ph=0
    elif state==EXTEND:
        target=0
        if abs(ha)<ath or t_ph>0.2:
            print(f"  EXTEND→WAIT2 @t={data.time*1000:.0f}ms  beta={np.degrees(beta):+.2f}°  "
                  f"phi={np.degrees(phi):+.1f}°  hip={np.degrees(ha):.1f}°")
            state=WAIT2;t_ph=0;pp=phi
    elif state==WAIT2:
        target=0
        if pp*phi<0:
            print(f"  WAIT2→IDLE (phi=0) @t={data.time*1000:.0f}ms  beta={np.degrees(beta):+.2f}°  "
                  f"phi={np.degrees(phi):+.1f}°  t_w2={t_ph*1000:.0f}ms")
            state=IDLE;t_ph=0
        pp=phi
        if t_ph>3:
            print(f"  WAIT2 TIMEOUT @t={data.time*1000:.0f}ms  beta={np.degrees(beta):+.2f}°")
            state=IDLE;t_ph=0

    data.ctrl[0]=target
    ts.append(data.time*1000)
    betas.append(np.degrees(beta))
    phis.append(np.degrees(phi))
    dcmd.append(np.degrees(target))
    dact.append(np.degrees(ha))
    sts.append(state)
    bdots.append(np.degrees(bd))
    taus.append(tau)
    bp=beta
    mujoco.mj_step(model,data)
    if np.any(np.isnan(data.qpos)):
        print(f"  NaN @{data.time*1000:.0f}ms");break

# Plot
fig,axes=plt.subplots(5,1,figsize=(18,18),sharex=True)
ts=np.array(ts);betas=np.array(betas);phis=np.array(phis)
dcmd=np.array(dcmd);dact=np.array(dact);sts=np.array(sts)
bdots=np.array(bdots);taus=np.array(taus)

colors={0:'white',1:'#FFD700',2:'#90EE90',3:'#87CEEB',4:'#DDA0DD'}
clabels={0:'IDLE',1:'FOLD',2:'WAIT1',3:'EXTEND',4:'WAIT2'}
for r in range(5):
    prev=sts[0];start=ts[0]
    for j in range(1,len(sts)):
        if sts[j]!=prev or j==len(sts)-1:
            axes[r].axvspan(start,ts[j],alpha=0.15,color=colors.get(prev,'w'))
            if r==0 and prev!=0:
                axes[r].text((start+ts[j])/2,axes[r].get_ylim()[0] if r>0 else 0,
                            clabels.get(prev,''),ha='center',fontsize=7,alpha=0.6)
            prev=sts[j];start=ts[j]

axes[0].plot(ts,betas,'r-',lw=1.5,label='beta')
axes[0].axhline(0,color='black',lw=0.5)
axes[0].axhline(5,color='gray',ls=':',alpha=0.5,label='beta0=5°')
axes[0].axhline(-5,color='gray',ls=':',alpha=0.5)
axes[0].set_ylabel('beta [°]',fontsize=12)
axes[0].legend(fontsize=9)
axes[0].set_title(f'K1={K1} K2={K2} Tw1={T_w1*1000:.0f}ms — Detailed Cycle Analysis',fontsize=13)

axes[1].plot(ts,phis,'b-',lw=1.5,label='phi')
axes[1].axhline(0,color='red',ls='--',lw=0.8)
axes[1].set_ylabel('phi [°]',fontsize=12)
axes[1].legend(fontsize=9)

axes[2].plot(ts,bdots,'m-',lw=0.8,label='beta_dot')
axes[2].axhline(0,color='black',lw=0.5)
axes[2].set_ylabel('beta_dot [°/s]',fontsize=12)
axes[2].legend(fontsize=9)

axes[3].plot(ts,dcmd,'g-',lw=2,label='delta cmd')
axes[3].plot(ts,dact,'k--',lw=1,label='delta actual')
axes[3].axhline(0,color='black',lw=0.5)
axes[3].set_ylabel('delta [°]',fontsize=12)
axes[3].legend(fontsize=9)

axes[4].plot(ts,taus,'orange',lw=1,label='servo torque')
axes[4].axhline(0,color='black',lw=0.5)
axes[4].set_ylabel('torque [Nm]',fontsize=12)
axes[4].set_xlabel('time [ms]',fontsize=12)
axes[4].legend(fontsize=9)

# Add phase labels on beta plot
for r in range(5):
    prev_s=sts[0];start_t=ts[0]
    for j in range(1,len(sts)):
        if sts[j]!=prev_s or j==len(sts)-1:
            mid=(start_t+ts[j])/2
            if prev_s in clabels and r==0:
                ypos = max(betas) * 0.9
                axes[0].text(mid,ypos,clabels[prev_s],ha='center',fontsize=7,
                           alpha=0.7,rotation=90)
            prev_s=sts[j];start_t=ts[j]

plt.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v15_cycle_detail.png')
plt.savefig(out,dpi=150)
print(f"\nSaved: {out}")

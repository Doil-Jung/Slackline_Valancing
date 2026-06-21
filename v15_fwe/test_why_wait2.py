"""
WHY is WAIT2 = 751ms?
Zoom into cycle 1 only. Track phi position AND velocity.
R_TARGET=0.3 => T_phi = 2*pi*sqrt(0.3/9.81) = 1.1s, T/4 = 275ms
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

# ===== First: measure natural phi period with NO fold, beta=0 =====
print("=== Natural phi period test (no fold, small initial phi) ===")
data = mujoco.MjData(model)
data.qpos[model.jnt_qposadr[model.joint('phi_Y').id]] = np.radians(5)  # initial phi=5°
mujoco.mj_forward(model, data)
phi_log = []; t_log = []
for i in range(3000):  # 3 seconds
    phi_log.append(np.degrees(data.joint('phi_Y').qpos[0]))
    t_log.append(data.time*1000)
    data.ctrl[0] = 0
    mujoco.mj_step(model, data)

phi_arr = np.array(phi_log)
# Find zero crossings
signs = np.sign(phi_arr)
crosses = np.where(np.diff(signs))[0]
if len(crosses) >= 2:
    half_period = t_log[crosses[1]] - t_log[crosses[0]]
    full_period = half_period * 2
    print(f"  phi zero-crossings at: {[f'{t_log[c]:.0f}ms' for c in crosses[:6]]}")
    print(f"  Half period: {half_period:.0f}ms, Full period: {full_period:.0f}ms")
    print(f"  Quarter period: {full_period/4:.0f}ms")
else:
    print(f"  No zero crossings found! phi range: {phi_arr.min():.1f}° to {phi_arr.max():.1f}°")

# Theoretical
print(f"\n  Theoretical (simple pendulum R={R_TARGET}m):")
print(f"    T = 2π√(R/g) = 2π√({R_TARGET}/{GRAV}) = {2*np.pi*np.sqrt(R_TARGET/GRAV)*1000:.0f}ms")

# Effective pendulum length from measured period
if len(crosses) >= 2:
    T_meas = full_period / 1000  # seconds
    L_eff = GRAV * (T_meas / (2*np.pi))**2
    print(f"  Effective pendulum length from measurement: {L_eff:.3f}m")

# ===== Now: cycle 1 zoom =====
print("\n\n=== Cycle 1 detail (K1=4.1, K2=0, Tw1=100ms) ===")
K1=4.1; K2=0; T_w1=0.10
data = mujoco.MjData(model)
data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(5)
mujoco.mj_forward(model, data)

IDLE,FOLD,WAIT1,EXTEND,WAIT2 = 0,1,2,3,4
state=IDLE;t_ph=0;target=0;fd=0;cyc=0
pp=0;bp=np.radians(5);ath=np.radians(2);md=np.radians(60)

N=int(1.5/DT)  # Only 1.5s (cycle 1 + a bit)
ts=[];betas=[];phis=[];phi_dots=[];dcmd=[];dact=[];sts=[]

for i in range(N):
    phi=data.joint('phi_Y').qpos[0]
    phi_dot=data.joint('phi_Y').qvel[0]
    beta=compute_beta(model,data)
    bd=(beta-bp)/DT
    ha=data.joint('hip').qpos[0]
    t_ph+=DT

    if state==IDLE:
        target=0
        if abs(beta)>np.radians(0.5) and cyc<2:
            fd=np.clip(K1*beta+K2*bd,-md,md)
            target=fd;state=FOLD;t_ph=0;cyc+=1
    elif state==FOLD:
        target=fd
        if abs(ha-fd)<ath or t_ph>0.2:state=WAIT1;t_ph=0
    elif state==WAIT1:
        target=fd
        if t_ph>=T_w1:target=0;state=EXTEND;t_ph=0
    elif state==EXTEND:
        target=0
        if abs(ha)<ath or t_ph>0.2:state=WAIT2;t_ph=0;pp=phi
    elif state==WAIT2:
        target=0
        if pp*phi<0:state=IDLE;t_ph=0
        pp=phi
        if t_ph>3:state=IDLE;t_ph=0

    data.ctrl[0]=target
    ts.append(data.time*1000)
    betas.append(np.degrees(beta))
    phis.append(np.degrees(phi))
    phi_dots.append(np.degrees(phi_dot))
    dcmd.append(np.degrees(target))
    dact.append(np.degrees(ha))
    sts.append(state)
    bp=beta
    mujoco.mj_step(model,data)
    if np.any(np.isnan(data.qpos)):break

ts=np.array(ts);betas=np.array(betas);phis=np.array(phis)
phi_dots=np.array(phi_dots);dcmd=np.array(dcmd);dact=np.array(dact);sts=np.array(sts)

# Print key moments
print(f"  t=0ms: phi=0°, phi_dot=0°/s")
for t_check in [9, 50, 109, 172, 300, 500, 700, 923]:
    idx = min(t_check, len(ts)-1)
    s_name = ['IDLE','FOLD','WAIT1','EXTEND','WAIT2'][sts[idx]]
    print(f"  t={ts[idx]:.0f}ms: phi={phis[idx]:+.1f}°  phi_dot={phi_dots[idx]:+.1f}°/s  "
          f"beta={betas[idx]:+.1f}°  state={s_name}")

# Plot
fig,axes=plt.subplots(4,1,figsize=(16,14),sharex=True)
colors={0:'white',1:'#FFD700',2:'#90EE90',3:'#87CEEB',4:'#DDA0DD'}
clabels={0:'IDLE',1:'FOLD',2:'WAIT1',3:'EXTEND',4:'WAIT2'}
for r in range(4):
    prev=sts[0];start=ts[0]
    for j in range(1,len(sts)):
        if sts[j]!=prev or j==len(sts)-1:
            axes[r].axvspan(start,ts[j],alpha=0.2,color=colors.get(prev,'w'))
            if r==0:
                axes[r].text((start+ts[j])/2, 0, clabels.get(prev,''),
                           ha='center',va='bottom',fontsize=9,fontweight='bold',alpha=0.7)
            prev=sts[j];start=ts[j]

axes[0].plot(ts,betas,'r-',lw=2,label='beta')
axes[0].axhline(0,color='black',lw=1);axes[0].axhline(5,color='gray',ls=':',lw=1)
axes[0].set_ylabel('beta [°]',fontsize=13);axes[0].legend(fontsize=11)
axes[0].set_title(f'Cycle 1 Zoom — WHY is WAIT2 = 751ms?  (R={R_TARGET}m, T_phi/4≈275ms)',fontsize=14)

axes[1].plot(ts,phis,'b-',lw=2,label='phi (position)')
axes[1].axhline(0,color='red',ls='--',lw=1);axes[1].set_ylabel('phi [°]',fontsize=13)
axes[1].legend(fontsize=11)

axes[2].plot(ts,phi_dots,'m-',lw=1.5,label='phi_dot (velocity)')
axes[2].axhline(0,color='black',lw=0.5);axes[2].set_ylabel('phi_dot [°/s]',fontsize=13)
axes[2].legend(fontsize=11)

axes[3].plot(ts,dcmd,'g-',lw=2,label='delta cmd')
axes[3].plot(ts,dact,'k--',lw=1,label='delta actual')
axes[3].axhline(0,color='black',lw=0.5)
axes[3].set_ylabel('delta [°]',fontsize=13);axes[3].set_xlabel('time [ms]',fontsize=13)
axes[3].legend(fontsize=11)

# Annotate key moments
for ax in axes:
    for t_mark, label in [(9,'fold done'),(109,'extend start'),(172,'wait2 start')]:
        ax.axvline(t_mark, color='gray', ls=':', alpha=0.5)

plt.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v15_why_wait2.png')
plt.savefig(out,dpi=150)
print(f"\nSaved: {out}")

"""
Fine-tune: smaller deltas (5-15°) with T_w1 sweep.
From previous result, d=20° was too strong for beta0=5°.
"""
import numpy as np, mujoco, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

M1=0.1;M2=0.25;L1=0.2;L2=0.2;BD=0.10;GRAV=9.81;DT=0.001
POLE_H=1.0;L_POLE=0.8;R_TARGET=0.3;B_THETA=0.01
LC1=L1/2;LC2=L2/2;Mt=M1+M2
HP=L_POLE/2;RH=HP-BD/2;RV=R_TARGET
lw2=L2/8;uw2=L2/8;bd2=BD/2
PR=max(0.005,L2*0.05);RR=max(0.003,L2*0.015);RM=0.01
KP=200; KD=10; TAU_MAX=10000

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
             gainprm="{KP} 0 0" biasprm="0 {-KP} {-KD}"
             ctrlrange="-1.57 1.57" ctrllimited="true"
             forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/>
  </actuator>
</mujoco>"""

def compute_beta(model, data):
    foot = data.xpos[model.body('lower_body').id]
    com_l = data.xipos[model.body('lower_body').id]
    com_u = data.xipos[model.body('upper_body').id]
    com = (M1 * com_l + M2 * com_u) / Mt
    return np.arctan2(com[0]-foot[0], com[2]-foot[2])

def run_fwe(delta_deg, T_w1, beta0_deg=5, sim_time=5.0, max_cycles=10):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(beta0_deg)
    mujoco.mj_forward(model, data)

    IDLE,FOLD,WAIT1,EXTEND,WAIT2 = 0,1,2,3,4
    state=IDLE; t_ph=0; target=0; fold_sign=1; cycle=0
    phi_prev=0; arrive_th=np.radians(3)
    N=int(sim_time/DT)
    ts=np.zeros(N);phis=np.zeros(N);betas=np.zeros(N)
    dcmd=np.zeros(N);dact=np.zeros(N);sts=np.zeros(N,dtype=int)
    # Track beta at key moments
    beta_at_idle = []

    for i in range(N):
        phi=data.joint('phi_Y').qpos[0]
        beta=compute_beta(model,data)
        hip_act=data.joint('hip').qpos[0]
        t_ph+=DT

        if state==IDLE:
            target=0
            if abs(beta)>np.radians(0.5) and cycle<max_cycles:
                fold_sign=np.sign(beta)
                target=fold_sign*np.radians(delta_deg)
                state=FOLD;t_ph=0;cycle+=1
        elif state==FOLD:
            target=fold_sign*np.radians(delta_deg)
            if abs(hip_act-target)<arrive_th or t_ph>0.2:
                state=WAIT1;t_ph=0
        elif state==WAIT1:
            target=fold_sign*np.radians(delta_deg)
            if t_ph>=T_w1:
                target=0;state=EXTEND;t_ph=0
        elif state==EXTEND:
            target=0
            if abs(hip_act)<arrive_th or t_ph>0.2:
                state=WAIT2;t_ph=0;phi_prev=phi
        elif state==WAIT2:
            target=0
            if phi_prev*phi<0:
                beta_at_idle.append(np.degrees(beta))
                state=IDLE;t_ph=0
            phi_prev=phi
            if t_ph>5: state=IDLE;t_ph=0

        data.ctrl[0]=target
        ts[i]=data.time;phis[i]=phi;betas[i]=beta;dcmd[i]=target;dact[i]=hip_act;sts[i]=state
        mujoco.mj_step(model,data)
        if np.any(np.isnan(data.qpos)):break

    return ts[:i+1],phis[:i+1],betas[:i+1],dcmd[:i+1],dact[:i+1],sts[:i+1],beta_at_idle


# Fine sweep
configs = [
    (5, 0.10), (5, 0.20), (5, 0.30), (5, 0.50),
    (8, 0.10), (8, 0.20), (8, 0.30), (8, 0.50),
    (12, 0.10), (12, 0.20), (12, 0.30), (12, 0.50),
]

print("delta  T_w1   beta_at_idle (each cycle)")
print("-"*70)
for dd, tw1 in configs:
    ts,phis,betas,dcmd,dact,sts,bai = run_fwe(dd, tw1, 5, 8.0, 15)
    bai_str = [f"{b:+.1f}" for b in bai[:8]]
    # Check convergence: are absolute values decreasing?
    converging = "?"
    if len(bai) >= 3:
        abs_vals = [abs(b) for b in bai]
        if all(abs_vals[i+1] < abs_vals[i] for i in range(min(3,len(abs_vals)-1))):
            converging = "CONV!"
        elif max(abs_vals[:4]) > 90:
            converging = "DIV"
        else:
            converging = "osc"
    print(f"d={dd:2d}° Tw1={tw1*1000:3.0f}ms: [{', '.join(bai_str)}]  {converging}")


# Plot best candidates (small delta, various T_w1)
best_configs = [(5,0.20), (5,0.30), (8,0.20), (8,0.30), (12,0.20), (12,0.30)]
fig, axes = plt.subplots(3, len(best_configs), figsize=(5*len(best_configs), 10), sharex='col')

for col, (dd, tw1) in enumerate(best_configs):
    ts,phis,betas,dcmd,dact,sts,bai = run_fwe(dd, tw1, 5, 8.0, 15)

    colors={0:'white',1:'#FFD700',2:'#90EE90',3:'#87CEEB',4:'#DDA0DD'}
    for r in range(3):
        prev=sts[0];start=ts[0]
        for j in range(1,len(sts)):
            if sts[j]!=prev or j==len(sts)-1:
                axes[r,col].axvspan(start*1000,ts[j]*1000,alpha=0.12,color=colors.get(prev,'w'))
                prev=sts[j];start=ts[j]

    axes[0,col].plot(ts*1000,np.degrees(phis),'b-',lw=1)
    axes[0,col].axhline(0,color='red',ls='--',lw=0.8)
    axes[0,col].set_ylabel('phi[°]')
    axes[0,col].set_title(f'd={dd}° Tw1={tw1*1000:.0f}ms',fontsize=10)

    axes[1,col].plot(ts*1000,np.degrees(betas),'r-',lw=1)
    axes[1,col].axhline(0,color='black',lw=0.5)
    axes[1,col].set_ylabel('beta[°]')
    axes[1,col].set_ylim(-50, 50)

    axes[2,col].plot(ts*1000,np.degrees(dcmd),'g-',lw=1.5,label='cmd')
    axes[2,col].plot(ts*1000,np.degrees(dact),'k--',lw=0.8,label='act')
    axes[2,col].set_ylabel('delta[°]');axes[2,col].set_xlabel('ms')

    # Annotate beta at idle
    bai_str = ", ".join([f"{b:+.1f}" for b in bai[:5]])
    axes[1,col].set_title(f'd={dd}° Tw1={tw1*1000:.0f}ms\nbeta@idle: {bai_str}', fontsize=8)

plt.suptitle('Fine-tuned 4-Phase FWE: small delta, T_w1 sweep', fontsize=12)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_fine_4phase.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

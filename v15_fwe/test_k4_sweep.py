"""
K1=3.5~4.5 (step 0.1), K2=0~0.5 (step 0.05), Tw1 sweep
NO beta threshold, phi=0 → immediate FOLD
"""
import numpy as np, mujoco, os, time

M1=0.1;M2=0.25;L1=0.2;L2=0.2;BD=0.10;GRAV=9.81;DT=0.001
POLE_H=1.0;L_POLE=0.8;R_TARGET=0.3;B_THETA=0.01
Mt=M1+M2;HP=L_POLE/2;RH=HP-BD/2;RV=R_TARGET
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

def run_fwe(K1,K2,T_w1,beta0_deg=5,sim_time=8.0,max_cyc=20):
    data=mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]]=np.radians(beta0_deg)
    mujoco.mj_forward(model,data)
    FOLD,WAIT1,EW=0,1,2
    md=np.radians(60);ath=np.radians(2)
    beta=compute_beta(model,data);bp=beta;bd=0.0
    fd=np.clip(K1*beta+K2*bd,-md,md);target=fd
    state=FOLD;t_ph=0;cyc=1;pp=0;bai=[]
    N=int(sim_time/DT)
    for i in range(N):
        phi=data.joint('phi_Y').qpos[0]
        beta=compute_beta(model,data)
        bd=(beta-bp)/DT;ha=data.joint('hip').qpos[0];t_ph+=DT
        if state==FOLD:
            target=fd
            if abs(ha-fd)<ath or t_ph>0.2:state=WAIT1;t_ph=0
        elif state==WAIT1:
            target=fd
            if t_ph>=T_w1:target=0;state=EW;t_ph=0;pp=phi
        elif state==EW:
            target=0
            if t_ph>0.005 and pp*phi<0:
                bai.append(np.degrees(beta));cyc+=1
                if cyc>max_cyc:break
                fd=np.clip(K1*beta+K2*bd,-md,md)
                target=fd;state=FOLD;t_ph=0
            pp=phi
            if t_ph>3:bai.append(np.degrees(beta));break
        data.ctrl[0]=target;bp=beta
        mujoco.mj_step(model,data)
        if np.any(np.isnan(data.qpos)):break
    return bai

# Correct K2 range: 0~0.5
K1_vals = [round(3.5+i*0.1,1) for i in range(11)]  # 3.5~4.5
K2_vals = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]  # 10
Tw_vals = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]  # 8

total=len(K1_vals)*len(K2_vals)*len(Tw_vals)
print(f"K1: {K1_vals[0]}~{K1_vals[-1]} ({len(K1_vals)})")
print(f"K2: {K2_vals} ({len(K2_vals)})")
print(f"Tw1: {len(Tw_vals)} values")
print(f"Total: {total} cases")

t0=time.time()
results=[]
for K1 in K1_vals:
    for K2 in K2_vals:
        for tw in Tw_vals:
            bai=run_fwe(K1,K2,tw,5,8.0)
            mx=max(abs(b) for b in bai[:8]) if bai else 999
            results.append((mx,len(bai),K1,K2,tw,bai))
elapsed=time.time()-t0
print(f"Done: {elapsed:.1f}s ({elapsed/60:.1f}min)\n")

results.sort()
print(f"{'#':>3} {'K1':>5} {'K2':>5} {'Tw1':>6} {'nc':>3} {'max':>6}  betas")
print("-"*95)
for rank,(score,nc,k1,k2,tw,bai) in enumerate(results[:30]):
    bai_s=", ".join([f"{b:+.1f}" for b in bai[:8]])
    conv=""
    if nc>=4:
        ab=[abs(b) for b in bai]
        if all(a<10 for a in ab):conv="★★"
        elif all(a<15 for a in ab[:5]):conv="★"
        elif all(a<30 for a in ab[:4]):conv="~"
    print(f"{rank+1:3d} {k1:5.1f} {k2:5.2f} {tw*1000:5.0f}ms {nc:3d} {score:6.1f}  [{bai_s}]  {conv}")

print("\n=== |beta|<10° for 4+ cycles ===")
for s,nc,k1,k2,tw,bai in results:
    if nc>=4 and all(abs(b)<10 for b in bai[:min(5,nc)]):
        bai_s=", ".join([f"{b:+.1f}" for b in bai[:10]])
        print(f"  K1={k1} K2={k2} Tw1={tw*1000:.0f}ms: [{bai_s}]")

print("\n=== |beta|<20° for 4+ cycles ===")
ct=0
for s,nc,k1,k2,tw,bai in results:
    if nc>=4 and all(abs(b)<20 for b in bai[:min(5,nc)]):
        bai_s=", ".join([f"{b:+.1f}" for b in bai[:10]])
        print(f"  K1={k1} K2={k2} Tw1={tw*1000:.0f}ms: [{bai_s}]")
        ct+=1
        if ct>=15:break

print("\n=== |beta|<30° for 4+ cycles ===")
ct=0
for s,nc,k1,k2,tw,bai in results:
    if nc>=4 and all(abs(b)<30 for b in bai[:min(5,nc)]):
        bai_s=", ".join([f"{b:+.1f}" for b in bai[:10]])
        print(f"  K1={k1} K2={k2} Tw1={tw*1000:.0f}ms: [{bai_s}]")
        ct+=1
        if ct>=15:break

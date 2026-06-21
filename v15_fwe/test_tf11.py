# -*- coding: utf-8 -*-
"""#3 exact: K1=3.0, Tw1=10ms, Tp=10ms, T_FOLD=11ms"""
import numpy as np, mujoco
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

M1=0.1;M2=0.25;L1=0.2;L2=0.2;BD=0.10;GRAV=9.81;DT=0.001
POLE_H=1.0;L_POLE=0.8;R_TARGET=0.15;B_THETA=0.01
Mt=M1+M2;HP=L_POLE/2;RH=HP-BD/2;RV=R_TARGET
lw2=L2/8;uw2=L2/8;bd2=BD/2
PR=max(0.005,L2*0.05);RR=max(0.003,L2*0.015);RM=0.01
KP=200;KD_s=10;TAU_MAX=10000

K1=3.0; T_w1=0.010; T_pred=0.010

xml=f"""<?xml version="1.0"?>
<mujoco model="v15">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10"><flag contact="disable"/></option>
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
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}" range="-1.57 1.57" limited="true"/>
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
  <equality><connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0" solref="0.001 1" solimp="0.999 0.999 0.0001"/></equality>
  <actuator><general name="hip_servo" joint="hip" gainprm="{KP} 0 0" biasprm="0 {-KP} {-KD_s}" ctrlrange="-1.57 1.57" ctrllimited="true" forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/></actuator>
</mujoco>"""

model=mujoco.MjModel.from_xml_string(xml)

def cb(m,d):
    f=d.xpos[m.body('lower_body').id]
    cl=d.xipos[m.body('lower_body').id]
    cu=d.xipos[m.body('upper_body').id]
    com=(M1*cl+M2*cu)/Mt
    return np.arctan2(com[0]-f[0],com[2]-f[2])

print(f"Tw1={T_w1*1000:.0f}ms Tp={T_pred*1000:.0f}ms R={R_TARGET} T_FOLD=19ms")
print(f"{'K1':>5} {'nc':>3} {'max':>6} {'min':>6}  betas")
print("-"*80)

for k1_x100 in [355]:
    K1 = k1_x100 / 100.0
    T_FOLD = 0.019
    data=mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]]=np.radians(5)
    mujoco.mj_forward(model,data)
    
    FOLD,W1,EXT,EW=0,1,2,3; md=np.radians(60)
    beta=cb(model,data);bp=beta;bd=0;be=beta
    dt_=np.clip(K1*be,-md,md);ds=0;st=FOLD;tp=0;cyc=1
    pp=data.joint('phi_Y').qpos[0];tfe=0;bai=[]
    
    for i in range(int(120/DT)):
        phi=data.joint('phi_Y').qpos[0]
        beta=cb(model,data)
        bd=(beta-bp)/DT if i>0 else 0
        ha=data.joint('hip').qpos[0]
        tp+=DT; be=beta+bd*T_pred
        pc=False
        if st!=FOLD and tfe>0.005 and pp*phi<0: pc=True
        if st==FOLD:
            f=min(tp/T_FOLD,1);ctrl=ds+f*(dt_-ds)
            if tp>=T_FOLD: st=W1;tp=0;tfe=0
        elif st==W1:
            ctrl=dt_;tfe+=DT
            if pc: bai.append(np.degrees(be));cyc+=1;ds=ha;dt_=np.clip(K1*be,-md,md);st=FOLD;tp=0;tfe=0
            elif tp>=T_w1: ds=dt_;dt_=0;st=EXT;tp=0
        elif st==EXT:
            f=min(tp/T_FOLD,1);ctrl=ds+f*(dt_-ds);tfe+=DT
            if pc: bai.append(np.degrees(be));cyc+=1;ds=ha;dt_=np.clip(K1*be,-md,md);st=FOLD;tp=0;tfe=0
            elif tp>=T_FOLD: st=EW;tp=0
        elif st==EW:
            ctrl=0;tfe+=DT
            if pc: bai.append(np.degrees(be));cyc+=1;ds=ha;dt_=np.clip(K1*be,-md,md);st=FOLD;tp=0;tfe=0
            if tp>30: bai.append(np.degrees(be));break
        if cyc>100 or abs(np.degrees(beta))>170: break
        pp=phi;data.ctrl[0]=ctrl;bp=beta
        mujoco.mj_step(model,data)
        if np.any(np.isnan(data.qpos)): break
    
    ab=[abs(b) for b in bai]
    mx=max(ab[:min(15,len(ab))]) if ab else 999
    mn=min(ab[:min(15,len(ab))]) if ab else 999
    bs=", ".join([f"{b:+.1f}" for b in bai[:15]])
    print(f"{K1:5.1f} {len(bai):3d} {mx:6.1f} {mn:6.1f}  [{bs}]")

print("Done!")

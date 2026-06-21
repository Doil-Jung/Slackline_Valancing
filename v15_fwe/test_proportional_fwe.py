"""
Full 3-parameter sweep: delta = K1*beta + K2*beta_dot, wait T_w1
Benchmark: 0.09s/case. ~960 cases ≈ 1.5 min
"""
import numpy as np, mujoco, os, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

M1=0.1;M2=0.25;L1=0.2;L2=0.2;BD=0.10;GRAV=9.81;DT=0.001
POLE_H=1.0;L_POLE=0.8;R_TARGET=0.3;B_THETA=0.01
Mt=M1+M2
HP=L_POLE/2;RH=HP-BD/2;RV=R_TARGET
lw2=L2/8;uw2=L2/8;bd2=BD/2
PR=max(0.005,L2*0.05);RR=max(0.003,L2*0.015);RM=0.01
KP=200; KD_s=10; TAU_MAX=10000

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

def compute_beta(m, d):
    foot=d.xpos[m.body('lower_body').id]
    cl=d.xipos[m.body('lower_body').id]
    cu=d.xipos[m.body('upper_body').id]
    com=(M1*cl+M2*cu)/Mt
    return np.arctan2(com[0]-foot[0],com[2]-foot[2])

def run_fwe(K1, K2, T_w1, beta0_deg=5, sim_time=5.0, max_cyc=15):
    data=mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]]=np.radians(beta0_deg)
    mujoco.mj_forward(model,data)
    IDLE,FOLD,WAIT1,EXTEND,WAIT2=0,1,2,3,4
    state=IDLE;t_ph=0;target=0;fd=0;cyc=0
    pp=0;bp=np.radians(beta0_deg);ath=np.radians(2)
    md=np.radians(60);bai=[]
    N=int(sim_time/DT)
    for i in range(N):
        phi=data.joint('phi_Y').qpos[0]
        beta=compute_beta(model,data)
        bd=(beta-bp)/DT
        ha=data.joint('hip').qpos[0]
        t_ph+=DT
        if state==IDLE:
            target=0
            if abs(beta)>np.radians(0.5) and cyc<max_cyc:
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
            if pp*phi<0:bai.append(np.degrees(beta));state=IDLE;t_ph=0
            pp=phi
            if t_ph>3:state=IDLE;t_ph=0
        data.ctrl[0]=target;bp=beta
        mujoco.mj_step(model,data)
        if np.any(np.isnan(data.qpos)):break
    return bai

# === Sweep grid ===
K1_vals = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]          # 10
K2_vals = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]   # 8
Tw_vals = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 
           0.15, 0.20, 0.25, 0.30, 0.40, 0.50]         # 12

total = len(K1_vals)*len(K2_vals)*len(Tw_vals)
print(f"Grid: K1({len(K1_vals)}) x K2({len(K2_vals)}) x Tw1({len(Tw_vals)}) = {total} cases")
print(f"Estimated time: {total*0.09:.0f}s ({total*0.09/60:.1f}min)")
print()

t0=time.time()
results=[]
done=0

for K1 in K1_vals:
    for K2 in K2_vals:
        for tw in Tw_vals:
            bai=run_fwe(K1,K2,tw,5,5.0)
            mx=max(abs(b) for b in bai[:5]) if bai else 999
            n_cyc=len(bai)
            results.append((mx,n_cyc,K1,K2,tw,bai))
            done+=1

elapsed=time.time()-t0
print(f"Done: {done} cases in {elapsed:.1f}s ({elapsed/60:.1f}min)")

# === Results ===
results.sort()

print("\n" + "="*85)
print("=== TOP 25 (lowest max|beta| across first 5 cycles) ===")
print("="*85)
print(f"{'#':>3} {'K1':>5} {'K2':>5} {'Tw1':>6}  {'cycles':>6} {'max|β|':>7}  cycle betas")
print("-"*85)
for rank,(score,nc,k1,k2,tw,bai) in enumerate(results[:25]):
    bai_s=", ".join([f"{b:+.1f}" for b in bai[:6]])
    conv=""
    if nc>=3:
        ab=[abs(b) for b in bai]
        if ab[-1]<ab[0] and all(a<15 for a in ab): conv="★CONV"
        elif all(a<20 for a in ab[:4]): conv="~bound"
    print(f"{rank+1:3d} {k1:5.1f} {k2:5.2f} {tw*1000:5.0f}ms  {nc:6d}  {score:7.1f}°  [{bai_s}]  {conv}")

# === Convergence check ===
print("\n\n=== CONVERGENT CANDIDATES (|beta| decreasing over 3+ cycles, all <20°) ===")
conv_found=False
for score,nc,k1,k2,tw,bai in results:
    if nc>=3:
        ab=[abs(b) for b in bai]
        if all(a<20 for a in ab) and ab[-1]<ab[0]:
            bai_s=", ".join([f"{b:+.1f}" for b in bai[:8]])
            print(f"  K1={k1:.1f} K2={k2:.2f} Tw1={tw*1000:.0f}ms: [{bai_s}]")
            conv_found=True
if not conv_found:
    print("  (none found)")
    # Show best bounded
    print("\n=== BEST BOUNDED (all cycles <45°) ===")
    for score,nc,k1,k2,tw,bai in results:
        if nc>=2 and all(abs(b)<45 for b in bai[:4]):
            bai_s=", ".join([f"{b:+.1f}" for b in bai[:8]])
            print(f"  K1={k1:.1f} K2={k2:.2f} Tw1={tw*1000:.0f}ms: [{bai_s}]  max={score:.1f}°")

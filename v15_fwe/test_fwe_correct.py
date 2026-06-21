"""
V15 FWE with CORRECT beta (foot→CoM angle) + natural servo response
"""
import os, sys, numpy as np, mujoco
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

# Servo - strong enough to track well
KP=500;KD=2;TAU_MAX=50.0

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
    """beta = arctan2(dx, dz) of foot→CoM vector"""
    foot = data.xpos[model.body('lower_body').id]
    com_l = data.xipos[model.body('lower_body').id]
    com_u = data.xipos[model.body('upper_body').id]
    com = (M1 * com_l + M2 * com_u) / Mt
    dx = com[0] - foot[0]
    dz = com[2] - foot[2]
    return np.arctan2(dx, dz)


def run_fwe(eps, T_w1, beta0_deg=5, sim_time=3.0, max_cycles=10,
            max_delta_deg=80):
    """Run FWE with natural servo, correct beta."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    hip_q = model.jnt_qposadr[model.joint('hip').id]
    ank_q = model.jnt_qposadr[model.joint('ankle').id]
    data.qpos[ank_q] = np.radians(beta0_deg)
    mujoco.mj_forward(model, data)

    max_delta = np.radians(max_delta_deg)

    # State machine
    IDLE,FOLD,WAIT1,EXTEND,WAIT2 = 0,1,2,3,4
    state=IDLE; t_ph=0; target=0; phi_prev=0; d_fold=0; sign=1; cycle=0
    arrive_th = np.radians(3)
    log = []

    N = int(sim_time/DT)
    ts=np.zeros(N);phis=np.zeros(N);betas=np.zeros(N)
    dcmd=np.zeros(N);dact=np.zeros(N);sts=np.zeros(N,dtype=int);taus=np.zeros(N)

    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        beta = compute_beta(model, data)
        hip_act = data.joint('hip').qpos[0]
        t_ph += DT

        if state == IDLE:
            target = 0
            if abs(beta) > np.radians(2) and cycle < max_cycles:
                sign = np.sign(beta)  # fold same direction as tilt
                d_fold = h_com*(1+eps)*abs(beta)/c_foot
                d_fold = min(d_fold, max_delta)
                target = sign * d_fold
                state = FOLD; t_ph = 0; cycle += 1
                msg = f"[#{cycle}] FOLD: beta={np.degrees(beta):.1f}d target={np.degrees(target):.0f}d"
                log.append(msg); print(f"  {msg}")

        elif state == FOLD:
            # servo drives to target, wait until it arrives
            if abs(hip_act - target) < arrive_th or t_ph > 0.5:
                state = WAIT1; t_ph2 = t_ph; t_ph = 0
                msg = f"[#{cycle}] WAIT1 (T_fold={t_ph2*1000:.0f}ms, hip={np.degrees(hip_act):.0f}d)"
                log.append(msg); print(f"  {msg}")

        elif state == WAIT1:
            target = sign * d_fold  # hold
            if t_ph >= T_w1:
                target = 0; state = EXTEND; t_ph = 0
                msg = f"[#{cycle}] EXTEND"; log.append(msg); print(f"  {msg}")

        elif state == EXTEND:
            target = 0
            if abs(hip_act) < arrive_th or t_ph > 0.5:
                state = WAIT2; t_ph = 0; phi_prev = phi
                msg = f"[#{cycle}] WAIT2: phi={np.degrees(phi):.1f}d beta={np.degrees(beta):.1f}d"
                log.append(msg); print(f"  {msg}")

        elif state == WAIT2:
            target = 0
            if phi_prev * phi < 0:
                msg = f"[#{cycle}] IDLE: phi=0! beta={np.degrees(beta):.1f}d T_w2={t_ph*1000:.0f}ms"
                log.append(msg); print(f"  {msg}")
                state = IDLE; t_ph = 0
            phi_prev = phi
            if t_ph > 3.0:
                msg = f"[#{cycle}] IDLE: timeout phi={np.degrees(phi):.1f}d"
                log.append(msg); print(f"  {msg}")
                state = IDLE; t_ph = 0

        data.ctrl[0] = target
        ts[i]=data.time;phis[i]=phi;betas[i]=beta
        dcmd[i]=target;dact[i]=hip_act;sts[i]=state;taus[i]=data.actuator_force[0]
        mujoco.mj_step(model,data)

        if np.any(np.isnan(data.qpos)):
            print(f"  NaN at t={data.time:.4f}"); break

    return ts[:i+1],phis[:i+1],betas[:i+1],dcmd[:i+1],dact[:i+1],sts[:i+1],taus[:i+1],log


# Run with different eps
fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex='col')
for col, eps in enumerate([4, 8, 12]):
    print(f"\n{'='*50}")
    print(f"eps={eps}, T_w1=260ms, beta0=5d, max_delta=80d")
    ts,phis,betas,dcmd,dact,sts,taus,log = run_fwe(
        eps=eps, T_w1=0.26, beta0_deg=5, sim_time=3.0, max_cycles=5)

    # Phase shading
    colors={0:'white',1:'#FFD700',2:'#90EE90',3:'#87CEEB',4:'#DDA0DD'}
    for r in range(4):
        prev=sts[0];start=ts[0]
        for j in range(1,len(sts)):
            if sts[j]!=prev or j==len(sts)-1:
                axes[r,col].axvspan(start*1000,ts[j]*1000,alpha=0.12,
                                   color=colors.get(prev,'white'))
                prev=sts[j];start=ts[j]

    axes[0,col].plot(ts*1000,np.degrees(phis),'b-',lw=1.5)
    axes[0,col].axhline(0,color='red',ls='--'); axes[0,col].set_ylabel('phi [d]')
    axes[0,col].set_title(f'eps={eps}')

    axes[1,col].plot(ts*1000,np.degrees(betas),'r-',lw=1.5)
    axes[1,col].axhline(0,color='black',lw=0.5); axes[1,col].set_ylabel('beta [d]')

    axes[2,col].plot(ts*1000,np.degrees(dcmd),'g-',lw=2,label='cmd')
    axes[2,col].plot(ts*1000,np.degrees(dact),'k--',lw=1,label='act')
    axes[2,col].set_ylabel('delta [d]'); axes[2,col].legend(fontsize=8)

    axes[3,col].plot(ts*1000,taus,'m-',lw=1)
    axes[3,col].set_ylabel('tau [Nm]'); axes[3,col].set_xlabel('ms')

plt.suptitle(f'V15 FWE — Correct beta, natural servo (kp={KP} tau≤{TAU_MAX}Nm)', fontsize=13)
plt.tight_layout()
out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v15_fwe_correct.png')
plt.savefig(out,dpi=150)
print(f"\nSaved: {out}")

"""
CORRECT 4-Phase FWE with high-torque servo:
  Phase 1 (FOLD):   set delta → beta overshoots negative
  Phase 2 (WAIT1):  hold fold for T_w1, overshoot amplifies  
  Phase 3 (EXTEND): unfold (delta→0) — this is BEFORE phi=0
  Phase 4 (WAIT2):  wait until phi crosses 0 → new cycle

Sweep T_w1 to find the right timing.
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

KP=200; KD=10; TAU_MAX=10000  # high torque, speed-limited

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
    """
    Correct 4-phase FWE:
      FOLD:   cmd = sign(beta) * delta_deg
      WAIT1:  hold for T_w1 seconds
      EXTEND: cmd = 0
      WAIT2:  wait for phi zero-crossing → new cycle
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(beta0_deg)
    mujoco.mj_forward(model, data)

    IDLE, FOLD, WAIT1, EXTEND, WAIT2 = 0, 1, 2, 3, 4
    state = IDLE; t_ph = 0; target = 0; fold_sign = 1; cycle = 0
    phi_prev = 0; arrive_th = np.radians(3)
    
    N = int(sim_time / DT)
    ts=np.zeros(N); phis=np.zeros(N); betas=np.zeros(N)
    dcmd=np.zeros(N); dact=np.zeros(N); sts=np.zeros(N,dtype=int)

    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        beta = compute_beta(model, data)
        hip_act = data.joint('hip').qpos[0]
        t_ph += DT

        if state == IDLE:
            target = 0
            if abs(beta) > np.radians(1) and cycle < max_cycles:
                fold_sign = np.sign(beta)
                target = fold_sign * np.radians(delta_deg)
                state = FOLD; t_ph = 0; cycle += 1
                print(f"  [{cycle}] FOLD: beta={np.degrees(beta):.1f}°  "
                      f"cmd={np.degrees(target):.0f}°")

        elif state == FOLD:
            target = fold_sign * np.radians(delta_deg)
            # Transition to WAIT1 when servo arrives near target
            if abs(hip_act - target) < arrive_th or t_ph > 0.2:
                state = WAIT1; t_ph = 0
                print(f"  [{cycle}] WAIT1: hip={np.degrees(hip_act):.0f}°  "
                      f"beta={np.degrees(beta):.1f}°  t_fold={t_ph*1000:.0f}ms")

        elif state == WAIT1:
            target = fold_sign * np.radians(delta_deg)  # HOLD fold
            if t_ph >= T_w1:
                target = 0  # EXTEND
                state = EXTEND; t_ph = 0
                print(f"  [{cycle}] EXTEND: beta={np.degrees(beta):.1f}°  "
                      f"phi={np.degrees(phi):.1f}°")

        elif state == EXTEND:
            target = 0
            # Wait for servo to arrive near 0
            if abs(hip_act) < arrive_th or t_ph > 0.2:
                state = WAIT2; t_ph = 0; phi_prev = phi
                print(f"  [{cycle}] WAIT2: beta={np.degrees(beta):.1f}°  "
                      f"phi={np.degrees(phi):.1f}°")

        elif state == WAIT2:
            target = 0
            if phi_prev * phi < 0:  # phi zero-crossing
                print(f"  [{cycle}] phi=0! beta={np.degrees(beta):.1f}°  "
                      f"T_w2={t_ph*1000:.0f}ms")
                state = IDLE; t_ph = 0
            phi_prev = phi
            if t_ph > 5: state = IDLE; t_ph = 0

        data.ctrl[0] = target
        ts[i]=data.time; phis[i]=phi; betas[i]=beta
        dcmd[i]=target; dact[i]=hip_act; sts[i]=state
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)):
            print(f"  NaN at {data.time:.4f}s"); break

    return ts[:i+1], phis[:i+1], betas[:i+1], dcmd[:i+1], dact[:i+1], sts[:i+1]


# Sweep: delta x T_w1
configs = [
    (10, 0.05),  (10, 0.10),  (10, 0.20),
    (20, 0.05),  (20, 0.10),  (20, 0.20),
    (20, 0.30),  (20, 0.50),
]
ncol = len(configs)
fig, axes = plt.subplots(3, ncol, figsize=(4*ncol, 10), sharex='col')

for col, (dd, tw1) in enumerate(configs):
    print(f"\n{'='*50}")
    print(f"delta={dd}° T_w1={tw1*1000:.0f}ms")
    ts,phis,betas,dcmd,dact,sts = run_fwe(dd, tw1, 5, 5.0)

    colors = {0:'white', 1:'#FFD700', 2:'#90EE90', 3:'#87CEEB', 4:'#DDA0DD'}
    for r in range(3):
        prev=sts[0]; start=ts[0]
        for j in range(1, len(sts)):
            if sts[j] != prev or j == len(sts)-1:
                axes[r,col].axvspan(start*1000, ts[j]*1000, alpha=0.12,
                                     color=colors.get(prev,'w'))
                prev=sts[j]; start=ts[j]

    axes[0,col].plot(ts*1000, np.degrees(phis), 'b-', lw=1.2)
    axes[0,col].axhline(0, color='red', ls='--', lw=0.8)
    axes[0,col].set_ylabel('phi[°]')
    axes[0,col].set_title(f'd={dd}° Tw1={tw1*1000:.0f}ms', fontsize=9)

    axes[1,col].plot(ts*1000, np.degrees(betas), 'r-', lw=1.2)
    axes[1,col].axhline(0, color='black', lw=0.5)
    axes[1,col].set_ylabel('beta[°]')

    axes[2,col].plot(ts*1000, np.degrees(dcmd), 'g-', lw=1.5, label='cmd')
    axes[2,col].plot(ts*1000, np.degrees(dact), 'k--', lw=0.8, label='act')
    axes[2,col].set_ylabel('delta[°]')
    axes[2,col].set_xlabel('ms')
    if col == 0: axes[2,col].legend(fontsize=7)

plt.suptitle('Correct 4-Phase FWE (high-TAU servo): FOLD→WAIT1→EXTEND→WAIT2(phi=0)', fontsize=12)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_correct_4phase.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

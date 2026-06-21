"""
Fix: very high TAU_MAX so servo actually tracks different deltas.
KD provides velocity damping (speed limit).
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

# Servo: high torque, KD for speed limit
# STS3215: ~60RPM=360deg/s=6.3rad/s
# tau = KP*(target-actual) - KD*vel, unlimited torque
KP = 200
KD = 10    # gives vel_max ~ KP*error/KD ~ 200*1.5/10 = 30 rad/s (fast but finite)
TAU_MAX = 10000  # effectively unlimited

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

model = mujoco.MjModel.from_xml_string(xml)

# =========================================
# Test 1: FOLD AND HOLD — different deltas should now differ!
# =========================================
fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True)
sim_time = 2.0

for dd in [5, 10, 20, 30, 45]:
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(5)
    mujoco.mj_forward(model, data)

    N = int(sim_time / DT)
    ts=[]; betas=[]; phis=[]; dacts=[]; taus_log=[]

    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        beta = compute_beta(model, data)
        hip_act = data.joint('hip').qpos[0]
        tau = data.actuator_force[0]
        data.ctrl[0] = np.radians(dd)
        
        ts.append(data.time * 1000)
        betas.append(np.degrees(beta))
        phis.append(np.degrees(phi))
        dacts.append(np.degrees(hip_act))
        taus_log.append(tau)
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)): break

    label = f'd={dd}°'
    axes[0].plot(ts, betas, lw=1.5, label=label)
    axes[1].plot(ts, phis, lw=1.5, label=label)
    axes[2].plot(ts, dacts, lw=1.5, label=label)
    axes[3].plot(ts, taus_log, lw=1, label=label)

    betas_arr = np.array(betas)
    imin = np.argmin(betas_arr)
    # Find zero crossings
    signs = np.sign(betas_arr)
    crosses = np.where(np.diff(signs))[0]
    cross_t = [f"{ts[c]:.0f}" for c in crosses[:6]]
    
    print(f"d={dd:2d}°: beta_min={betas_arr[imin]:.2f}° @{ts[imin]:.0f}ms  "
          f"hip@50ms={dacts[min(50,len(dacts)-1)]:.1f}°  "
          f"beta@200ms={betas_arr[min(200,len(betas_arr)-1)]:.2f}°  "
          f"zero-cross(ms): {cross_t}")

# No fold reference
data = mujoco.MjData(model)
data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(5)
mujoco.mj_forward(model, data)
ts0=[]; b0=[]; p0=[]
for i in range(int(sim_time/DT)):
    phi=data.joint('phi_Y').qpos[0]
    beta=compute_beta(model,data)
    data.ctrl[0]=0
    ts0.append(data.time*1000);b0.append(np.degrees(beta));p0.append(np.degrees(phi))
    mujoco.mj_step(model,data)
    if np.any(np.isnan(data.qpos)):break
axes[0].plot(ts0, b0, 'k--', lw=1, alpha=0.5, label='no fold')
axes[1].plot(ts0, p0, 'k--', lw=1, alpha=0.5, label='no fold')

axes[0].set_ylabel('beta [°]', fontsize=12)
axes[0].axhline(0, color='red', lw=1)
axes[0].legend(fontsize=9); axes[0].set_ylim(-30, 50)
axes[0].set_title(f'FOLD AND HOLD (KP={KP} KD={KD} TAU≤{TAU_MAX}): do curves separate?', fontsize=13)

axes[1].set_ylabel('phi [°]', fontsize=12)
axes[1].axhline(0, color='red', lw=1); axes[1].legend(fontsize=9)

axes[2].set_ylabel('hip actual [°]', fontsize=12)
axes[2].legend(fontsize=9)

axes[3].set_ylabel('torque [Nm]', fontsize=12)
axes[3].set_xlabel('time [ms]', fontsize=12)
axes[3].legend(fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_high_tau.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

"""
Microsecond-level tracking of beta during FOLD phase.
Question: does beta ever go negative (overshoot) during fold?
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
h_com=(M1*LC1+M2*(L1+LC2))/Mt;c_foot=M2*LC2/(M1*LC1+M2*L1+M2*LC2)
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
    foot = data.xpos[model.body('lower_body').id]
    com_l = data.xipos[model.body('lower_body').id]
    com_u = data.xipos[model.body('upper_body').id]
    com = (M1 * com_l + M2 * com_u) / Mt
    return np.arctan2(com[0]-foot[0], com[2]-foot[2])


model = mujoco.MjModel.from_xml_string(xml)

# --- Test: constant delta command from beta=+5d ---
# Track beta every ms for first 200ms
fig, axes = plt.subplots(4, 1, figsize=(14, 14))

# Also track foot pos and CoM pos separately
for dd in [0, +10, +20, +30, +40]:
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(5)
    mujoco.mj_forward(model, data)

    N = int(0.3 / DT)
    ts=[]; betas=[]; foot_xs=[]; com_xs=[]; dacts=[]

    for i in range(N):
        beta = compute_beta(model, data)
        foot = data.xpos[model.body('lower_body').id].copy()
        com_l = data.xipos[model.body('lower_body').id].copy()
        com_u = data.xipos[model.body('upper_body').id].copy()
        com = (M1*com_l + M2*com_u) / Mt

        data.ctrl[0] = np.radians(dd)
        ts.append(data.time*1000)
        betas.append(np.degrees(beta))
        foot_xs.append(foot[0]*1000)  # mm
        com_xs.append(com[0]*1000)    # mm
        dacts.append(np.degrees(data.joint('hip').qpos[0]))
        mujoco.mj_step(model, data)

    label = f'd={dd}d'
    axes[0].plot(ts, betas, lw=1.5, label=label)
    axes[1].plot(ts, foot_xs, lw=1.5, label=label)
    axes[2].plot(ts, com_xs, lw=1.5, label=label)
    axes[3].plot(ts, dacts, lw=1.5, label=label)

    # Find min beta
    betas_arr = np.array(betas)
    imin = np.argmin(betas_arr)
    print(f"  d={dd:+2d}d: beta_min={betas_arr[imin]:.2f}d @ t={ts[imin]:.0f}ms  "
          f"beta@100ms={betas_arr[100]:.2f}d  beta@200ms={betas_arr[200]:.2f}d")

axes[0].set_ylabel('beta [d]')
axes[0].axhline(0, color='black', lw=1)
axes[0].axhline(5, color='gray', ls=':', alpha=0.5, label='initial')
axes[0].legend(fontsize=9)
axes[0].set_title('beta during constant fold (beta0=+5d) — DOES beta go negative?')

axes[1].set_ylabel('foot X [mm]')
axes[1].legend(fontsize=9)

axes[2].set_ylabel('CoM X [mm]')
axes[2].legend(fontsize=9)

axes[3].set_ylabel('hip actual [d]')
axes[3].set_xlabel('time [ms]')
axes[3].legend(fontsize=9)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_beta_detail.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

"""
Fix the servo saturation problem.
With KP=500 and TAU_MAX=50, the servo is ALWAYS saturated.
Options:
  1) Increase TAU_MAX so servo can actually track different deltas
  2) Decrease KP so torque demand is lower
  3) Or just use very high TAU to make it effectively kinematic

Let's try TAU_MAX=200 (strong servo) and see if different delta commands
actually produce different behavior.
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

def make_xml(kp, kd, tau_max):
    return f"""<?xml version="1.0"?>
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
             gainprm="{kp} 0 0" biasprm="0 {-kp} {-kd}"
             ctrlrange="-1.57 1.57" ctrllimited="true"
             forcerange="{-tau_max} {tau_max}" forcelimited="true"/>
  </actuator>
</mujoco>"""

def compute_beta(model, data):
    foot = data.xpos[model.body('lower_body').id]
    com_l = data.xipos[model.body('lower_body').id]
    com_u = data.xipos[model.body('upper_body').id]
    com = (M1 * com_l + M2 * com_u) / Mt
    return np.arctan2(com[0]-foot[0], com[2]-foot[2])


# Compare different servo settings
servo_configs = [
    (500,  2,   50, "KP500 TAU50 (current)"),
    (500,  2,  500, "KP500 TAU500"),
    (100,  1,  500, "KP100 TAU500"),
    (2000, 5, 2000, "KP2000 TAU2000"),
]

fig, axes = plt.subplots(3, len(servo_configs), figsize=(6*len(servo_configs), 12))

for col, (kp, kd, tau, title) in enumerate(servo_configs):
    xml = make_xml(kp, kd, tau)
    model = mujoco.MjModel.from_xml_string(xml)

    print(f"\n{'='*50}")
    print(f"{title}")
    
    for dd in [10, 20, 30, 45]:
        data = mujoco.MjData(model)
        data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(5)
        mujoco.mj_forward(model, data)

        N = int(1.0 / DT)
        ts=[]; betas=[]; dacts=[]; taus_log=[]

        for i in range(N):
            beta = compute_beta(model, data)
            hip_act = data.joint('hip').qpos[0]
            data.ctrl[0] = np.radians(dd)
            ts.append(data.time*1000)
            betas.append(np.degrees(beta))
            dacts.append(np.degrees(hip_act))
            taus_log.append(data.actuator_force[0])
            mujoco.mj_step(model, data)
            if np.any(np.isnan(data.qpos)): break

        axes[0,col].plot(ts, betas, lw=1.5, label=f'd={dd}°')
        axes[1,col].plot(ts, dacts, lw=1.5, label=f'd={dd}°')
        axes[2,col].plot(ts, taus_log, lw=1, label=f'd={dd}°')

        betas_arr = np.array(betas)
        imin = np.argmin(betas_arr)
        print(f"  d={dd:2d}°: beta_min={betas_arr[imin]:.2f}° @{ts[imin]:.0f}ms"
              f"  hip@50ms={dacts[50]:.1f}°  tau@50ms={taus_log[50]:.1f}Nm")

    axes[0,col].axhline(0, color='red', lw=1)
    axes[0,col].set_ylabel('beta [°]')
    axes[0,col].set_title(title, fontsize=10)
    axes[0,col].legend(fontsize=8)
    axes[0,col].set_ylim(-30, 50)

    axes[1,col].set_ylabel('hip actual [°]')
    axes[1,col].legend(fontsize=8)

    axes[2,col].set_ylabel('torque [Nm]')
    axes[2,col].set_xlabel('ms')
    axes[2,col].legend(fontsize=8)

plt.suptitle('Servo tuning: do different delta commands produce different behavior?', fontsize=13)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_servo_tuning.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

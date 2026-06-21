"""
K1=4.5, T_w1=60ms 상세 시계열 진단
"""
import numpy as np, mujoco, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

T_FOLD = 0.020
K1 = 4.5
T_w1 = 0.060

data = mujoco.MjData(model)
data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(5)
mujoco.mj_forward(model, data)

FOLD, WAIT1, EXTEND, EW = 0, 1, 2, 3
md = np.radians(60)

beta = compute_beta(model, data); bp = beta; bd = 0.0
beta_eff = beta
delta_target = np.clip(K1 * beta_eff, -md, md)
delta_start = 0.0

state = FOLD; t_ph = 0; cyc = 1; pp = 0

# Recording arrays
rec_t=[]; rec_beta=[]; rec_phi=[]; rec_delta_cmd=[]; rec_delta_act=[]
rec_phase=[]; rec_beta_dot=[]; rec_ctrl=[]

# Cycle transition log
transitions = []

N = int(4.0 / DT)  # 4 seconds
for i in range(N):
    phi = data.joint('phi_Y').qpos[0]
    beta = compute_beta(model, data)
    bd = (beta - bp) / DT if i > 0 else 0.0
    ha = data.joint('hip').qpos[0]
    t_ph += DT
    
    # Record every step
    if i % 2 == 0:
        rec_t.append(data.time)
        rec_beta.append(np.degrees(beta))
        rec_phi.append(np.degrees(phi))
        rec_delta_cmd.append(np.degrees(delta_target))
        rec_delta_act.append(np.degrees(ha))
        rec_phase.append(state)
        rec_beta_dot.append(np.degrees(bd))
    
    if state == FOLD:
        frac = min(t_ph / T_FOLD, 1.0)
        ctrl = delta_start + frac * (delta_target - delta_start)
        if t_ph >= T_FOLD: state = WAIT1; t_ph = 0
    elif state == WAIT1:
        ctrl = delta_target
        if t_ph >= T_w1:
            delta_start = delta_target; delta_target = 0.0
            state = EXTEND; t_ph = 0
    elif state == EXTEND:
        frac = min(t_ph / T_FOLD, 1.0)
        ctrl = delta_start + frac * (delta_target - delta_start)
        if t_ph >= T_FOLD: state = EW; t_ph = 0; pp = phi
    elif state == EW:
        ctrl = 0.0
        if t_ph > 0.005 and pp * phi < 0:
            transitions.append({
                'cyc': cyc, 't': data.time,
                'beta': np.degrees(beta), 'beta_dot': np.degrees(bd),
                'phi': np.degrees(phi), 'beta_eff': np.degrees(beta + bd*T_FOLD),
                'delta': np.degrees(np.clip(K1*(beta+bd*T_FOLD), -md, md))
            })
            cyc += 1
            if cyc > 15 or abs(np.degrees(beta)) > 170: break
            beta_eff = beta + bd * T_FOLD
            delta_start = 0.0
            delta_target = np.clip(K1 * beta_eff, -md, md)
            state = FOLD; t_ph = 0
        pp = phi
        if t_ph > 3: break
    
    data.ctrl[0] = ctrl; bp = beta
    mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)): break

# Print transition details
print(f"K1={K1}, T_w1={T_w1*1000:.0f}ms, T_fold={T_FOLD*1000:.0f}ms")
print(f"{'cyc':>3} {'t_ms':>7} {'beta':>7} {'b_dot':>8} {'b_eff':>7} {'delta':>7} {'phi':>7}")
print("-"*55)
for tr in transitions:
    print(f"{tr['cyc']:3d} {tr['t']*1000:7.0f} {tr['beta']:+7.1f} {tr['beta_dot']:+8.1f} "
          f"{tr['beta_eff']:+7.1f} {tr['delta']:+7.1f} {tr['phi']:+7.2f}")

# === Plot ===
fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
tms = [t*1000 for t in rec_t]

# 1. Beta
axes[0].plot(tms, rec_beta, 'r-', lw=1.5)
axes[0].axhline(0, color='k', ls='--', alpha=0.3)
for tr in transitions:
    axes[0].axvline(tr['t']*1000, color='gray', ls=':', alpha=0.5)
    axes[0].annotate(f"C{tr['cyc']}: {tr['beta']:+.1f}",
                     (tr['t']*1000, tr['beta']), fontsize=8,
                     textcoords="offset points", xytext=(5, 10))
axes[0].set_ylabel('beta [deg]')
axes[0].set_title(f'K1={K1}, T_w1={T_w1*1000:.0f}ms, T_fold={T_FOLD*1000:.0f}ms')
axes[0].grid(True, alpha=0.3)

# 2. Beta_dot
axes[1].plot(tms, rec_beta_dot, 'darkred', lw=1)
axes[1].axhline(0, color='k', ls='--', alpha=0.3)
for tr in transitions:
    axes[1].axvline(tr['t']*1000, color='gray', ls=':', alpha=0.5)
axes[1].set_ylabel('beta_dot [deg/s]')
axes[1].grid(True, alpha=0.3)

# 3. Phi
axes[2].plot(tms, rec_phi, 'b-', lw=1.5)
axes[2].axhline(0, color='k', ls='--', alpha=0.3)
for tr in transitions:
    axes[2].axvline(tr['t']*1000, color='gray', ls=':', alpha=0.5)
axes[2].set_ylabel('phi [deg]')
axes[2].grid(True, alpha=0.3)

# 4. Delta cmd vs actual
axes[3].plot(tms, rec_delta_cmd, 'g-', lw=1, label='cmd (target)')
axes[3].plot(tms, rec_delta_act, 'm-', lw=1, label='actual')
axes[3].axhline(0, color='k', ls='--', alpha=0.3)
for tr in transitions:
    axes[3].axvline(tr['t']*1000, color='gray', ls=':', alpha=0.5)
axes[3].set_ylabel('delta [deg]')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

# 5. Phase
phase_colors = {0:'red', 1:'orange', 2:'green', 3:'blue'}
phase_names = {0:'FOLD', 1:'WAIT1', 2:'EXTEND', 3:'EXT_WAIT'}
for j in range(len(tms)-1):
    axes[4].axvspan(tms[j], tms[j+1], alpha=0.4, color=phase_colors.get(rec_phase[j], 'gray'))
for tr in transitions:
    axes[4].axvline(tr['t']*1000, color='gray', ls=':', alpha=0.5)
axes[4].set_ylabel('Phase')
axes[4].set_xlabel('Time [ms]')
axes[4].set_yticks([0,1,2,3])
axes[4].set_yticklabels(['FOLD','WAIT1','EXTEND','EXT_WAIT'])

plt.tight_layout()
plt.savefig('v15_diverge_analysis.png', dpi=150)
print("\nSaved v15_diverge_analysis.png")

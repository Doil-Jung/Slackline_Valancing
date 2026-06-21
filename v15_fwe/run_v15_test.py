"""V15 Headless Test — Run FWE and plot convergence"""
import os, sys
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fwe_controller import FWEController, Phase

# ============================================================
# Robot params (same as v15_mujoco.py)
# ============================================================
M1=0.1; M2=0.25; L1=0.2; L2=0.2; BD=0.10; GRAV=9.81; DT=0.001
L_POLE=0.8; POLE_H=1.0; R_TARGET=0.3
KP_SERVO=200.0; KD_SERVO=4.0; B_THETA=0.01
LC1=L1/2; LC2=L2/2; Mt=M1+M2
p1=M1*LC1+M2*L1; p2=M2*LC2; ps=p1+p2
h_com=(M1*LC1+M2*(L1+LC2))/Mt; c_foot=p2/ps

HALF_POLE=L_POLE/2; ROPE_HORIZ=HALF_POLE-BD/2; ROPE_VERT=R_TARGET
FOOT_Z=POLE_H-ROPE_VERT; ROPE_MASS=0.01
LOWER_W=L2/4; UPPER_W=L2/4
POLE_RADIUS=max(0.005,L2*0.05); POLE_TOP_R=max(0.008,L2*0.075)
ROPE_RADIUS=max(0.003,L2*0.015)
HIP_R=max(0.008,L2*0.08); HIP_H=max(0.005,BD*0.15)

def generate_mjcf():
    rh=ROPE_HORIZ;rv=ROPE_VERT;hp=HALF_POLE
    lw2=LOWER_W/2;uw2=UPPER_W/2;bd2=BD/2
    return f"""<?xml version="1.0"?>
<mujoco model="slackline_v15_fwe">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0.001"/>
  </default>
  <worldbody>
    <geom type="plane" size="3 3 0.01" rgba="0.3 0.3 0.35 1" contype="1" conaffinity="1"/>
    <geom name="pole_a" type="cylinder" pos="0 {hp} {POLE_H/2}" size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <geom name="pole_b" type="cylinder" pos="0 {-hp} {POLE_H/2}" size="{POLE_RADIUS} {POLE_H/2}" rgba="0.5 0.5 0.5 1"/>
    <body name="rope_a_mount" pos="0 {hp} {POLE_H}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_a_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_a" type="capsule" size="{ROPE_RADIUS}" fromto="0 0 0 0 {-rh} {-rv}" rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="lower_body" pos="0 {-rh} {-rv}">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom name="lower_geom" type="box" size="{lw2} {bd2} {L1/2}" pos="0 {-bd2} {L1/2}" mass="{M1}" rgba="0.02 0.84 0.63 0.8"/>
        <body name="ankle_b_target" pos="0 {-BD} 0"/>
        <geom name="hip_cyl" type="cylinder" size="{HIP_R} {HIP_H}" pos="0 {-bd2} {L1}" euler="90 0 0" rgba="1 0.75 0.04 1" mass="0.01"/>
        <body name="upper_body" pos="0 {-bd2} {L1}">
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}"/>
          <geom name="upper_geom" type="box" size="{uw2} {bd2} {L2/2}" pos="0 0 {L2/2}" mass="{M2}" rgba="0.51 0.22 0.93 0.8"/>
        </body>
      </body>
    </body>
    <body name="rope_b_mount" pos="0 {-hp} {POLE_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_b_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_b" type="capsule" size="{ROPE_RADIUS}" fromto="0 0 0 0 {rh} {-rv}" rgba="1 0.8 0 1" mass="{ROPE_MASS}"/>
      <body name="rope_b_end" pos="0 {rh} {-rv}"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0"
             solref="0.001 1" solimp="0.999 0.999 0.0001"/>
  </equality>
</mujoco>"""


def get_state(model, data):
    phi = data.joint('phi_Y').qpos[0]
    lb_id = model.body('lower_body').id
    xmat = data.xmat[lb_id].reshape(3,3)
    alpha = np.arctan2(-xmat[2,0], xmat[2,2])
    ub_id = model.body('upper_body').id
    xmat_u = data.xmat[ub_id].reshape(3,3)
    theta = np.arctan2(-xmat_u[2,0], xmat_u[2,2])
    return phi, alpha, theta


def run_test(eps, T_f, T_w1, alpha0, theta0, sim_time=3.0):
    """Run headless simulation, return time series."""
    model = mujoco.MjModel.from_xml_string(generate_mjcf())
    data = mujoco.MjData(model)

    hip_qadr = model.jnt_qposadr[model.joint('hip').id]
    ankle_qadr = model.jnt_qposadr[model.joint('ankle').id]

    data.qpos[ankle_qadr] = alpha0
    data.qpos[hip_qadr] = theta0 - alpha0
    mujoco.mj_forward(model, data)

    fwe = FWEController(
        eps=eps, T_f=T_f, T_w1=T_w1,
        h_com=h_com, c_foot=c_foot,
        p1=p1, p2=p2, Mt=Mt,
        beta_threshold=np.radians(1.0)
    )

    N = int(sim_time / DT)
    ts = np.zeros(N)
    phis = np.zeros(N)
    betas = np.zeros(N)
    deltas_cmd = np.zeros(N)
    deltas_act = np.zeros(N)
    phases = np.zeros(N, dtype=int)

    for i in range(N):
        phi, alpha, theta = get_state(model, data)
        beta = (p1*alpha + p2*theta) / (Mt*h_com)

        delta_cmd = fwe.step(DT, phi, alpha, theta)
        # Direct kinematic control: set hip qpos directly
        data.qpos[hip_qadr] = delta_cmd
        # No actuator — position set directly via qpos

        ts[i] = data.time
        phis[i] = phi
        betas[i] = beta
        deltas_cmd[i] = delta_cmd
        deltas_act[i] = data.joint('hip').qpos[0]
        phases[i] = list(Phase).index(fwe.phase)

        mujoco.mj_step(model, data)

    return ts, phis, betas, deltas_cmd, deltas_act, phases, fwe


# ============================================================
# Run test
# ============================================================
print("V15 Headless Test - FWE 4-Phase")
print(f"  Robot: M1={M1} M2={M2} L1={L1} L2={L2} R={R_TARGET}")
print(f"  h_com={h_com:.4f} c_foot={c_foot:.4f}")

# Test with best robot params
EPS = 4.0; TF = 0.150; TW1 = 0.200
ALPHA0 = np.radians(5.0)
THETA0 = np.radians(5.0)
SIM_TIME = 3.0

print(f"\n  FWE: eps={EPS} Tf={TF*1000:.0f}ms Tw1={TW1*1000:.0f}ms")
print(f"  Init: alpha0={np.degrees(ALPHA0):.1f}d theta0={np.degrees(THETA0):.1f}d")
print(f"  Sim: {SIM_TIME}s")
print()

ts, phis, betas, dcmd, dact, phases, fwe = run_test(
    EPS, TF, TW1, ALPHA0, THETA0, SIM_TIME)

# Plot
fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

phase_colors = {0:'white', 1:'#FFD700', 2:'#90EE90', 3:'#87CEEB', 4:'#DDA0DD'}
phase_names = {0:'IDLE', 1:'FOLD', 2:'WAIT1', 3:'EXTEND', 4:'WAIT2'}

# Shade phases
for ax in axes:
    prev_p = phases[0]
    start_t = ts[0]
    for i in range(1, len(phases)):
        if phases[i] != prev_p or i == len(phases)-1:
            ax.axvspan(start_t*1000, ts[i]*1000, alpha=0.15,
                      color=phase_colors.get(prev_p, 'white'))
            prev_p = phases[i]
            start_t = ts[i]

axes[0].plot(ts*1000, np.degrees(phis), 'b-', lw=1.5)
axes[0].axhline(0, color='red', ls='--', alpha=0.5)
axes[0].set_ylabel('phi [deg]', fontsize=12)
axes[0].set_title(f'V15 FWE MuJoCo: eps={EPS}, Tf={TF*1000:.0f}ms, '
                  f'Tw1={TW1*1000:.0f}ms | beta0=5d', fontsize=12)

axes[1].plot(ts*1000, np.degrees(betas), 'r-', lw=1.5)
axes[1].axhline(0, color='black', ls='-', lw=0.5)
axes[1].set_ylabel('beta [deg]', fontsize=12)

axes[2].plot(ts*1000, np.degrees(dcmd), 'g-', lw=1.5, label='cmd')
axes[2].plot(ts*1000, np.degrees(dact), 'k--', lw=1, alpha=0.5, label='actual')
axes[2].set_ylabel('delta [deg]', fontsize=12)
axes[2].legend(fontsize=9)

axes[3].plot(ts*1000, phases, 'k-', lw=1)
axes[3].set_yticks(range(5))
axes[3].set_yticklabels(['IDLE','FOLD','W1','EXT','W2'])
axes[3].set_ylabel('Phase', fontsize=12)
axes[3].set_xlabel('time [ms]', fontsize=12)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_test_result.png')
plt.savefig(out, dpi=150)
print(f"\nSaved: {out}")

# Print FWE log
print("\n--- FWE Phase Log ---")
for msg in fwe.phase_log:
    print(f"  {msg}")

print(f"\nFinal: phi={np.degrees(phis[-1]):.2f}d beta={np.degrees(betas[-1]):.2f}d")
print(f"Cycles completed: {fwe.cycle_count}")
print("\nDone!")

# -*- coding: utf-8 -*-
"""
ODE 수렴 파라미터를 MuJoCo에서 검증
eps=11.5, T_f=20ms, T_w1=340ms (r=-0.34 in ODE)
서보 제한을 매우 크게 해서 이상적 조건에서 테스트
"""
import numpy as np, mujoco, time, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === Robot parameters ===
M1=0.1; M2=0.25; L1=0.2; L2=0.2; BD=0.10; GRAV=9.81; DT=0.001
POLE_H=1.0; L_POLE=0.8; R_TARGET=0.3; B_THETA=0.01
Mt=M1+M2; HP=L_POLE/2; RH=HP-BD/2; RV=R_TARGET
lw2=L2/8; uw2=L2/8; bd2=BD/2
PR=max(0.005,L2*0.05); RR=max(0.003,L2*0.015); RM=0.01

# Derived physical parameters (same as robot_4phase_check.py)
LC1=L1/2; LC2=L2/2
p1=M1*LC1+M2*L1; p2=M2*LC2; ps=p1+p2
h_com=(M1*LC1+M2*(L1+LC2))/Mt
c_foot=p2/ps

# === Servo: VERY HIGH limits (idealized) ===
KP = 5000       # very stiff position tracking
KD_s = 50       # damping
TAU_MAX = 100000 # essentially unlimited torque

# === FWE parameters (from ODE convergence) ===
EPS = 11.5
T_FOLD = 0.020   # 20ms
T_W1 = 0.340     # 340ms
BETA0_DEG = 5.0

print("=" * 60)
print("MuJoCo Verification of ODE-convergent FWE parameters")
print("=" * 60)
print(f"  Robot: M1={M1} M2={M2} L1={L1} L2={L2} R={R_TARGET}")
print(f"  h_com={h_com:.4f}m  c_foot={c_foot:.4f}")
print(f"  FWE: eps={EPS} T_f={T_FOLD*1000:.0f}ms T_w1={T_W1*1000:.0f}ms")
print(f"  Servo: KP={KP} KD={KD_s} TAU_MAX={TAU_MAX}")
d_fold_deg = np.degrees(h_com*(1+EPS)*np.radians(BETA0_DEG)/c_foot)
v_peak = 2*d_fold_deg/T_FOLD
print(f"  delta_fold(beta0={BETA0_DEG}d) = {d_fold_deg:.1f}deg")
print(f"  v_peak = {v_peak:.0f} deg/s (idealized, no limit)")
print()

# === MuJoCo model ===
xml = f"""<?xml version="1.0"?>
<mujoco model="v15_ode_verify">
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
                 range="-3.14 3.14" limited="true"/>
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
             ctrlrange="-3.14 3.14" ctrllimited="true"
             forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/>
  </actuator>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)

def compute_beta(m, d):
    """CoM-based beta calculation"""
    foot = d.xpos[m.body('lower_body').id]
    cl = d.xipos[m.body('lower_body').id]
    cu = d.xipos[m.body('upper_body').id]
    com = (M1*cl + M2*cu) / Mt
    return np.arctan2(com[0]-foot[0], com[2]-foot[2])

# === Run simulation ===
data = mujoco.MjData(model)
data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(BETA0_DEG)
mujoco.mj_forward(model, data)

# State machine
FOLD, WAIT1, EXTEND, EW = 0, 1, 2, 3
state_names = ['FOLD', 'WAIT1', 'EXTEND', 'EXT_WAIT']

beta = compute_beta(model, data)
bp = beta; bd = 0.0

# First cycle: delta from beta
delta_target = h_com * (1+EPS) * beta / c_foot  # signed
delta_start = 0.0
md = np.radians(120)  # very large joint limit
delta_target = np.clip(delta_target, -md, md)

state = FOLD; t_ph = 0; cyc = 1; pp = 0

# Recording
rec_t=[]; rec_beta=[]; rec_phi=[]; rec_delta_cmd=[]; rec_delta_act=[]
rec_phase=[]; rec_beta_dot=[]
transitions = []

SIM_TIME = 6.0  # seconds
MAX_CYC = 20
N = int(SIM_TIME / DT)

print(f"Running {SIM_TIME}s simulation ({N} steps)...")
print(f"Initial beta = {np.degrees(beta):.2f} deg")
print(f"Initial delta_target = {np.degrees(delta_target):.1f} deg")
print()

t_start = time.time()

for i in range(N):
    phi = data.joint('phi_Y').qpos[0]
    beta = compute_beta(model, data)
    bd = (beta - bp) / DT if i > 0 else 0.0
    ha = data.joint('hip').qpos[0]
    t_ph += DT
    
    # Record
    if i % 2 == 0:
        rec_t.append(data.time)
        rec_beta.append(np.degrees(beta))
        rec_phi.append(np.degrees(phi))
        rec_delta_cmd.append(np.degrees(delta_target))
        rec_delta_act.append(np.degrees(ha))
        rec_phase.append(state)
        rec_beta_dot.append(np.degrees(bd))
    
    # State machine
    if state == FOLD:
        frac = min(t_ph / T_FOLD, 1.0)
        ctrl = delta_start + frac * (delta_target - delta_start)
        if t_ph >= T_FOLD:
            state = WAIT1; t_ph = 0
    
    elif state == WAIT1:
        ctrl = delta_target
        if t_ph >= T_W1:
            delta_start = delta_target
            delta_target = 0.0
            state = EXTEND; t_ph = 0
    
    elif state == EXTEND:
        frac = min(t_ph / T_FOLD, 1.0)
        ctrl = delta_start + frac * (delta_target - delta_start)
        if t_ph >= T_FOLD:
            state = EW; t_ph = 0; pp = phi
    
    elif state == EW:
        ctrl = 0.0
        # Wait for phi zero-crossing
        if t_ph > 0.005 and pp * phi < 0:
            # Log transition
            transitions.append({
                'cyc': cyc, 't': data.time,
                'beta': np.degrees(beta),
                'beta_dot': np.degrees(bd),
                'phi': np.degrees(phi),
            })
            cyc += 1
            if cyc > MAX_CYC or abs(np.degrees(beta)) > 170:
                break
            
            # Next cycle: compute new delta from current beta
            beta_eff = beta + bd * T_FOLD
            delta_start = 0.0
            delta_target = h_com * (1+EPS) * beta_eff / c_foot
            delta_target = np.clip(delta_target, -md, md)
            state = FOLD; t_ph = 0
        pp = phi
        if t_ph > 3:
            transitions.append({
                'cyc': cyc, 't': data.time,
                'beta': np.degrees(beta),
                'beta_dot': np.degrees(bd),
                'phi': np.degrees(phi),
            })
            break
    
    data.ctrl[0] = ctrl
    bp = beta
    mujoco.mj_step(model, data)
    if np.any(np.isnan(data.qpos)):
        print(f"NaN at t={data.time:.3f}s!")
        break

elapsed = time.time() - t_start
print(f"Simulation done in {elapsed:.1f}s")
print()

# === Print results ===
print(f"{'cyc':>3} {'t(ms)':>7} {'beta(d)':>8} {'bdot':>8} {'phi(d)':>7}")
print("-" * 40)
for tr in transitions:
    print(f"{tr['cyc']:3d} {tr['t']*1000:7.0f} {tr['beta']:+8.2f} "
          f"{tr['beta_dot']:+8.1f} {tr['phi']:+7.2f}")

# Check convergence
if len(transitions) >= 2:
    betas = [tr['beta'] for tr in transitions]
    print(f"\nbeta sequence: {[f'{b:+.2f}' for b in betas]}")
    
    abs_betas = [abs(b) for b in betas]
    if len(abs_betas) >= 3:
        ratios = [abs_betas[i+1]/abs_betas[i] if abs_betas[i] > 0.01 else 0 
                  for i in range(len(abs_betas)-1)]
        print(f"decay ratios:  {[f'{r:.3f}' for r in ratios]}")
    
    if all(abs_betas[i+1] < abs_betas[i] for i in range(min(4, len(abs_betas)-1))):
        print("\n*** CONVERGING! beta is decreasing each cycle ***")
    elif any(abs(b) > 90 for b in betas):
        print("\n*** DIVERGING! beta exceeded 90 deg ***")
    else:
        print(f"\n*** Unclear. Max |beta| = {max(abs_betas):.1f} deg ***")

# === Plot ===
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
tms = [t*1000 for t in rec_t]

# Beta
axes[0].plot(tms, rec_beta, 'r-', lw=1.5)
axes[0].axhline(0, color='k', ls='--', alpha=0.3)
for tr in transitions:
    axes[0].axvline(tr['t']*1000, color='gray', ls=':', alpha=0.5)
    axes[0].annotate(f"C{tr['cyc']}: {tr['beta']:+.1f}d",
                     (tr['t']*1000, tr['beta']), fontsize=8,
                     textcoords="offset points", xytext=(5, 10))
axes[0].set_ylabel('beta [deg]')
axes[0].set_title(f'ODE-verified FWE in MuJoCo: eps={EPS} Tf={T_FOLD*1000:.0f}ms '
                  f'Tw1={T_W1*1000:.0f}ms | KP={KP} TAU={TAU_MAX}')
axes[0].grid(True, alpha=0.3)

# Phi
axes[1].plot(tms, rec_phi, 'b-', lw=1.5)
axes[1].axhline(0, color='k', ls='--', alpha=0.3)
for tr in transitions:
    axes[1].axvline(tr['t']*1000, color='gray', ls=':', alpha=0.5)
axes[1].set_ylabel('phi [deg]')
axes[1].grid(True, alpha=0.3)

# Delta
axes[2].plot(tms, rec_delta_cmd, 'g-', lw=1, label='cmd')
axes[2].plot(tms, rec_delta_act, 'm-', lw=1, label='actual')
axes[2].axhline(0, color='k', ls='--', alpha=0.3)
axes[2].set_ylabel('delta [deg]')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Phase
pc = {0:'red', 1:'orange', 2:'green', 3:'blue'}
for j in range(len(tms)-1):
    axes[3].axvspan(tms[j], tms[j+1], alpha=0.3, color=pc.get(rec_phase[j], 'gray'))
axes[3].set_ylabel('Phase')
axes[3].set_xlabel('Time [ms]')
axes[3].set_yticks([0,1,2,3])
axes[3].set_yticklabels(['FOLD','W1','EXTEND','EW'])

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_ode_verify.png')
plt.savefig(out_path, dpi=150)
print(f"\nSaved: {out_path}")

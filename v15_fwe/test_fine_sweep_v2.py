# -*- coding: utf-8 -*-
"""
MuJoCo fine sweep: K1=2~8, T_w1=20~200ms
Control: delta = K1 * beta_eff, beta_eff = beta + bdot * T_fold
Also tries PD variant: delta = K1 * beta + K2 * bdot
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

def run_fwe(K1, T_w1, K2=0, beta0_deg=5, sim_time=6.0, max_cyc=15, record=False):
    """
    Control law: delta = K1 * beta_eff + K2 * beta_dot
    where beta_eff = beta + beta_dot * T_FOLD
    
    If K2=0: delta = K1 * (beta + bdot*T_FOLD) = K1*beta + K1*T_FOLD*bdot
    If K2!=0: delta = K1*beta + K2*bdot (PD-like)
    """
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(beta0_deg)
    mujoco.mj_forward(model, data)
    
    FOLD, WAIT1, EXTEND, EW = 0, 1, 2, 3
    md = np.radians(60)
    
    beta = compute_beta(model, data); bp = beta; bd = 0.0
    
    if K2 == 0:
        delta_target = np.clip(K1 * beta, -md, md)  # first cycle: bdot=0
    else:
        delta_target = np.clip(K1 * beta, -md, md)
    delta_start = 0.0
    
    state = FOLD; t_ph = 0; cyc = 1; pp = 0
    betas_at_transition = []
    rec = {'t':[], 'b':[], 'p':[], 'dc':[], 'da':[], 'ph':[]} if record else None
    
    N = int(sim_time / DT)
    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        beta = compute_beta(model, data)
        bd = (beta - bp) / DT if i > 0 else 0.0
        ha = data.joint('hip').qpos[0]
        t_ph += DT
        
        if record and i % 5 == 0:
            rec['t'].append(data.time); rec['b'].append(np.degrees(beta))
            rec['p'].append(np.degrees(phi)); rec['dc'].append(np.degrees(delta_target))
            rec['da'].append(np.degrees(ha)); rec['ph'].append(state)
        
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
                betas_at_transition.append(np.degrees(beta))
                cyc += 1
                if cyc > max_cyc or abs(np.degrees(beta)) > 170: break
                if K2 == 0:
                    beta_eff = beta + bd * T_FOLD
                    delta_target = np.clip(K1 * beta_eff, -md, md)
                else:
                    delta_target = np.clip(K1 * beta + K2 * bd, -md, md)
                delta_start = 0.0
                state = FOLD; t_ph = 0
            pp = phi
            if t_ph > 3: betas_at_transition.append(np.degrees(beta)); break
        
        data.ctrl[0] = ctrl; bp = beta
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)): break
    
    if record:
        return betas_at_transition, rec
    return betas_at_transition

# =============================================================
# Sweep 1: K1 fine, T_w1 fine (proportional control)
# =============================================================
print("=" * 70)
print("Sweep 1: delta = K1 * (beta + bdot*T_fold)")
print(f"T_FOLD={T_FOLD*1000:.0f}ms, beta0=5deg")
print("=" * 70)

K1_vals = np.arange(2.0, 8.1, 0.5)
Tw1_vals = np.arange(0.020, 0.201, 0.010)

t0 = time.time()
results = []

for K1 in K1_vals:
    for tw in Tw1_vals:
        bai = run_fwe(K1=K1, T_w1=tw)
        if bai and len(bai) >= 3:
            ab = [abs(b) for b in bai]
            mx = max(ab[:min(6,len(ab))])
            # Check if converging: each successive |beta| smaller
            converging = all(ab[i+1] < ab[i]*1.1 for i in range(min(4,len(ab)-1)))
            results.append({
                'K1':K1, 'tw':tw, 'nc':len(bai), 'max':mx,
                'bai':bai, 'conv':converging
            })

elapsed = time.time() - t0
print(f"Done: {elapsed:.0f}s, {len(results)} valid results\n")

# Sort by max beta in first 6 cycles
results.sort(key=lambda x: x['max'])

print(f"{'#':>3} {'K1':>5} {'Tw1':>6} {'nc':>3} {'max':>6} {'conv':>5}  betas")
print("-" * 90)
for rank, r in enumerate(results[:30]):
    bai_s = ", ".join([f"{b:+.1f}" for b in r['bai'][:8]])
    tag = "***" if r['conv'] and r['max'] < 10 else ("**" if r['conv'] else "")
    print(f"{rank+1:3d} {r['K1']:5.1f} {r['tw']*1000:5.0f}ms {r['nc']:3d} "
          f"{r['max']:6.1f} {tag:>5}  [{bai_s}]")

# =============================================================
# Sweep 2: PD control: delta = K1*beta + K2*bdot
# =============================================================
print(f"\n{'='*70}")
print("Sweep 2: delta = K1*beta + K2*bdot (PD control)")
print(f"{'='*70}")

K1_pd = [3.0, 4.0, 5.0, 6.0]
K2_pd = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
Tw1_pd = [0.040, 0.060, 0.080, 0.100, 0.120]

results_pd = []
for K1 in K1_pd:
    for K2 in K2_pd:
        for tw in Tw1_pd:
            bai = run_fwe(K1=K1, T_w1=tw, K2=K2)
            if bai and len(bai) >= 3:
                ab = [abs(b) for b in bai]
                mx = max(ab[:min(6,len(ab))])
                converging = all(ab[i+1] < ab[i]*1.1 for i in range(min(4,len(ab)-1)))
                results_pd.append({
                    'K1':K1, 'K2':K2, 'tw':tw, 'nc':len(bai), 'max':mx,
                    'bai':bai, 'conv':converging
                })

results_pd.sort(key=lambda x: x['max'])
print(f"\n{'#':>3} {'K1':>5} {'K2':>6} {'Tw1':>6} {'nc':>3} {'max':>6} {'conv':>5}  betas")
print("-" * 95)
for rank, r in enumerate(results_pd[:20]):
    bai_s = ", ".join([f"{b:+.1f}" for b in r['bai'][:8]])
    tag = "***" if r['conv'] and r['max'] < 10 else ("**" if r['conv'] else "")
    print(f"{rank+1:3d} {r['K1']:5.1f} {r['K2']:6.3f} {r['tw']*1000:5.0f}ms {r['nc']:3d} "
          f"{r['max']:6.1f} {tag:>5}  [{bai_s}]")

# =============================================================
# Plot best case
# =============================================================
all_results = results + results_pd
conv_cases = [r for r in all_results if r.get('conv') and r['max'] < 30]
if not conv_cases:
    conv_cases = all_results[:1]

conv_cases.sort(key=lambda x: x['max'])
best = conv_cases[0]

K2_best = best.get('K2', 0)
bai, rec = run_fwe(K1=best['K1'], T_w1=best['tw'], K2=K2_best, record=True)

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
tms = [t*1000 for t in rec['t']]

axes[0].plot(tms, rec['b'], 'r-', lw=1.5)
axes[0].axhline(0, color='k', ls='--', alpha=0.3)
axes[0].set_ylabel('beta [deg]')
ctrl_label = f"K1={best['K1']}"
if K2_best: ctrl_label += f" K2={K2_best}"
axes[0].set_title(f"Best: {ctrl_label} Tw1={best['tw']*1000:.0f}ms | "
                  f"max|beta|={best['max']:.1f}d | "
                  f"betas={[f'{b:+.1f}' for b in bai[:6]]}")
axes[0].grid(True, alpha=0.3)

axes[1].plot(tms, rec['p'], 'b-', lw=1.5)
axes[1].axhline(0, color='k', ls='--', alpha=0.3)
axes[1].set_ylabel('phi [deg]'); axes[1].grid(True, alpha=0.3)

axes[2].plot(tms, rec['dc'], 'g-', lw=1, label='cmd')
axes[2].plot(tms, rec['da'], 'm-', lw=1, label='actual')
axes[2].set_ylabel('delta [deg]'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

pc={0:'red',1:'orange',2:'green',3:'blue'}
for j in range(len(tms)-1):
    axes[3].axvspan(tms[j],tms[j+1],alpha=0.3,color=pc.get(rec['ph'][j],'gray'))
axes[3].set_ylabel('Phase'); axes[3].set_xlabel('Time [ms]')
axes[3].set_yticks([0,1,2,3]); axes[3].set_yticklabels(['FOLD','W1','EXT','EW'])

plt.tight_layout()
plt.savefig('v15_fine_sweep_best.png', dpi=150)
print(f"\nSaved v15_fine_sweep_best.png")
print("Done!")

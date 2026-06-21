# -*- coding: utf-8 -*-
"""
R=0.15, phi=0 any-phase, beta_eff 예측 제어
beta_eff = beta + bdot * T_pred 를 일관되게 사용
T_pred도 스윕 파라미터로 추가
"""
import numpy as np, mujoco, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

M1=0.1; M2=0.25; L1=0.2; L2=0.2; BD=0.10; GRAV=9.81; DT=0.001
POLE_H=1.0; L_POLE=0.8; R_TARGET=0.15; B_THETA=0.01
Mt=M1+M2; HP=L_POLE/2; RH=HP-BD/2; RV=R_TARGET
lw2=L2/8; uw2=L2/8; bd2=BD/2
PR=max(0.005,L2*0.05); RR=max(0.003,L2*0.015); RM=0.01
KP=200; KD_s=10; TAU_MAX=10000
T_FOLD=0.020

xml = f"""<?xml version="1.0"?>
<mujoco model="v15_R015_pred">
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

def run(K1, T_w1, T_pred, beta0_deg=5, sim_time=6.0, max_cyc=30, record=False):
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(beta0_deg)
    mujoco.mj_forward(model, data)
    
    FOLD, WAIT1, EXTEND, EW = 0, 1, 2, 3
    md = np.radians(60)
    beta = compute_beta(model, data); bp = beta; bd = 0.0
    # First cycle: beta_eff with bdot=0
    beta_eff = beta  
    delta_target = np.clip(K1 * beta_eff, -md, md)
    delta_start = 0.0
    state = FOLD; t_ph = 0; cyc = 1
    phi_prev = data.joint('phi_Y').qpos[0]
    t_since_fold_end = 0
    bai = []
    rec = {'t':[], 'b':[], 'p':[], 'dc':[], 'da':[], 'ph':[], 'be':[]} if record else None
    
    N = int(sim_time / DT)
    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        beta = compute_beta(model, data)
        bd = (beta - bp) / DT if i > 0 else 0.0
        ha = data.joint('hip').qpos[0]
        t_ph += DT
        
        # beta_eff 계산 (매 스텝)
        beta_eff = beta + bd * T_pred
        
        if record and i % 3 == 0:
            rec['t'].append(data.time); rec['b'].append(np.degrees(beta))
            rec['p'].append(np.degrees(phi)); rec['dc'].append(np.degrees(delta_target))
            rec['da'].append(np.degrees(ha)); rec['ph'].append(state)
            rec['be'].append(np.degrees(beta_eff))
        
        # phi=0 crossing (not during FOLD, min time after fold)
        phi_crossed = False
        if state != FOLD and t_since_fold_end > 0.005:
            if phi_prev * phi < 0:
                phi_crossed = True
        
        def start_new_cycle():
            nonlocal state, t_ph, cyc, delta_start, delta_target, t_since_fold_end
            bai.append(np.degrees(beta_eff))  # beta_eff 기록
            cyc += 1
            delta_start = ha
            delta_target = np.clip(K1 * beta_eff, -md, md)
            state = FOLD; t_ph = 0; t_since_fold_end = 0
        
        if state == FOLD:
            frac = min(t_ph / T_FOLD, 1.0)
            ctrl = delta_start + frac * (delta_target - delta_start)
            if t_ph >= T_FOLD:
                state = WAIT1; t_ph = 0; t_since_fold_end = 0
        elif state == WAIT1:
            ctrl = delta_target
            t_since_fold_end += DT
            if phi_crossed:
                start_new_cycle()
                if cyc > max_cyc or abs(np.degrees(beta)) > 170: break
            elif t_ph >= T_w1:
                delta_start = delta_target; delta_target = 0.0
                state = EXTEND; t_ph = 0
        elif state == EXTEND:
            frac = min(t_ph / T_FOLD, 1.0)
            ctrl = delta_start + frac * (delta_target - delta_start)
            t_since_fold_end += DT
            if phi_crossed:
                start_new_cycle()
                if cyc > max_cyc or abs(np.degrees(beta)) > 170: break
            elif t_ph >= T_FOLD:
                state = EW; t_ph = 0
        elif state == EW:
            ctrl = 0.0
            t_since_fold_end += DT
            if phi_crossed:
                start_new_cycle()
                if cyc > max_cyc or abs(np.degrees(beta)) > 170: break
            if t_ph > 2: bai.append(np.degrees(beta_eff)); break
        
        phi_prev = phi
        data.ctrl[0] = ctrl; bp = beta
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)): break
    
    if record: return bai, rec
    return bai

# === Sweep: K1 x T_w1 x T_pred ===
K1_vals = np.arange(2.0, 10.1, 0.5)
Tw1_vals = np.arange(0.010, 0.101, 0.010)
Tpred_vals = [0.010, 0.020, 0.040, 0.060, 0.080, 0.100]

print(f"R={R_TARGET}m | Sweep: {len(K1_vals)} K1 x {len(Tw1_vals)} Tw1 x {len(Tpred_vals)} Tpred")
t0 = time.time()
results = []

for T_pred in Tpred_vals:
    for K1 in K1_vals:
        for tw in Tw1_vals:
            bai = run(K1=K1, T_w1=tw, T_pred=T_pred)
            if bai and len(bai) >= 4:
                ab = [abs(b) for b in bai]
                mx = max(ab[:min(10,len(ab))])
                n_good = sum(1 for a in ab if a < 15)
                # Check alternating sign (key indicator!)
                n_alt = 0
                for j in range(len(bai)-1):
                    if bai[j] * bai[j+1] < 0: n_alt += 1
                results.append({'K1':K1,'tw':tw,'tp':T_pred,'nc':len(bai),
                               'max':mx,'bai':bai,'n_good':n_good,'n_alt':n_alt})

elapsed = time.time()-t0
print(f"Done: {elapsed:.0f}s, {len(results)} valid\n")

# Sort by n_good, then n_alt (alternating = good), then max
results.sort(key=lambda x: (-x['n_good'], -x['n_alt'], x['max']))

# Show cases with alternating signs
alt_cases = [r for r in results if r['n_alt'] >= 3]
print(f"=== ALTERNATING SIGN (3+ alternations): {len(alt_cases)} ===")
for rank, r in enumerate(alt_cases[:25]):
    bai_s = ", ".join([f"{b:+.1f}" for b in r['bai'][:12]])
    print(f"{rank+1:3d} K1={r['K1']:4.1f} Tw1={r['tw']*1000:4.0f}ms Tp={r['tp']*1000:4.0f}ms "
          f"nc={r['nc']:2d} alt={r['n_alt']} g={r['n_good']:2d} mx={r['max']:5.1f}  [{bai_s}]")

print(f"\n=== TOP 25 overall ===")
for rank, r in enumerate(results[:25]):
    bai_s = ", ".join([f"{b:+.1f}" for b in r['bai'][:12]])
    print(f"{rank+1:3d} K1={r['K1']:4.1f} Tw1={r['tw']*1000:4.0f}ms Tp={r['tp']*1000:4.0f}ms "
          f"nc={r['nc']:2d} alt={r['n_alt']} g={r['n_good']:2d} mx={r['max']:5.1f}  [{bai_s}]")

# Plot best
plot_r = alt_cases[0] if alt_cases else results[0]
bai, rec = run(K1=plot_r['K1'], T_w1=plot_r['tw'], T_pred=plot_r['tp'],
               record=True, sim_time=8.0)

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
tms = [t*1000 for t in rec['t']]

axes[0].plot(tms, rec['b'], 'r-', lw=1.5, label='beta')
axes[0].plot(tms, rec['be'], 'r--', lw=1, alpha=0.5, label='beta_eff')
axes[0].axhline(0,color='k',ls='--',alpha=.3)
axes[0].set_ylabel('beta [deg]'); axes[0].legend(); axes[0].grid(True,alpha=.3)
axes[0].set_title(f"R={R_TARGET} | K1={plot_r['K1']} Tw1={plot_r['tw']*1000:.0f}ms "
                  f"Tp={plot_r['tp']*1000:.0f}ms | alt={plot_r['n_alt']} | "
                  f"betas={[f'{b:+.1f}' for b in bai[:10]]}")

axes[1].plot(tms, rec['p'], 'b-', lw=1.5)
axes[1].axhline(0,color='k',ls='--',alpha=.3)
axes[1].set_ylabel('phi [deg]'); axes[1].grid(True,alpha=.3)

axes[2].plot(tms, rec['dc'], 'g-', lw=1, label='cmd')
axes[2].plot(tms, rec['da'], 'm-', lw=1, label='actual')
axes[2].set_ylabel('delta [deg]'); axes[2].legend(); axes[2].grid(True,alpha=.3)

axes[3].plot(tms, rec['be'], 'orange', lw=1.5)
axes[3].axhline(0,color='k',ls='--',alpha=.3)
axes[3].set_ylabel('beta_eff [deg]'); axes[3].grid(True,alpha=.3)

pc={0:'red',1:'orange',2:'green',3:'blue'}
for j in range(len(tms)-1):
    axes[4].axvspan(tms[j],tms[j+1],alpha=.3,color=pc.get(rec['ph'][j],'gray'))
axes[4].set_ylabel('Phase'); axes[4].set_xlabel('Time [ms]')
axes[4].set_yticks([0,1,2,3]); axes[4].set_yticklabels(['FOLD','W1','EXT','EW'])

plt.tight_layout()
plt.savefig('v15_R015_betaeff.png', dpi=150)
print(f"\nSaved v15_R015_betaeff.png")
print("Done!")

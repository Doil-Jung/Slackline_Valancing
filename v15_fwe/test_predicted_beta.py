"""
Predicted-beta FWE:
  delta = K1 * beta_predicted
  beta_predicted = beta + beta_dot * T_eff
  T_eff = T_servo_travel + T_w1  (time from fold command to extend)

This reduces 3 params (K1,K2,Tw1) to 2 params (K1, Tw1).
If K2_equiv = K1 * T_eff, then with K1=4.0, K2=0.15:
  T_eff = 0.15/4.0 = 37.5ms
"""
import numpy as np, mujoco, os, time
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

T_SERVO = 0.009  # servo travel time ~9ms (from previous measurements)

def run_fwe(K1, T_w1, beta0_deg=5, sim_time=8.0, max_cyc=20, record=False):
    """
    delta = K1 * (beta + beta_dot * T_eff)
    T_eff = T_SERVO + T_w1
    """
    data=mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]]=np.radians(beta0_deg)
    mujoco.mj_forward(model,data)
    
    FOLD,WAIT1,EW=0,1,2
    md=np.radians(60);ath=np.radians(2)
    T_eff = T_SERVO + T_w1
    
    beta=compute_beta(model,data); bp=beta; bd=0.0
    # First cycle: beta_dot=0
    beta_pred = beta + bd * T_eff
    fd=np.clip(K1*beta_pred,-md,md); target=fd
    state=FOLD; t_ph=0; cyc=1; pp=0; bai=[]
    
    # Recording
    rec_t=[];rec_b=[];rec_p=[];rec_dc=[];rec_da=[];rec_ph=[]
    
    N=int(sim_time/DT)
    for i in range(N):
        phi=data.joint('phi_Y').qpos[0]
        beta=compute_beta(model,data)
        bd=(beta-bp)/DT; ha=data.joint('hip').qpos[0]; t_ph+=DT
        
        if record and i%5==0:
            rec_t.append(data.time)
            rec_b.append(np.degrees(beta))
            rec_p.append(np.degrees(phi))
            rec_dc.append(np.degrees(fd))
            rec_da.append(np.degrees(ha))
            rec_ph.append(state)
        
        if state==FOLD:
            target=fd
            if abs(ha-fd)<ath or t_ph>0.2: state=WAIT1;t_ph=0
        elif state==WAIT1:
            target=fd
            if t_ph>=T_w1: target=0;state=EW;t_ph=0;pp=phi
        elif state==EW:
            target=0
            if t_ph>0.005 and pp*phi<0:
                bai.append(np.degrees(beta));cyc+=1
                if cyc>max_cyc or abs(np.degrees(beta))>170: break
                # Next cycle: use predicted beta
                beta_pred = beta + bd * T_eff
                fd=np.clip(K1*beta_pred,-md,md)
                target=fd;state=FOLD;t_ph=0
            pp=phi
            if t_ph>3: bai.append(np.degrees(beta));break
        
        data.ctrl[0]=target; bp=beta
        mujoco.mj_step(model,data)
        if np.any(np.isnan(data.qpos)):break
    
    if record:
        return bai, rec_t, rec_b, rec_p, rec_dc, rec_da, rec_ph
    return bai

# === Verify the K2=K1*T_eff relationship ===
print("="*70)
print("Verifying: delta = K1*(beta + beta_dot*T_eff) vs K1*beta + K2*beta_dot")
print("="*70)
print(f"T_SERVO = {T_SERVO*1000:.0f}ms")
print()

# Compare: K1=4.0, K2=0.15 → T_eff = 0.15/4.0 = 37.5ms → T_w1 = 37.5 - 9 = 28.5ms
T_eff_implied = 0.15 / 4.0
Tw1_implied = T_eff_implied - T_SERVO
print(f"K1=4.0, K2=0.15 implies T_eff={T_eff_implied*1000:.1f}ms, T_w1={Tw1_implied*1000:.1f}ms")
print()

# Benchmark
t0=time.time()
bai = run_fwe(K1=4.0, T_w1=Tw1_implied)
dt_case=time.time()-t0
print(f"K1=4.0, T_w1={Tw1_implied*1000:.1f}ms: {[f'{b:+.1f}' for b in bai]}")
print(f"Time per case: {dt_case:.3f}s")

# Also test T_w1=20ms (what worked before)  
bai2 = run_fwe(K1=4.0, T_w1=0.02)
print(f"K1=4.0, T_w1=20ms:  {[f'{b:+.1f}' for b in bai2]}")

# === K1 x T_w1 sweep ===
K1_vals = np.arange(2.0, 8.1, 0.25)  # 25 values
Tw1_vals = np.arange(0.005, 0.151, 0.005)  # 30 values  
total = len(K1_vals) * len(Tw1_vals)
print(f"\nSweep: {len(K1_vals)} K1 x {len(Tw1_vals)} Tw1 = {total} cases")
print(f"Estimated time: {total * dt_case:.0f}s ({total * dt_case / 60:.1f} min)")

t_start = time.time()
results = []
for K1 in K1_vals:
    for tw in Tw1_vals:
        bai = run_fwe(K1=K1, T_w1=tw)
        if bai:
            mx = max(abs(b) for b in bai[:8])
            results.append((mx, len(bai), K1, tw, bai))

elapsed = time.time() - t_start
print(f"Done: {elapsed:.1f}s ({elapsed/60:.1f}min)\n")

results.sort()
print(f"{'#':>3} {'K1':>5} {'Tw1':>6} {'nc':>3} {'max':>6}  betas")
print("-"*85)
for rank,(score,nc,k1,tw,bai) in enumerate(results[:30]):
    bai_s=", ".join([f"{b:+.1f}" for b in bai[:8]])
    K2_equiv = k1 * (T_SERVO + tw)
    conv=""
    if nc>=4:
        ab=[abs(b) for b in bai]
        if all(a<10 for a in ab):conv="**"
        elif all(a<15 for a in ab[:5]):conv="*"
        elif all(a<30 for a in ab[:4]):conv="~"
    print(f"{rank+1:3d} {k1:5.2f} {tw*1000:5.0f}ms {nc:3d} {score:6.1f}  K2eq={K2_equiv:.3f}  [{bai_s}]  {conv}")

# Detailed categories
for lim_name, lim in [("10deg", 10), ("20deg", 20), ("30deg", 30)]:
    matches = [(s,nc,k1,tw,bai) for s,nc,k1,tw,bai in results 
               if nc>=4 and all(abs(b)<lim for b in bai[:min(5,nc)])]
    print(f"\n=== |beta|<{lim}deg for 4+ cycles: {len(matches)} ===")
    for s,nc,k1,tw,bai in matches[:10]:
        bai_s=", ".join([f"{b:+.1f}" for b in bai[:10]])
        K2_eq = k1*(T_SERVO+tw)
        print(f"  K1={k1:.2f} Tw1={tw*1000:.0f}ms (K2eq={K2_eq:.3f}): [{bai_s}]")

# Plot best
if results:
    best_s, best_nc, best_k1, best_tw, _ = results[0]
    print(f"\nPlotting best: K1={best_k1:.2f}, Tw1={best_tw*1000:.0f}ms")
    bai, rt, rb, rp, rdc, rda, rph = run_fwe(K1=best_k1, T_w1=best_tw, record=True)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    tms = [t*1000 for t in rt]
    
    axes[0].plot(tms, rb, 'r-', lw=1.5, label='beta')
    axes[0].axhline(0, color='k', ls='--', alpha=0.3)
    axes[0].set_ylabel('beta [deg]')
    K2_eq = best_k1*(T_SERVO+best_tw)
    axes[0].set_title(f"Predicted-beta FWE: K1={best_k1:.2f}, Tw1={best_tw*1000:.0f}ms (K2_equiv={K2_eq:.3f})")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(tms, rp, 'b-', lw=1.5, label='phi')
    axes[1].axhline(0, color='k', ls='--', alpha=0.3)
    axes[1].set_ylabel('phi [deg]')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(tms, rdc, 'g-', lw=1, label='delta cmd')
    axes[2].plot(tms, rda, 'm-', lw=1, label='delta actual')
    axes[2].set_ylabel('delta [deg]')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    phase_colors = {0: 'red', 1: 'orange', 2: 'blue'}
    for j in range(len(tms)-1):
        axes[3].axvspan(tms[j], tms[j+1], alpha=0.3, color=phase_colors.get(rph[j], 'gray'))
    axes[3].set_ylabel('Phase')
    axes[3].set_xlabel('Time [ms]')
    axes[3].set_yticks([0,1,2]); axes[3].set_yticklabels(['FOLD','WAIT1','EXT_W'])
    
    plt.tight_layout()
    plt.savefig('v15_predicted_beta.png', dpi=150)
    print("Saved v15_predicted_beta.png")

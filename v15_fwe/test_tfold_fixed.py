"""
Fixed T_fold=20ms:
  delta = K1 * (beta + beta_dot * T_fold)
  Servo ramps linearly over T_fold=20ms
  K1 fine sweep around 4, T_w1 coarse sweep
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

T_FOLD = 0.020  # 20ms fixed

def run_fwe(K1, T_w1, beta0_deg=5, sim_time=8.0, max_cyc=20, record=False):
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(beta0_deg)
    mujoco.mj_forward(model, data)
    
    FOLD, WAIT1, EXTEND, EW = 0, 1, 2, 3
    md = np.radians(60)
    
    beta = compute_beta(model, data); bp = beta; bd = 0.0
    beta_pred = beta  # first cycle beta_dot=0
    delta_target = np.clip(K1 * beta_pred, -md, md)
    delta_start = 0.0
    
    state = FOLD; t_ph = 0; cyc = 1; pp = 0; bai = []
    rec_t=[]; rec_b=[]; rec_p=[]; rec_dc=[]; rec_da=[]; rec_ph=[]
    
    N = int(sim_time / DT)
    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        beta = compute_beta(model, data)
        bd = (beta - bp) / DT if i > 0 else 0.0
        ha = data.joint('hip').qpos[0]
        t_ph += DT
        
        if record and i % 5 == 0:
            rec_t.append(data.time); rec_b.append(np.degrees(beta))
            rec_p.append(np.degrees(phi)); rec_dc.append(np.degrees(delta_target))
            rec_da.append(np.degrees(ha)); rec_ph.append(state)
        
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
                bai.append(np.degrees(beta)); cyc += 1
                if cyc > max_cyc or abs(np.degrees(beta)) > 170: break
                beta_pred = beta + bd * T_FOLD
                delta_start = 0.0
                delta_target = np.clip(K1 * beta_pred, -md, md)
                state = FOLD; t_ph = 0
            pp = phi
            if t_ph > 3: bai.append(np.degrees(beta)); break
        
        data.ctrl[0] = ctrl; bp = beta
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)): break
    
    if record:
        return bai, rec_t, rec_b, rec_p, rec_dc, rec_da, rec_ph
    return bai

# === Sweep ===
K1_vals = np.arange(4.0, 5.01, 0.05)   # fine: 4.0~5.0 step 0.05 (21 values)
Tw1_vals = np.arange(0.030, 0.101, 0.005)  # 30~100ms step 5ms (15 values)

total = len(K1_vals) * len(Tw1_vals)
t0 = time.time()
bai_test = run_fwe(4.0, 0.02)
dt_case = time.time() - t0

print(f"T_FOLD = {T_FOLD*1000:.0f}ms fixed")
print(f"delta = K1 * (beta + beta_dot * {T_FOLD*1000:.0f}ms)")
print(f"K1: {K1_vals[0]:.1f} ~ {K1_vals[-1]:.1f} step 0.1 ({len(K1_vals)})")
print(f"Tw1: {[f'{t*1000:.0f}ms' for t in Tw1_vals]} ({len(Tw1_vals)})")
print(f"Total: {total} cases, est {total*dt_case:.0f}s ({total*dt_case/60:.1f}min)")
print(f"\nK1=4.0 Tw1=20ms test: {[f'{b:+.1f}' for b in bai_test]}")

results = []
for K1 in K1_vals:
    for tw in Tw1_vals:
        bai = run_fwe(K1=K1, T_w1=tw)
        if bai:
            mx = max(abs(b) for b in bai[:8])
            results.append((mx, len(bai), round(K1,2), tw, bai))

elapsed = time.time() - t0
print(f"\nDone: {elapsed:.1f}s\n")

results.sort()
print(f"{'#':>3} {'K1':>5} {'Tw1':>6} {'nc':>3} {'max':>6}  betas")
print("-"*85)
for rank, (score, nc, k1, tw, bai) in enumerate(results[:30]):
    bai_s = ", ".join([f"{b:+.1f}" for b in bai[:8]])
    conv = ""
    if nc >= 4:
        ab = [abs(b) for b in bai]
        if all(a < 10 for a in ab): conv = "**"
        elif all(a < 15 for a in ab[:5]): conv = "*"
        elif all(a < 30 for a in ab[:4]): conv = "~"
    print(f"{rank+1:3d} {k1:5.2f} {tw*1000:5.0f}ms {nc:3d} {score:6.1f}  [{bai_s}]  {conv}")

for lim in [10, 20, 30]:
    matches = [(s,nc,k1,tw,bai) for s,nc,k1,tw,bai in results 
               if nc >= 4 and all(abs(b) < lim for b in bai[:min(5, nc)])]
    print(f"\n=== |beta|<{lim}deg, 4+ cycles: {len(matches)} ===")
    for s, nc, k1, tw, bai in matches[:15]:
        bai_s = ", ".join([f"{b:+.1f}" for b in bai[:10]])
        print(f"  K1={k1:.2f} Tw1={tw*1000:.0f}ms: [{bai_s}]")

# Plot best
if results:
    _, _, bk1, btw, _ = results[0]
    bai, rt, rb, rp, rdc, rda, rph = run_fwe(bk1, btw, record=True)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    tms = [t*1000 for t in rt]
    axes[0].plot(tms, rb, 'r-', lw=1.5); axes[0].axhline(0,color='k',ls='--',alpha=.3)
    axes[0].set_ylabel('beta [deg]'); axes[0].set_title(f"K1={bk1:.2f} Tw1={btw*1000:.0f}ms  T_fold=20ms")
    axes[0].grid(True,alpha=.3)
    axes[1].plot(tms, rp, 'b-', lw=1.5); axes[1].axhline(0,color='k',ls='--',alpha=.3)
    axes[1].set_ylabel('phi [deg]'); axes[1].grid(True,alpha=.3)
    axes[2].plot(tms, rdc, 'g-', lw=1, label='cmd'); axes[2].plot(tms, rda, 'm-', lw=1, label='actual')
    axes[2].set_ylabel('delta [deg]'); axes[2].legend(); axes[2].grid(True,alpha=.3)
    pc={0:'red',1:'orange',2:'green',3:'blue'}
    for j in range(len(tms)-1): axes[3].axvspan(tms[j],tms[j+1],alpha=.3,color=pc.get(rph[j],'gray'))
    axes[3].set_ylabel('Phase'); axes[3].set_xlabel('Time [ms]')
    axes[3].set_yticks([0,1,2,3]); axes[3].set_yticklabels(['FOLD','W1','EXT','EW'])
    plt.tight_layout(); plt.savefig('v15_tfold_fixed.png', dpi=150)
    print("Saved v15_tfold_fixed.png")

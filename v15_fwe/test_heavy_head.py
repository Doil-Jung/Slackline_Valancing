# -*- coding: utf-8 -*-
"""
머리 질량 효과 테스트: M_HEAD를 0.1~5kg으로 변화시키며 lambda 변화 확인
그리고 가장 좋은 조건에서 FWE 수렴 검증
"""
import numpy as np, mujoco, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRAV=9.81; DT=0.001
M1=0.1; M2=0.25; L1=0.2; L2=0.2; BD=0.10
POLE_H=1.0; L_POLE=0.8; B_THETA=0.01
lw2=L2/8; uw2=L2/8; bd2=BD/2
PR=max(0.005,L2*0.05); RR=max(0.003,L2*0.015); RM=0.01
HEAD_R=0.02; T_FOLD=0.020

def calc_params(R, M_HEAD):
    Mt=M1+M2+M_HEAD
    LC1=L1/2; LC2=L2/2
    p1=M1*LC1+M2*L1+M_HEAD*L1; p2=M2*LC2+M_HEAD*L2; ps=p1+p2
    h_com=(M1*LC1+M2*(L1+LC2)+M_HEAD*(L1+L2))/Mt
    c_foot=p2/ps
    J_aa=M1*LC1**2+M1*L1**2/12+M2*L1**2+M_HEAD*L1**2
    J_tt=M2*LC2**2+M2*L2**2/12+M_HEAD*L2**2
    J_at=M2*L1*LC2+M_HEAD*L1*L2
    J_tot=J_aa+J_tt+2*J_at
    C_sd=(-J_aa*p2+J_tt*p1+J_at*(p1-p2))/ps**2
    M11=Mt*R**2; M12=R; M22=J_tot/ps**2
    det_M=M11*M22-M12**2
    iM11=M22/det_M; iM12=-M12/det_M; iM22=M11/det_M
    gMR=GRAV*Mt*R; g_ps=GRAV/ps
    w=np.sqrt(abs(iM11*gMR)); la=np.sqrt(abs(iM22*g_ps))
    Tq=np.pi/(2*w)
    return {'Mt':Mt,'h':h_com,'c_foot':c_foot,'w':w,'la':la,'Tq':Tq,'la_Tq':la*Tq,'la_w':la/w}

# === Part 1: Lambda vs M_HEAD for various R ===
print("=" * 70)
print("Part 1: lambda vs M_HEAD (R=0.3, 0.6, 1.0)")
print("=" * 70)

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
for ax, R in zip(axes1, [0.3, 0.6, 1.0]):
    mh_vals = np.arange(0, 5.1, 0.1)
    ws, las, ratios = [], [], []
    for mh in mh_vals:
        p = calc_params(R, mh)
        ws.append(p['w']); las.append(p['la']); ratios.append(p['la_w'])
    
    ax.plot(mh_vals, las, 'r-', lw=2, label='lambda')
    ax.plot(mh_vals, ws, 'b-', lw=2, label='omega')
    ax.axhline(np.sqrt(GRAV/0.4), color='gray', ls='--', alpha=0.5, label='sqrt(g/0.4)=4.95')
    ax.set_xlabel('M_HEAD [kg]')
    ax.set_ylabel('rad/s')
    ax.set_title(f'R={R}m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 20)

plt.suptitle('omega, lambda vs M_HEAD', fontsize=14)
plt.tight_layout()
plt.savefig('v15_lambda_vs_mhead.png', dpi=150)
print("Saved v15_lambda_vs_mhead.png")

# Print table
print(f"\n{'M_HEAD':>7} | {'R=0.3':^25} | {'R=0.6':^25} | {'R=1.0':^25}")
print(f"{'':>7} | {'w':>6} {'la':>6} {'la/w':>6} {'la*Tq':>6} | "
      f"{'w':>6} {'la':>6} {'la/w':>6} {'la*Tq':>6} | "
      f"{'w':>6} {'la':>6} {'la/w':>6} {'la*Tq':>6}")
print("-" * 100)
for mh in [0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
    row = f"{mh:7.1f} |"
    for R in [0.3, 0.6, 1.0]:
        p = calc_params(R, mh)
        row += f" {p['w']:6.2f} {p['la']:6.2f} {p['la_w']:6.2f} {p['la_Tq']:6.2f} |"
    print(row)

# === Part 2: Find best config and run MuJoCo sweep ===
# Pick a promising config: R=0.6, M_HEAD=2.0 (la/w should be ~1 or less)
R_TEST = 0.6
MH_TEST = 0.5
p = calc_params(R_TEST, MH_TEST)
print(f"\n{'='*70}")
print(f"Part 2: MuJoCo sweep with R={R_TEST}, M_HEAD={MH_TEST}")
print(f"  w={p['w']:.3f}  la={p['la']:.3f}  la/w={p['la_w']:.3f}  la*Tq={p['la_Tq']:.3f}")
print(f"{'='*70}")

Mt_test = M1+M2+MH_TEST
HP=L_POLE/2; RH=HP-BD/2; RV=R_TEST
KP=200; KD_s=10; TAU_MAX=10000

xml = f"""<?xml version="1.0"?>
<mujoco model="v15_heavy_head">
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
          <geom name="head" type="sphere" size="0.03" pos="0 0 {L2}"
                mass="{MH_TEST}" rgba="1 0.3 0.3 0.9"/>
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
    com=(M1*cl+(M2+MH_TEST)*cu)/Mt_test
    return np.arctan2(com[0]-foot[0],com[2]-foot[2])

def run_fwe(K1, T_w1, beta0_deg=5, sim_time=10.0, max_cyc=25, record=False):
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(beta0_deg)
    mujoco.mj_forward(model, data)
    
    FOLD, WAIT1, EXTEND, EW = 0, 1, 2, 3
    md = np.radians(60)
    beta = compute_beta(model, data); bp = beta; bd = 0.0
    delta_target = np.clip(K1 * beta, -md, md)
    delta_start = 0.0
    state = FOLD; t_ph = 0; cyc = 1; pp = 0
    bai = []
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
                bai.append(np.degrees(beta))
                cyc += 1
                if cyc > max_cyc or abs(np.degrees(beta)) > 170: break
                beta_eff = beta + bd * T_FOLD
                delta_start = 0.0
                delta_target = np.clip(K1 * beta_eff, -md, md)
                state = FOLD; t_ph = 0
            pp = phi
            if t_ph > 3: bai.append(np.degrees(beta)); break
        
        data.ctrl[0] = ctrl; bp = beta
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)): break
    
    if record: return bai, rec
    return bai

# Sweep
K1_vals = np.arange(1.0, 12.1, 0.5)
Tw1_vals = np.arange(0.020, 0.401, 0.020)

t0 = time.time()
results = []
for K1 in K1_vals:
    for tw in Tw1_vals:
        bai = run_fwe(K1=K1, T_w1=tw)
        if bai and len(bai) >= 3:
            ab = [abs(b) for b in bai]
            mx = max(ab[:min(8,len(ab))])
            bounded = all(a < 20 for a in ab[:min(6,len(ab))])
            n_good = sum(1 for a in ab if a < 20)
            results.append({'K1':K1,'tw':tw,'nc':len(bai),'max':mx,'bai':bai,
                           'bounded':bounded,'n_good':n_good})

elapsed = time.time()-t0
print(f"Done: {elapsed:.0f}s, {len(results)} valid\n")

results.sort(key=lambda x: (-x['n_good'], x['max']))

bounded = [r for r in results if r['bounded']]
print(f"=== BOUNDED (|beta|<20d, 6+ cyc): {len(bounded)} ===")
for rank, r in enumerate(bounded[:25]):
    bai_s = ", ".join([f"{b:+.1f}" for b in r['bai'][:12]])
    print(f"{rank+1:3d} K1={r['K1']:5.1f} Tw1={r['tw']*1000:5.0f}ms nc={r['nc']:2d} "
          f"max={r['max']:5.1f}  [{bai_s}]")

print(f"\n=== ALL top 20 (by n_good then max) ===")
for rank, r in enumerate(results[:20]):
    bai_s = ", ".join([f"{b:+.1f}" for b in r['bai'][:12]])
    tag = "B" if r['bounded'] else ""
    print(f"{rank+1:3d} K1={r['K1']:5.1f} Tw1={r['tw']*1000:5.0f}ms nc={r['nc']:2d} "
          f"max={r['max']:5.1f} g={r['n_good']:2d} {tag:>2}  [{bai_s}]")

# Plot best
plot_r = bounded[0] if bounded else results[0]
bai, rec = run_fwe(K1=plot_r['K1'], T_w1=plot_r['tw'], record=True, sim_time=12.0)

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
tms = [t*1000 for t in rec['t']]
axes[0].plot(tms, rec['b'], 'r-', lw=1.5); axes[0].axhline(0,color='k',ls='--',alpha=.3)
axes[0].set_ylabel('beta [deg]')
axes[0].set_title(f"R={R_TEST} M_HEAD={MH_TEST}kg | K1={plot_r['K1']} Tw1={plot_r['tw']*1000:.0f}ms | "
                  f"w={p['w']:.2f} la={p['la']:.2f} la/w={p['la_w']:.2f}")
axes[0].grid(True, alpha=.3)
axes[1].plot(tms, rec['p'], 'b-', lw=1.5); axes[1].axhline(0,color='k',ls='--',alpha=.3)
axes[1].set_ylabel('phi [deg]'); axes[1].grid(True, alpha=.3)
axes[2].plot(tms, rec['dc'], 'g-', lw=1, label='cmd')
axes[2].plot(tms, rec['da'], 'm-', lw=1, label='actual')
axes[2].set_ylabel('delta [deg]'); axes[2].legend(); axes[2].grid(True, alpha=.3)
pc={0:'red',1:'orange',2:'green',3:'blue'}
for j in range(len(tms)-1):
    axes[3].axvspan(tms[j],tms[j+1],alpha=.3,color=pc.get(rec['ph'][j],'gray'))
axes[3].set_ylabel('Phase'); axes[3].set_xlabel('Time [ms]')
axes[3].set_yticks([0,1,2,3]); axes[3].set_yticklabels(['FOLD','W1','EXT','EW'])
plt.tight_layout()
plt.savefig('v15_heavy_head_best.png', dpi=150)
print(f"\nSaved v15_heavy_head_best.png")
print("Done!")

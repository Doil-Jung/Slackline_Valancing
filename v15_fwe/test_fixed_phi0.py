"""
Fix: detect phi=0 crossing DURING extend, not after.
Merge EXTEND+WAIT2: as soon as we command delta=0, start watching phi=0.

Then re-run K1 sweep with this fix.
"""
import numpy as np, mujoco, os, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

M1=0.1;M2=0.25;L1=0.2;L2=0.2;BD=0.10;GRAV=9.81;DT=0.001
POLE_H=1.0;L_POLE=0.8;R_TARGET=0.3;B_THETA=0.01
Mt=M1+M2
HP=L_POLE/2;RH=HP-BD/2;RV=R_TARGET
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


def run_fwe_fixed(K1, K2, T_w1, beta0_deg=5, sim_time=5.0, max_cyc=15, verbose=False):
    """
    Fixed state machine: 3 states only (FOLD → WAIT1 → EXTEND_WAIT)
    EXTEND_WAIT: command delta=0, AND watch phi=0 simultaneously.
    No separate WAIT2 — phi=0 detection starts at EXTEND command.
    """
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[model.joint('ankle').id]] = np.radians(beta0_deg)
    mujoco.mj_forward(model, data)

    # States: 0=IDLE, 1=FOLD, 2=WAIT1, 3=EXTEND_WAIT (merged EXTEND+WAIT2)
    IDLE, FOLD, WAIT1, EXTEND_WAIT = 0, 1, 2, 3
    state = IDLE; t_ph = 0; target = 0; fd = 0; cyc = 0
    phi_prev = 0; bp = np.radians(beta0_deg)
    ath = np.radians(2); md = np.radians(60)
    beta_at_idle = []
    
    N = int(sim_time / DT)
    ts=[]; betas=[]; phis=[]; dcmd=[]; dact=[]; sts=[]; phi_dots=[]

    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        phi_dot = data.joint('phi_Y').qvel[0]
        beta = compute_beta(model, data)
        bd = (beta - bp) / DT
        ha = data.joint('hip').qpos[0]
        t_ph += DT

        if state == IDLE:
            target = 0
            if abs(beta) > np.radians(0.5) and cyc < max_cyc:
                fd = np.clip(K1 * beta + K2 * bd, -md, md)
                target = fd; state = FOLD; t_ph = 0; cyc += 1
                phi_prev = phi  # record phi at fold start
                if verbose:
                    print(f"  [{cyc}] FOLD: beta={np.degrees(beta):+.1f}° "
                          f"delta={np.degrees(fd):+.1f}° phi={np.degrees(phi):+.1f}°")

        elif state == FOLD:
            target = fd
            if abs(ha - fd) < ath or t_ph > 0.2:
                state = WAIT1; t_ph = 0
                if verbose:
                    print(f"      WAIT1: hip={np.degrees(ha):.1f}° beta={np.degrees(beta):+.1f}°")

        elif state == WAIT1:
            target = fd
            if t_ph >= T_w1:
                # EXTEND + start watching phi=0 immediately
                target = 0; state = EXTEND_WAIT; t_ph = 0
                phi_prev = phi  # record phi at extend start for zero-cross detection
                if verbose:
                    print(f"      EXTEND_WAIT: beta={np.degrees(beta):+.1f}° "
                          f"phi={np.degrees(phi):+.1f}°")

        elif state == EXTEND_WAIT:
            target = 0
            # Watch for phi=0 crossing from the moment we start extending
            if t_ph > 0.005 and phi_prev * phi < 0:  # small deadtime to avoid immediate trigger
                beta_at_idle.append(np.degrees(beta))
                if verbose:
                    print(f"      phi=0! beta={np.degrees(beta):+.1f}° "
                          f"phi={np.degrees(phi):+.1f}° t_ew={t_ph*1000:.0f}ms")
                state = IDLE; t_ph = 0
            phi_prev = phi
            if t_ph > 3: state = IDLE; t_ph = 0

        data.ctrl[0] = target
        ts.append(data.time * 1000)
        betas.append(np.degrees(beta))
        phis.append(np.degrees(phi))
        phi_dots.append(np.degrees(phi_dot))
        dcmd.append(np.degrees(target))
        dact.append(np.degrees(ha))
        sts.append(state)
        bp = beta
        mujoco.mj_step(model, data)
        if np.any(np.isnan(data.qpos)): break

    return (np.array(ts), np.array(betas), np.array(phis), np.array(dcmd),
            np.array(dact), np.array(sts), beta_at_idle, np.array(phi_dots))


# ======= Test 1: Detailed single run =======
print("=== Fixed state machine: K1=4.1, K2=0, Tw1=100ms ===")
ts,betas,phis,dcmd,dact,sts,bai,phi_dots = run_fwe_fixed(4.1, 0, 0.10, 5, 3.0, verbose=True)
print(f"\nbeta_at_idle: {[f'{b:+.1f}' for b in bai]}")


# ======= Test 2: Full 3-param sweep with fixed state machine =======
print("\n\n=== Full sweep with fixed state machine ===")
K1_vals = [2, 3, 4, 5, 6, 8, 10, 15, 20]
K2_vals = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
Tw_vals = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]

total = len(K1_vals) * len(K2_vals) * len(Tw_vals)
print(f"Grid: {len(K1_vals)}x{len(K2_vals)}x{len(Tw_vals)} = {total} cases")

t0 = time.time()
results = []
for K1 in K1_vals:
    for K2 in K2_vals:
        for tw in Tw_vals:
            _,_,_,_,_,_,bai,_ = run_fwe_fixed(K1, K2, tw, 5, 5.0)
            mx = max(abs(b) for b in bai[:5]) if bai else 999
            results.append((mx, len(bai), K1, K2, tw, bai))

elapsed = time.time() - t0
print(f"Done: {total} cases in {elapsed:.1f}s")

# TOP 25
results.sort()
print(f"\n{'#':>3} {'K1':>5} {'K2':>5} {'Tw1':>6}  {'nc':>3} {'max|β|':>7}  betas")
print("-"*85)
for rank,(score,nc,k1,k2,tw,bai) in enumerate(results[:25]):
    bai_s=", ".join([f"{b:+.1f}" for b in bai[:6]])
    conv = ""
    if nc >= 3:
        ab = [abs(b) for b in bai]
        if all(a < 15 for a in ab): conv = "★"
        elif all(a < 30 for a in ab[:4]): conv = "~"
    print(f"{rank+1:3d} {k1:5.1f} {k2:5.2f} {tw*1000:5.0f}ms  {nc:3d}  {score:7.1f}°  [{bai_s}]  {conv}")

# Convergent check
print("\n=== CONVERGENT (all |beta|<15°, 3+ cycles) ===")
found = False
for score,nc,k1,k2,tw,bai in results:
    if nc >= 3 and all(abs(b) < 15 for b in bai):
        bai_s = ", ".join([f"{b:+.1f}" for b in bai[:8]])
        print(f"  K1={k1} K2={k2} Tw1={tw*1000:.0f}ms: [{bai_s}]")
        found = True
if not found:
    print("  (none)")
    print("\n=== BOUNDED (all |beta|<30°, 3+ cycles) ===")
    for score,nc,k1,k2,tw,bai in results:
        if nc >= 3 and all(abs(b) < 30 for b in bai[:4]):
            bai_s = ", ".join([f"{b:+.1f}" for b in bai[:8]])
            print(f"  K1={k1} K2={k2} Tw1={tw*1000:.0f}ms: [{bai_s}]")


# ======= Plot best case =======
if results:
    _,_,bk1,bk2,btw,_ = results[0]
    ts,betas,phis,dcmd,dact,sts,bai,phi_dots = run_fwe_fixed(bk1, bk2, btw, 5, 5.0, verbose=False)
    
    fig,axes=plt.subplots(4,1,figsize=(18,14),sharex=True)
    colors={0:'white',1:'#FFD700',2:'#90EE90',3:'#87CEEB'}
    clabels={0:'IDLE',1:'FOLD',2:'WAIT1',3:'EXT+W2'}
    for r in range(4):
        prev=sts[0];start=ts[0]
        for j in range(1,len(sts)):
            if sts[j]!=prev or j==len(sts)-1:
                axes[r].axvspan(start,ts[j],alpha=0.15,color=colors.get(prev,'w'))
                prev=sts[j];start=ts[j]

    axes[0].plot(ts,betas,'r-',lw=1.5)
    axes[0].axhline(0,color='black',lw=0.5)
    axes[0].set_ylabel('beta [°]')
    axes[0].set_title(f'Best: K1={bk1} K2={bk2} Tw1={btw*1000:.0f}ms (fixed phi=0 detection)')

    axes[1].plot(ts,phis,'b-',lw=1.5)
    axes[1].axhline(0,color='red',ls='--',lw=0.8)
    axes[1].set_ylabel('phi [°]')

    axes[2].plot(ts,phi_dots,'m-',lw=1)
    axes[2].axhline(0,color='black',lw=0.5)
    axes[2].set_ylabel('phi_dot [°/s]')

    axes[3].plot(ts,dcmd,'g-',lw=1.5,label='cmd')
    axes[3].plot(ts,dact,'k--',lw=0.8,label='act')
    axes[3].set_ylabel('delta [°]');axes[3].set_xlabel('ms')
    axes[3].legend()

    plt.tight_layout()
    out=os.path.join(os.path.dirname(os.path.abspath(__file__)),'v15_fixed_phi0.png')
    plt.savefig(out,dpi=150)
    print(f"\nSaved: {out}")

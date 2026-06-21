"""
V15 FWE — Natural servo response test
=====================================
No triangle profile. Just set target angle, let PD servo with
force/speed limits move naturally. Much more realistic.
"""
import os, sys, numpy as np, mujoco
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Robot params
# ============================================================
M1=0.1; M2=0.25; L1=0.2; L2=0.2; BD=0.10; GRAV=9.81; DT=0.001
POLE_H=1.0; L_POLE=0.8; R_TARGET=0.3; B_THETA=0.01
LC1=L1/2; LC2=L2/2; Mt=M1+M2
p1=M1*LC1+M2*L1; p2=M2*LC2; ps=p1+p2
h_com=(M1*LC1+M2*(L1+LC2))/Mt; c_foot=p2/ps

HP=L_POLE/2; RH=HP-BD/2; RV=R_TARGET; FOOT_Z=POLE_H-RV
lw2=L2/8; uw2=L2/8; bd2=BD/2
PR=max(0.005,L2*0.05); RR=max(0.003,L2*0.015)
RM=0.01

# Servo parameters (emulating real RC servo)
TAU_MAX = 2.0      # max servo torque [Nm]
KP_SERVO = 80.0    # position gain
KD_SERVO = 0.3     # velocity damping → v_max ≈ TAU_MAX/KD ≈ 6.7 rad/s ≈ 380°/s

print(f"Servo: tau_max={TAU_MAX}Nm kp={KP_SERVO} kd={KD_SERVO}")
print(f"  v_max ≈ {np.degrees(TAU_MAX/KD_SERVO):.0f}°/s")
print(f"  h_com={h_com:.4f} c_foot={c_foot:.4f}")

# ============================================================
# MJCF — with force-limited position servo
# ============================================================
xml = f"""<?xml version="1.0"?>
<mujoco model="v15_fwe">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0.001"/>
  </default>
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
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}"/>
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
             gainprm="{KP_SERVO} 0 0"
             biasprm="0 {-KP_SERVO} {-KD_SERVO}"
             ctrlrange="-1.57 1.57" ctrllimited="true"
             forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/>
  </actuator>
</mujoco>"""

# ============================================================
# Simple FWE state machine (no triangle profile)
# ============================================================
class SimpleFWE:
    """Just set target angle and let servo do its thing."""
    IDLE = 0; FOLD = 1; WAIT1 = 2; EXTEND = 3; WAIT2 = 4
    NAMES = ['IDLE','FOLD','WAIT1','EXTEND','WAIT2']

    def __init__(self, eps, T_w1, max_delta=np.radians(80)):
        self.eps = eps
        self.T_w1 = T_w1
        self.max_delta = max_delta
        self.state = self.IDLE
        self.t_phase = 0.0
        self.d_fold = 0.0
        self.target = 0.0
        self.fold_sign = 1.0
        self.phi_prev = 0.0
        self.cycle = 0
        self.log = []
        self.arrive_thresh = np.radians(3)  # "close enough" to target

    def compute_beta(self, alpha, theta):
        return (p1*alpha + p2*theta) / (Mt*h_com)

    def step(self, dt, phi, alpha, theta, hip_actual):
        beta = self.compute_beta(alpha, theta)
        self.t_phase += dt

        if self.state == self.IDLE:
            self.target = 0.0
            if abs(beta) > np.radians(2.0):
                self.fold_sign = -np.sign(beta)  # fold opposite to tilt
                self.d_fold = h_com*(1+self.eps)*abs(beta)/c_foot
                self.d_fold = min(self.d_fold, self.max_delta)
                self.target = self.fold_sign * self.d_fold
                self.state = self.FOLD
                self.t_phase = 0.0
                self.cycle += 1
                self._log(f"FOLD: beta={np.degrees(beta):.1f}d "
                          f"target={np.degrees(self.target):.0f}d")

        elif self.state == self.FOLD:
            self.target = self.fold_sign * self.d_fold
            # Transition when servo arrives (or timeout)
            err = abs(hip_actual - self.target)
            if err < self.arrive_thresh or self.t_phase > 0.5:
                self.state = self.WAIT1
                self.t_phase = 0.0
                self._log(f"WAIT1: arrived (err={np.degrees(err):.1f}d "
                          f"T_fold={self.t_phase*1000:.0f}ms)")

        elif self.state == self.WAIT1:
            self.target = self.fold_sign * self.d_fold  # hold
            if self.t_phase >= self.T_w1:
                self.state = self.EXTEND
                self.t_phase = 0.0
                self._log("EXTEND")

        elif self.state == self.EXTEND:
            self.target = 0.0  # go back to straight
            err = abs(hip_actual)
            if err < self.arrive_thresh or self.t_phase > 0.5:
                self.state = self.WAIT2
                self.t_phase = 0.0
                self.phi_prev = phi
                self._log(f"WAIT2: phi={np.degrees(phi):.1f}d")

        elif self.state == self.WAIT2:
            self.target = 0.0
            if self.phi_prev * phi < 0:
                self._log(f"IDLE: phi=0! beta={np.degrees(beta):.1f}d "
                          f"T_w2={self.t_phase*1000:.0f}ms")
                self.state = self.IDLE
                self.t_phase = 0.0
            self.phi_prev = phi
            if self.t_phase > 3.0:
                self._log(f"IDLE: timeout phi={np.degrees(phi):.1f}d")
                self.state = self.IDLE
                self.t_phase = 0.0

        return self.target

    def _log(self, msg):
        self.log.append(msg)
        print(f"  [#{self.cycle}] {msg}")


# ============================================================
# Run tests with different eps values
# ============================================================
def run_sim(eps, T_w1, beta0_deg=5, sim_time=3.0, max_cycles=3):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    hip_q = model.jnt_qposadr[model.joint('hip').id]
    ank_q = model.jnt_qposadr[model.joint('ankle').id]

    a0 = np.radians(beta0_deg)
    data.qpos[ank_q] = a0; data.qpos[hip_q] = 0
    mujoco.mj_forward(model, data)

    fwe = SimpleFWE(eps=eps, T_w1=T_w1, max_delta=np.radians(80))

    N = int(sim_time/DT)
    ts=np.zeros(N); phis=np.zeros(N); betas=np.zeros(N)
    dcmd=np.zeros(N); dact=np.zeros(N); states=np.zeros(N,dtype=int)
    taus=np.zeros(N)

    for i in range(N):
        phi = data.joint('phi_Y').qpos[0]
        lb=model.body('lower_body').id; xm=data.xmat[lb].reshape(3,3)
        alpha=np.arctan2(-xm[2,0],xm[2,2])
        ub=model.body('upper_body').id; xmu=data.xmat[ub].reshape(3,3)
        theta=np.arctan2(-xmu[2,0],xmu[2,2])
        hip_act = data.joint('hip').qpos[0]
        beta = (p1*alpha+p2*theta)/(Mt*h_com)

        if fwe.cycle >= max_cycles and fwe.state == SimpleFWE.IDLE:
            target = 0.0
        else:
            target = fwe.step(DT, phi, alpha, theta, hip_act)
        data.ctrl[0] = target

        ts[i]=data.time; phis[i]=phi; betas[i]=beta
        dcmd[i]=target; dact[i]=hip_act; states[i]=fwe.state
        taus[i]=data.actuator_force[0] if model.nu > 0 else 0

        mujoco.mj_step(model, data)

        if np.any(np.isnan(data.qpos)):
            print(f"  NaN at t={data.time:.4f}s, stopping")
            ts=ts[:i]; phis=phis[:i]; betas=betas[:i]
            dcmd=dcmd[:i]; dact=dact[:i]; states=states[:i]; taus=taus[:i]
            break

    return ts, phis, betas, dcmd, dact, states, taus, fwe


# Test multiple eps
results = {}
for eps in [4, 8, 12]:
    print(f"\n{'='*50}")
    print(f"eps={eps}, T_w1=260ms, beta0=5d")
    print(f"  d_fold(5d) = {np.degrees(h_com*(1+eps)*np.radians(5)/c_foot):.0f}d")
    ts, phis, betas, dcmd, dact, states, taus, fwe = run_sim(
        eps=eps, T_w1=0.26, beta0_deg=5, sim_time=3.0, max_cycles=5)
    results[eps] = (ts, phis, betas, dcmd, dact, states, taus, fwe)

# Plot all
fig, axes = plt.subplots(4, len(results), figsize=(6*len(results), 14),
                         squeeze=False, sharex='col')

for col, (eps, (ts, phis, betas, dcmd, dact, states, taus, fwe)) in enumerate(results.items()):
    ax_phi, ax_beta, ax_delta, ax_tau = axes[0,col], axes[1,col], axes[2,col], axes[3,col]

    # Phase shading
    colors = {0:'white',1:'#FFD700',2:'#90EE90',3:'#87CEEB',4:'#DDA0DD'}
    for ax in [ax_phi, ax_beta, ax_delta, ax_tau]:
        prev=states[0]; start=ts[0]
        for i in range(1,len(states)):
            if states[i]!=prev or i==len(states)-1:
                ax.axvspan(start*1000,ts[i]*1000,alpha=0.12,color=colors.get(prev,'white'))
                prev=states[i]; start=ts[i]

    ax_phi.plot(ts*1000, np.degrees(phis), 'b-', lw=1.5)
    ax_phi.axhline(0, color='red', ls='--', alpha=0.5)
    ax_phi.set_ylabel('phi [d]')
    ax_phi.set_title(f'eps={eps} | d_fold={np.degrees(h_com*(1+eps)*np.radians(5)/c_foot):.0f}d')

    ax_beta.plot(ts*1000, np.degrees(betas), 'r-', lw=1.5)
    ax_beta.axhline(0, color='black', lw=0.5)
    ax_beta.set_ylabel('beta [d]')

    ax_delta.plot(ts*1000, np.degrees(dcmd), 'g-', lw=1.5, label='cmd')
    ax_delta.plot(ts*1000, np.degrees(dact), 'k--', lw=1, alpha=0.5, label='act')
    ax_delta.set_ylabel('delta [d]')
    ax_delta.legend(fontsize=8)

    ax_tau.plot(ts*1000, taus, 'm-', lw=1)
    ax_tau.axhline(TAU_MAX, color='gray', ls=':', alpha=0.5)
    ax_tau.axhline(-TAU_MAX, color='gray', ls=':', alpha=0.5)
    ax_tau.set_ylabel('tau [Nm]')
    ax_tau.set_xlabel('ms')

plt.suptitle(f'V15 FWE Natural Servo (tau_max={TAU_MAX}Nm, v_max≈{np.degrees(TAU_MAX/KD_SERVO):.0f}°/s)',
             fontsize=13, y=1.01)
plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v15_natural_servo.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")

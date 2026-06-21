# -*- coding: utf-8 -*-
"""
ODE 정확 조건 + 고강성 서보 (KP를 키워 운동학적 강제에 접근)
==============================================================
v4 결과: KP=200 으로는 hip 이 52° → 143° 폭주 (서보 너무 약함)

v5: KP 를 시스템이 허용하는 한 키워서 ODE 의 운동학적 강제에 접근
  - KP=5000 (25배 증가)
  - KD=100 (임계감쇠 근처)
  - DT=0.0001 (10배 작게, 수치 안정성)
  - 추가: 서보 진단 자세히 (cmd, actual, err, torque)
  - 측정: beta 와 delta_balance 모두
"""
import numpy as np
import mujoco
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# 로봇 파라미터
g = 9.81
M1 = 0.1; M2 = 0.25
L1 = 0.2; L2 = 0.2
BD = 0.10
R_TARGET = 0.3
GRAV = 9.81

# 서보를 강하게 + DT 작게
KP = float(sys.argv[1]) if len(sys.argv) > 1 else 5000.0
KD_S = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
DT = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0001
TAU_MAX = 10000.0
B_THETA = 0.01

LC1 = L1/2; LC2 = L2/2
Mt = M1 + M2
I1 = M1*L1**2/12; I2 = M2*L2**2/12
p1 = M1*LC1 + M2*L1; p2 = M2*LC2; ps = p1 + p2
h_com = (M1*LC1 + M2*(L1+LC2)) / Mt
c_foot = p2 / ps

J_aa = M1*LC1**2 + I1 + M2*L1**2
J_tt = M2*LC2**2 + I2
J_at = M2*L1*LC2
J_tot = J_aa + J_tt + 2*J_at
C_sd = (-J_aa*p2 + J_tt*p1 + J_at*(p1-p2)) / ps**2
M11 = Mt*R_TARGET**2; M12 = R_TARGET; M22 = J_tot/ps**2
det_M = M11*M22 - M12**2
iM11 = M22/det_M; iM12 = -M12/det_M; iM22 = M11/det_M
gMR = g*Mt*R_TARGET; g_ps = g/ps
w_eff = np.sqrt(iM11*gMR); lam_eff = np.sqrt(iM22*g_ps)
T_quarter = np.pi/(2*w_eff)

# 시험 조건
eps = 11.5
T_f = 0.020
T_w1 = 0.340
delta_b0 = np.radians(5.0)
d_fold = h_com * (1+eps) * abs(delta_b0) / c_foot * np.sign(delta_b0)

# 서보의 이론적 응답
omega_servo_min = np.sqrt(KP / 0.05)   # I_eff ~ 0.05 추정
period_servo = 2*np.pi/omega_servo_min

print("="*70)
print(f"  KP={KP}  KD={KD_S}  DT={DT}  (TAU_MAX={TAU_MAX})")
print(f"  서보 대략 응답 주파수 ~ {omega_servo_min:.0f} rad/s, "
      f"주기 ~ {period_servo*1000:.1f} ms")
print(f"  T_f={T_f*1000:.0f}ms 와 비교: T_f/period_servo = {T_f/period_servo:.2f}")
print(f"  (>1 이면 서보 추적 양호)")
print(f"  d_fold = {np.degrees(d_fold):.2f}°")
print("="*70)
print()

# ============================================================
# MuJoCo XML
# ============================================================
HP = 0.8/2.0; RH = HP - BD/2.0; RV = R_TARGET
POLE_H = 1.0
lw2 = L2/8; uw2 = L2/8; bd2 = BD/2
PR = max(0.005, L2*0.05); RR = max(0.003, L2*0.015); RM = 0.01

xml = f"""<?xml version="1.0"?>
<mujoco model="v15_v5">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="300" tolerance="1e-12">
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
             solref="0.0001 1" solimp="0.9999 0.9999 0.00001"/>
  </equality>
  <actuator>
    <general name="hip_servo" joint="hip"
             gainprm="{KP} 0 0" biasprm="0 {-KP} {-KD_S}"
             biastype="affine"
             ctrlrange="-1.57 1.57" ctrllimited="true"
             forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/>
  </actuator>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

ankle_qadr = model.jnt_qposadr[model.joint('ankle').id]
hip_qadr = model.jnt_qposadr[model.joint('hip').id]

# 초기: alpha=theta=delta_b0 (phi=0, beta=delta_b0)
data.qpos[ankle_qadr] = delta_b0
data.qpos[hip_qadr] = 0.0
mujoco.mj_forward(model, data)

def measure(m, d):
    phi = d.joint('phi_Y').qpos[0]
    foot = d.xpos[m.body('lower_body').id]
    cl = d.xipos[m.body('lower_body').id]
    cu = d.xipos[m.body('upper_body').id]
    com = (M1*cl + M2*cu) / Mt
    beta = np.arctan2(com[0]-foot[0], com[2]-foot[2])
    x_com_world = com[0]
    delta_b = x_com_world / h_com
    return phi, beta, x_com_world, delta_b

phi0_mj, beta0_mj, xcom0, db0 = measure(model, data)
print(f"[INIT] phi={np.degrees(phi0_mj):.2f}°  beta={np.degrees(beta0_mj):.2f}°  "
      f"x_CoM={xcom0*100:.2f}cm  δ_b={np.degrees(db0):.2f}°")
print()

# ============================================================
# 실행 (DT 가 작으므로 N 매우 커짐)
# ============================================================
t_e_start = T_f + T_w1
t_e_end = 2*T_f + T_w1
T_END = t_e_end + 0.8

# 기록은 다운샘플링 (1ms 마다)
log_dt = 0.001
log_interval = max(1, int(log_dt / DT))

rec = {k: [] for k in ['t','phi','beta','x_com','delta_b',
                        'alpha','theta','d_cmd','d_act','tau','phase']}

phi_prev = 0.0
db_prev = db0
phi_crossed_t = None
db_crossed_t = None
hip_overshoot_max = 0.0

N = int(T_END / DT)
for i in range(N):
    t = data.time

    if t < T_f:
        phase = 0
        frac = t / T_f
        ctrl = frac * d_fold
    elif t < t_e_start:
        phase = 1
        ctrl = d_fold
    elif t < t_e_end:
        phase = 2
        frac = (t - t_e_start) / T_f
        ctrl = d_fold * (1 - frac)
    else:
        phase = 3
        ctrl = 0.0

    phi, beta, xcom, db = measure(model, data)
    hip_act = data.joint('hip').qpos[0]

    # 트래킹
    if abs(hip_act) > hip_overshoot_max:
        hip_overshoot_max = abs(hip_act)

    if i % log_interval == 0:
        alpha = data.joint('ankle').qpos[0]
        theta = alpha + hip_act
        # actuator force
        tau = data.actuator_force[0] if len(data.actuator_force) > 0 else 0.0
        rec['t'].append(t)
        rec['phi'].append(np.degrees(phi))
        rec['beta'].append(np.degrees(beta))
        rec['x_com'].append(xcom*100)
        rec['delta_b'].append(np.degrees(db))
        rec['alpha'].append(np.degrees(alpha))
        rec['theta'].append(np.degrees(theta))
        rec['d_cmd'].append(np.degrees(ctrl))
        rec['d_act'].append(np.degrees(hip_act))
        rec['tau'].append(tau)
        rec['phase'].append(phase)

    if phase == 3:
        if phi_crossed_t is None and (t - t_e_end) > 0.005:
            if phi_prev * phi < 0:
                phi_crossed_t = t
        if db_crossed_t is None and (t - t_e_end) > 0.005:
            if db_prev * db < 0:
                db_crossed_t = t

    data.ctrl[0] = ctrl
    phi_prev = phi
    db_prev = db
    mujoco.mj_step(model, data)

    if np.any(np.isnan(data.qpos)):
        print(f"NaN at t={t:.4f}s — 시뮬 발산. KP 너무 큰지/DT 너무 큰지 확인")
        break

    if (phi_crossed_t is not None and db_crossed_t is not None and
        (t - max(phi_crossed_t, db_crossed_t)) > 0.05):
        break

# ============================================================
# 결과
# ============================================================
def value_at(t_target, key):
    if t_target is None: return None
    idx = min(range(len(rec['t'])), key=lambda k: abs(rec['t'][k]-t_target))
    return rec[key][idx]

print(f"[MuJoCo]")
print(f"  hip 최대 |각도| = {np.degrees(hip_overshoot_max):.2f}° (cmd 최대 = {np.degrees(d_fold):.2f}°)")
err_max = max(abs(c-a) for c,a in zip(rec['d_cmd'], rec['d_act']))
print(f"  서보 추적 오차 max = {err_max:.2f}°")
tau_max = max(abs(t_) for t_ in rec['tau'])
print(f"  토크 max = {tau_max:.2f} N·m  (limit={TAU_MAX})")
print()

print(f"  교차 검출:")
print(f"    phi=0  at  t={phi_crossed_t*1000:.1f}ms" if phi_crossed_t else "    phi=0: 검출 안됨")
print(f"    δ_b=0  at  t={db_crossed_t*1000:.1f}ms" if db_crossed_t else "    δ_b=0: 검출 안됨")
print()

if phi_crossed_t:
    print(f"[At phi=0]")
    print(f"  beta = {value_at(phi_crossed_t,'beta'):.3f}°  → r_beta = {value_at(phi_crossed_t,'beta')/np.degrees(delta_b0):.4f}")
    print(f"  δ_b  = {value_at(phi_crossed_t,'delta_b'):.3f}°  → r_δb   = {value_at(phi_crossed_t,'delta_b')/np.degrees(delta_b0):.4f}")
if db_crossed_t:
    print(f"[At δ_b=0]")
    print(f"  phi  = {value_at(db_crossed_t,'phi'):.3f}°")
    print(f"  T_w2(δ_b) = {(db_crossed_t-t_e_end)*1000:.1f}ms")

# ============================================================
# ODE
# ============================================================
def run_ode():
    sigma0 = Mt*h_com*delta_b0
    a_fold = 4*d_fold/T_f**2
    t2_o = T_f + T_w1; t3_o = t2_o + T_f
    T_end_o = t3_o + 4*T_quarter

    def get_dd(t):
        if t < T_f/2: return +a_fold
        elif t < T_f: return -a_fold
        elif t < t2_o: return 0.0
        elif t < t2_o+T_f/2: return -a_fold
        elif t < t3_o: return +a_fold
        else: return 0.0

    def rhs(t, y):
        phi, dp, sig, ds = y
        dd = get_dd(t)
        r1 = -gMR*phi; r2 = g_ps*sig - C_sd*dd
        return [dp, iM11*r1+iM12*r2, ds, iM12*r1+iM22*r2]

    sol = solve_ivp(rhs, (0, T_end_o), [0,0,sigma0,0],
                    method='RK45', rtol=1e-10, atol=1e-12,
                    max_step=T_f/40, dense_output=True)
    return sol, t3_o

sol_ode, t3_ode = run_ode()
t_ode_grid = np.linspace(0, min(rec['t'][-1], sol_ode.t[-1]), 3000)
phi_ode = np.array([sol_ode.sol(t)[0] for t in t_ode_grid])
beta_ode = np.array([sol_ode.sol(t)[2]/(Mt*h_com) for t in t_ode_grid])
db_ode = beta_ode - (R_TARGET/h_com)*phi_ode

# ODE 의 phi=0 교차
t_grid = np.linspace(t3_ode+0.003, sol_ode.t[-1], 8000)
phi_grid = np.array([sol_ode.sol(t)[0] for t in t_grid])
phi0_ode_t = None
for k in range(len(phi_grid)-1):
    if phi_grid[k]*phi_grid[k+1] < 0:
        phi0_ode_t = brentq(lambda t: sol_ode.sol(t)[0], t_grid[k], t_grid[k+1])
        break
if phi0_ode_t:
    beta_at_phi0 = sol_ode.sol(phi0_ode_t)[2]/(Mt*h_com)
    print(f"\n[ODE @ phi=0]  t={phi0_ode_t*1000:.1f}ms  beta={np.degrees(beta_at_phi0):.3f}°  "
          f"r={beta_at_phi0/delta_b0:.4f}")

# ============================================================
# 플롯
# ============================================================
fig, axes = plt.subplots(7, 1, figsize=(14, 18), sharex=True)
phase_colors = ['#FFE08A', '#A8E6A1', '#9CC9F2', '#E2C6F2']
phase_bounds = [(0, T_f), (T_f, t_e_start), (t_e_start, t_e_end), (t_e_end, rec['t'][-1])]
for ax in axes:
    for (a_, b_), c_ in zip(phase_bounds, phase_colors):
        ax.axvspan(a_*1000, b_*1000, alpha=0.25, color=c_)

axes[0].plot([t*1000 for t in rec['t']], rec['delta_b'], 'darkgreen', lw=2, label='MuJoCo')
axes[0].plot(t_ode_grid*1000, np.degrees(db_ode), 'darkgreen', ls='--', lw=1.5, alpha=0.7, label='ODE')
axes[0].axhline(0, color='k', lw=0.5)
if db_crossed_t: axes[0].axvline(db_crossed_t*1000, color='red', ls=':')
axes[0].set_ylabel('δ_balance [°]')
axes[0].set_title(f'KP={KP}  KD={KD_S}  DT={DT*1000:.2f}ms  |  '
                  f'd_fold={np.degrees(d_fold):.0f}°  err_max={err_max:.1f}°  '
                  f'τ_max={tau_max:.0f}N·m')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].plot([t*1000 for t in rec['t']], rec['beta'], 'r-', lw=2, label='MuJoCo')
axes[1].plot(t_ode_grid*1000, np.degrees(beta_ode), 'r--', lw=1.5, alpha=0.7, label='ODE')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('beta [°]'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

axes[2].plot([t*1000 for t in rec['t']], rec['phi'], 'b-', lw=2, label='MuJoCo')
axes[2].plot(t_ode_grid*1000, np.degrees(phi_ode), 'b--', lw=1.5, alpha=0.7, label='ODE')
axes[2].axhline(0, color='k', lw=0.5)
if phi_crossed_t: axes[2].axvline(phi_crossed_t*1000, color='blue', ls=':')
axes[2].set_ylabel('phi [°]'); axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

axes[3].plot([t*1000 for t in rec['t']], rec['d_cmd'], 'g-', lw=2, label='cmd')
axes[3].plot([t*1000 for t in rec['t']], rec['d_act'], 'm--', lw=1.5, label='actual')
axes[3].axhline(0, color='k', lw=0.5)
axes[3].set_ylabel('hip [°]'); axes[3].legend(fontsize=9); axes[3].grid(True, alpha=0.3)

axes[4].plot([t*1000 for t in rec['t']],
             [c-a for c,a in zip(rec['d_cmd'], rec['d_act'])], 'k-', lw=1)
axes[4].axhline(0, color='k', lw=0.5)
axes[4].set_ylabel('servo err [°]'); axes[4].grid(True, alpha=0.3)

axes[5].plot([t*1000 for t in rec['t']], rec['tau'], 'orange', lw=1)
axes[5].axhline(0, color='k', lw=0.5)
axes[5].set_ylabel('hip torque [N·m]'); axes[5].grid(True, alpha=0.3)

axes[6].plot([t*1000 for t in rec['t']], rec['alpha'], 'orange', label='alpha')
axes[6].plot([t*1000 for t in rec['t']], rec['theta'], 'purple', label='theta')
axes[6].axhline(0, color='k', lw=0.5)
axes[6].set_ylabel('angle [°]'); axes[6].set_xlabel('time [ms]')
axes[6].legend(fontsize=9); axes[6].grid(True, alpha=0.3)

plt.tight_layout()
out = f'v15_ode_match_v5_kp{int(KP)}.png'
plt.savefig(out, dpi=130)
print(f"\nSaved: {out}")

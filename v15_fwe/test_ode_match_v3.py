# -*- coding: utf-8 -*-
"""
ODE 정확 조건 + test_diverge.py 와 같은 PD 서보 (안정적)
=========================================================
- KP=200, KD=10, TAU_MAX=10000  (test_diverge 가 안정적으로 돌아간 설정)
- d_fold = 52deg (eps=11.5 에서 ODE 가 요구한 값)
- T_f=20ms, T_w1=340ms
- linear ramp (test_diverge 스타일)
- ODE 와 동시 비교
"""
import numpy as np
import mujoco
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 로봇 파라미터
# ============================================================
g = 9.81
M1 = 0.1; M2 = 0.25
L1 = 0.2; L2 = 0.2
BD = 0.10
R_TARGET = 0.3
GRAV = 9.81
DT = 0.001

KP = 200.0
KD_S = 10.0
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
beta0 = np.radians(5.0)
d_fold = h_com * (1+eps) * abs(beta0) / c_foot * np.sign(beta0)

print(f"d_fold = {np.degrees(d_fold):.2f} deg")
print(f"avg ramp velocity = {np.degrees(d_fold)/T_f:.0f} deg/s (linear)")
print()

# ============================================================
# MuJoCo XML — test_diverge 와 같은 구조
# ============================================================
HP = 0.8/2.0; RH = HP - BD/2.0; RV = R_TARGET
POLE_H = 1.0
lw2 = L2/8; uw2 = L2/8; bd2 = BD/2
PR = max(0.005, L2*0.05); RR = max(0.003, L2*0.015); RM = 0.01

xml = f"""<?xml version="1.0"?>
<mujoco model="v15_v3">
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
             gainprm="{KP} 0 0" biasprm="0 {-KP} {-KD_S}"
             ctrlrange="-1.57 1.57" ctrllimited="true"
             forcerange="{-TAU_MAX} {TAU_MAX}" forcelimited="true"/>
  </actuator>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

ankle_qadr = model.jnt_qposadr[model.joint('ankle').id]
hip_qadr = model.jnt_qposadr[model.joint('hip').id]

# 초기: alpha=theta=beta0 -> beta=beta0
data.qpos[ankle_qadr] = beta0
data.qpos[hip_qadr] = 0.0
mujoco.mj_forward(model, data)

def compute_beta_mj(m, d):
    foot = d.xpos[m.body('lower_body').id]
    cl = d.xipos[m.body('lower_body').id]
    cu = d.xipos[m.body('upper_body').id]
    com = (M1*cl + M2*cu) / Mt
    return np.arctan2(com[0]-foot[0], com[2]-foot[2])

print(f"[INIT] beta = {np.degrees(compute_beta_mj(model, data)):.3f} deg")

# 단계 시간
t_e_start = T_f + T_w1
t_e_end = 2*T_f + T_w1
T_END = t_e_end + 1.0

rec = {k: [] for k in ['t','phi','beta','alpha','theta','d_cmd','d_act','ctrl','phase']}

phi_prev = 0.0
phi_crossed_t = None

N = int(T_END / DT)
for i in range(N):
    t = data.time

    # 단계별 ctrl
    if t < T_f:
        phase = 0
        frac = t / T_f
        ctrl = frac * d_fold
        d_cmd = ctrl
    elif t < t_e_start:
        phase = 1
        ctrl = d_fold
        d_cmd = d_fold
    elif t < t_e_end:
        phase = 2
        frac = (t - t_e_start) / T_f
        ctrl = d_fold * (1 - frac)
        d_cmd = ctrl
    else:
        phase = 3
        ctrl = 0.0
        d_cmd = 0.0

    # 측정
    phi = data.joint('phi_Y').qpos[0]
    beta = compute_beta_mj(model, data)
    alpha = data.joint('ankle').qpos[0]
    hip_act = data.joint('hip').qpos[0]
    theta = alpha + hip_act

    rec['t'].append(t)
    rec['phi'].append(np.degrees(phi))
    rec['beta'].append(np.degrees(beta))
    rec['alpha'].append(np.degrees(alpha))
    rec['theta'].append(np.degrees(theta))
    rec['d_cmd'].append(np.degrees(d_cmd))
    rec['d_act'].append(np.degrees(hip_act))
    rec['ctrl'].append(np.degrees(ctrl))
    rec['phase'].append(phase)

    if phase == 3 and phi_crossed_t is None and (t - t_e_end) > 0.005:
        if phi_prev * phi < 0:
            phi_crossed_t = t

    data.ctrl[0] = ctrl
    phi_prev = phi
    mujoco.mj_step(model, data)

    if np.any(np.isnan(data.qpos)):
        print(f"NaN at t={t:.3f}s")
        break

    if phi_crossed_t is not None and (t - phi_crossed_t) > 0.05:
        break

# 결과
if phi_crossed_t is not None:
    T_w2_mj = phi_crossed_t - t_e_end
    closest = min(range(len(rec['t'])), key=lambda k: abs(rec['t'][k]-phi_crossed_t))
    beta_end_mj = rec['beta'][closest]
    r_mj = beta_end_mj / np.degrees(beta0)
else:
    T_w2_mj = None
    beta_end_mj = rec['beta'][-1]
    r_mj = beta_end_mj / np.degrees(beta0)

print(f"\n[MuJoCo PD] T_w2={T_w2_mj*1000 if T_w2_mj else 'None'}ms  "
      f"beta_end={beta_end_mj:.2f}d  r={r_mj:.4f}")

# 서보 추적 진단
import numpy as np
err_max = max(abs(c - a) for c, a in zip(rec['d_cmd'], rec['d_act']))
print(f"[Servo] max(|cmd-actual|) = {err_max:.2f} deg")
hip_max = max(abs(h) for h in rec['d_act'])
hip_min = min(rec['d_act'])
print(f"[Hip] range = [{hip_min:.1f}, {hip_max:.1f}] deg  (limit ±90)")

# ============================================================
# ODE
# ============================================================
def run_ode():
    sigma0 = Mt*h_com*beta0
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
    t_w2_grid = np.linspace(t3_o+0.003, T_end_o, 8000)
    phi_w2 = np.array([sol.sol(t)[0] for t in t_w2_grid])
    cross_t = None
    for k in range(len(phi_w2)-1):
        if phi_w2[k]*phi_w2[k+1] < 0:
            cross_t = brentq(lambda t: sol.sol(t)[0], t_w2_grid[k], t_w2_grid[k+1], xtol=1e-10)
            break
    return sol, t3_o, cross_t

sol_ode, t3_ode, cross_t_ode = run_ode()
if cross_t_ode:
    y_end = sol_ode.sol(cross_t_ode)
    beta_end_ode = y_end[2]/(Mt*h_com)
    r_ode = beta_end_ode / beta0
    T_w2_ode = cross_t_ode - t3_ode
    print(f"\n[ODE] T_w2={T_w2_ode*1000:.1f}ms  beta_end={np.degrees(beta_end_ode):.2f}d  r={r_ode:.4f}")
else:
    beta_end_ode = None; r_ode = None; T_w2_ode = None

t_ode = np.linspace(0, min(rec['t'][-1], sol_ode.t[-1]), 3000)
phi_ode = np.array([sol_ode.sol(t)[0] for t in t_ode])
beta_ode_t = np.array([sol_ode.sol(t)[2]/(Mt*h_com) for t in t_ode])

# ============================================================
# 플롯
# ============================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
phase_colors = ['#FFE08A', '#A8E6A1', '#9CC9F2', '#E2C6F2']
phase_bounds = [(0, T_f), (T_f, t_e_start), (t_e_start, t_e_end), (t_e_end, rec['t'][-1])]
for ax in axes:
    for (a_, b_), c_ in zip(phase_bounds, phase_colors):
        ax.axvspan(a_*1000, b_*1000, alpha=0.25, color=c_)

axes[0].plot([t*1000 for t in rec['t']], rec['beta'], 'r-', lw=2, label='MuJoCo')
axes[0].plot(t_ode*1000, np.degrees(beta_ode_t), 'r--', lw=1.5, label='ODE', alpha=0.7)
axes[0].axhline(0, color='k', lw=0.5)
if phi_crossed_t:
    axes[0].axvline(phi_crossed_t*1000, color='blue', ls=':')
if cross_t_ode:
    axes[0].axvline(cross_t_ode*1000, color='cyan', ls=':')
axes[0].set_ylabel('beta [deg]')
title = f'eps={eps}, T_f={T_f*1000}ms, T_w1={T_w1*1000}ms, d_fold={np.degrees(d_fold):.0f}d, KP={KP}, KD={KD_S}\n'
title += f'r_MJ={r_mj:.4f}  r_ODE={r_ode:.4f}' if r_ode else f'r_MJ={r_mj:.4f}'
axes[0].set_title(title)
axes[0].legend(loc='best', fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].plot([t*1000 for t in rec['t']], rec['phi'], 'b-', lw=2, label='MuJoCo')
axes[1].plot(t_ode*1000, np.degrees(phi_ode), 'b--', lw=1.5, label='ODE', alpha=0.7)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('phi [deg]'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

axes[2].plot([t*1000 for t in rec['t']], rec['d_cmd'], 'g-', lw=2, label='cmd')
axes[2].plot([t*1000 for t in rec['t']], rec['d_act'], 'm--', lw=1.5, label='actual')
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_ylabel('hip [deg]'); axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

axes[3].plot([t*1000 for t in rec['t']],
             [c-a for c,a in zip(rec['d_cmd'], rec['d_act'])], 'k-', lw=1)
axes[3].axhline(0, color='k', lw=0.5)
axes[3].set_ylabel('servo err [deg]'); axes[3].grid(True, alpha=0.3)

axes[4].plot([t*1000 for t in rec['t']], rec['alpha'], 'orange', label='alpha')
axes[4].plot([t*1000 for t in rec['t']], rec['theta'], 'purple', label='theta')
axes[4].axhline(0, color='k', lw=0.5)
axes[4].set_ylabel('angle [deg]'); axes[4].set_xlabel('time [ms]')
axes[4].legend(fontsize=9); axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v15_ode_match_v3.png', dpi=130)
print("\nSaved: v15_ode_match_v3.png")

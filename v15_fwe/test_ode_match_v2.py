# -*- coding: utf-8 -*-
"""
ODE 정확 조건 + Hip 운동학적 강제 (kinematic forcing)
======================================================
ODE 모델은 hip 가속도 dd 를 외부 함수로 받음 — 즉 hip 운동은
prescribed (강제) 되는 것. 토크 제어가 아님.

v1 (test_ode_match.py) 의 문제:
  - PD 서보로 제어 -> KP 가 너무 높아 폭주
  - r=0.47 은 카오스 운동 중 우연히 phi=0 지나간 값

v2 (이 파일):
  - hip qpos, qvel 을 매 step 직접 강제 (kinematic constraint)
  - ODE 와 동일한 hip 궤적 보장
  - 이러면 MuJoCo 의 'rest of physics' (줄, 발, 관성 결합) 만의
    효과를 ODE 와 비교할 수 있음
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
DT = 0.0005   # 더 작게 (kinematic forcing 정확도)

LC1 = L1/2; LC2 = L2/2
Mt = M1 + M2
I1 = M1*L1**2/12; I2 = M2*L2**2/12
p1 = M1*LC1 + M2*L1
p2 = M2*LC2
ps = p1 + p2
h_com = (M1*LC1 + M2*(L1+LC2)) / Mt
c_foot = p2 / ps

J_aa = M1*LC1**2 + I1 + M2*L1**2
J_tt = M2*LC2**2 + I2
J_at = M2*L1*LC2
J_tot = J_aa + J_tt + 2*J_at
C_sd = (-J_aa*p2 + J_tt*p1 + J_at*(p1-p2)) / ps**2
M11 = Mt*R_TARGET**2
M12 = R_TARGET
M22 = J_tot/ps**2
det_M = M11*M22 - M12**2
iM11 = M22/det_M; iM12 = -M12/det_M; iM22 = M11/det_M
gMR = g*Mt*R_TARGET
g_ps = g/ps
w_eff = np.sqrt(iM11*gMR)
lam_eff = np.sqrt(iM22*g_ps)
T_quarter = np.pi/(2*w_eff)

# ============================================================
# 시험 조건
# ============================================================
eps = 11.5
T_f = 0.020
T_w1 = 0.340
beta0 = np.radians(5.0)
d_fold = h_com * (1+eps) * abs(beta0) / c_foot * np.sign(beta0)

print(f"d_fold = {np.degrees(d_fold):.2f} deg = {d_fold:.4f} rad")
print(f"h_com={h_com:.4f}  c_foot={c_foot:.4f}")
print(f"w_eff={w_eff:.4f}  lam_eff={lam_eff:.4f}")

# 명령 프로파일 (위치, 속도, 가속도)
def fold_kinematics(t_local, d, Tf):
    """Triangular acceleration:
       a(t) = +a for t<Tf/2, -a for Tf/2<t<Tf, 0 otherwise
       Returns (pos, vel, acc)
    """
    if t_local <= 0:
        return (0.0, 0.0, 0.0)
    if t_local >= Tf:
        return (d, 0.0, 0.0)
    a_mag = 4*abs(d)/Tf**2
    sign = np.sign(d)
    half = Tf/2
    if t_local < half:
        pos = sign * 0.5 * a_mag * t_local**2
        vel = sign * a_mag * t_local
        acc = sign * a_mag
    else:
        dt = t_local - half
        pos = sign * (0.5*a_mag*half**2 + a_mag*half*dt - 0.5*a_mag*dt**2)
        vel = sign * (a_mag*half - a_mag*dt)
        acc = -sign * a_mag
    return (pos, vel, acc)

# ============================================================
# MuJoCo XML — hip 을 actuator 가 아닌 free joint 로 두고
#               qpos/qvel 을 매 step 직접 설정
# ============================================================
HP = 0.8/2.0; RH = HP - BD/2.0; RV = R_TARGET
POLE_H = 1.0
lw2 = L2/8; uw2 = L2/8; bd2 = BD/2
PR = max(0.005, L2*0.05); RR = max(0.003, L2*0.015); RM = 0.01

xml = f"""<?xml version="1.0"?>
<mujoco model="v15_kin">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>
  <default><geom contype="0" conaffinity="0"/><joint damping="0" armature="0"/></default>
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
          <joint name="hip" type="hinge" axis="0 1 0"
                 range="-3.5 3.5" limited="true"/>
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
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

ankle_qadr = model.jnt_qposadr[model.joint('ankle').id]
ankle_dadr = model.jnt_dofadr[model.joint('ankle').id]
hip_qadr = model.jnt_qposadr[model.joint('hip').id]
hip_dadr = model.jnt_dofadr[model.joint('hip').id]

# 초기: alpha = theta = beta0 (uniform tilt, β=β0)
data.qpos[ankle_qadr] = beta0
data.qpos[hip_qadr] = 0.0
mujoco.mj_forward(model, data)

def compute_beta_mj(m, d):
    foot = d.xpos[m.body('lower_body').id]
    cl = d.xipos[m.body('lower_body').id]
    cu = d.xipos[m.body('upper_body').id]
    com = (M1*cl + M2*cu) / Mt
    return np.arctan2(com[0]-foot[0], com[2]-foot[2])

beta_init = compute_beta_mj(model, data)
print(f"[INIT MJ] beta = {np.degrees(beta_init):.3f} deg")

# ============================================================
# MuJoCo 실행 + hip kinematic forcing
# ============================================================
t_e_start = T_f + T_w1
t_e_end = 2*T_f + T_w1
T_END = t_e_end + 1.0

rec = {k: [] for k in ['t', 'phi', 'beta', 'alpha', 'theta', 'hip_pos', 'hip_vel',
                        'hip_cmd_pos', 'hip_cmd_vel', 'phase']}

phi_prev = 0.0
phi_crossed_t = None
phase = 0

N = int(T_END / DT)
for i in range(N):
    t = data.time

    # 단계별 hip 명령 계산
    if t < T_f:
        phase = 0
        d_pos, d_vel, _ = fold_kinematics(t, d_fold, T_f)
    elif t < t_e_start:
        phase = 1
        d_pos, d_vel = d_fold, 0.0
    elif t < t_e_end:
        phase = 2
        p_e, v_e, _ = fold_kinematics(t - t_e_start, d_fold, T_f)
        d_pos = d_fold - p_e
        d_vel = -v_e
    else:
        phase = 3
        d_pos, d_vel = 0.0, 0.0

    # Hip kinematic forcing (qpos & qvel 직접 설정)
    data.qpos[hip_qadr] = d_pos
    data.qvel[hip_dadr] = d_vel

    # forward kinematics 동기화
    mujoco.mj_forward(model, data)

    # 측정
    phi = data.joint('phi_Y').qpos[0]
    beta = compute_beta_mj(model, data)
    alpha = data.joint('ankle').qpos[0]
    hip_pos_actual = data.joint('hip').qpos[0]
    hip_vel_actual = data.joint('hip').qvel[0]
    theta = alpha + hip_pos_actual

    rec['t'].append(t)
    rec['phi'].append(np.degrees(phi))
    rec['beta'].append(np.degrees(beta))
    rec['alpha'].append(np.degrees(alpha))
    rec['theta'].append(np.degrees(theta))
    rec['hip_pos'].append(np.degrees(hip_pos_actual))
    rec['hip_vel'].append(np.degrees(hip_vel_actual))
    rec['hip_cmd_pos'].append(np.degrees(d_pos))
    rec['hip_cmd_vel'].append(np.degrees(d_vel))
    rec['phase'].append(phase)

    # phi=0 교차 (W2 시작 후 5ms 이상)
    if phase == 3 and phi_crossed_t is None and (t - t_e_end) > 0.005:
        if phi_prev * phi < 0:
            phi_crossed_t = t

    phi_prev = phi
    mujoco.mj_step(model, data)

    if phi_crossed_t is not None and (t - phi_crossed_t) > 0.1:
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

print(f"\n[MuJoCo RESULT (kinematic forcing)]")
if phi_crossed_t:
    print(f"  T_w2 = {T_w2_mj*1000:.1f} ms")
print(f"  beta_end = {beta_end_mj:.3f} deg")
print(f"  r = {r_mj:.4f}")

# ============================================================
# ODE 동일 조건
# ============================================================
def run_ode():
    sigma0 = Mt*h_com*beta0
    a_fold = 4*d_fold/T_f**2
    t2_o = T_f + T_w1
    t3_o = t2_o + T_f
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
        r1 = -gMR*phi
        r2 = g_ps*sig - C_sd*dd
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
else:
    beta_end_ode = None
    r_ode = None
    T_w2_ode = None

print(f"\n[ODE RESULT]")
if T_w2_ode:
    print(f"  T_w2 = {T_w2_ode*1000:.1f} ms")
    print(f"  beta_end = {np.degrees(beta_end_ode):.3f} deg")
    print(f"  r = {r_ode:.4f}")

# ODE 궤적
t_ode = np.linspace(0, min(rec['t'][-1], sol_ode.t[-1]), 3000)
phi_ode = np.array([sol_ode.sol(t)[0] for t in t_ode])
beta_ode_t = np.array([sol_ode.sol(t)[2]/(Mt*h_com) for t in t_ode])

# ============================================================
# 플롯
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
phase_colors = ['#FFE08A', '#A8E6A1', '#9CC9F2', '#E2C6F2']
phase_bounds = [(0, T_f), (T_f, t_e_start), (t_e_start, t_e_end), (t_e_end, rec['t'][-1])]
for ax in axes:
    for (a_, b_), c_ in zip(phase_bounds, phase_colors):
        ax.axvspan(a_*1000, b_*1000, alpha=0.25, color=c_)

# 1. beta
axes[0].plot([t*1000 for t in rec['t']], rec['beta'], 'r-', lw=2, label='MuJoCo (kin)')
axes[0].plot(t_ode*1000, np.degrees(beta_ode_t), 'r--', lw=1.5, label='ODE', alpha=0.7)
axes[0].axhline(0, color='k', lw=0.5)
if phi_crossed_t:
    axes[0].axvline(phi_crossed_t*1000, color='blue', ls=':', alpha=0.7,
                    label=f'MJ phi=0 @ {phi_crossed_t*1000:.0f}ms')
if cross_t_ode:
    axes[0].axvline(cross_t_ode*1000, color='cyan', ls=':', alpha=0.7,
                    label=f'ODE phi=0 @ {cross_t_ode*1000:.0f}ms')
axes[0].set_ylabel('beta [deg]')
title = f'eps={eps}, T_f={T_f*1000:.0f}ms, T_w1={T_w1*1000:.0f}ms, d_fold={np.degrees(d_fold):.0f}d\n'
title += f'r_MJ={r_mj:.4f}  |  r_ODE={r_ode:.4f}' if r_ode is not None else f'r_MJ={r_mj:.4f}'
axes[0].set_title(title)
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(True, alpha=0.3)

# 2. phi
axes[1].plot([t*1000 for t in rec['t']], rec['phi'], 'b-', lw=2, label='MuJoCo')
axes[1].plot(t_ode*1000, np.degrees(phi_ode), 'b--', lw=1.5, label='ODE', alpha=0.7)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('phi [deg]')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(True, alpha=0.3)

# 3. hip cmd vs actual
axes[2].plot([t*1000 for t in rec['t']], rec['hip_cmd_pos'], 'g-', lw=2, label='cmd pos')
axes[2].plot([t*1000 for t in rec['t']], rec['hip_pos'], 'm--', lw=1.5, label='actual pos')
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_ylabel('hip [deg]')
axes[2].legend(loc='best', fontsize=9)
axes[2].grid(True, alpha=0.3)

# 4. alpha, theta
axes[3].plot([t*1000 for t in rec['t']], rec['alpha'], 'orange', lw=1.5, label='alpha')
axes[3].plot([t*1000 for t in rec['t']], rec['theta'], 'purple', lw=1.5, label='theta')
axes[3].axhline(0, color='k', lw=0.5)
axes[3].set_ylabel('angle [deg]')
axes[3].set_xlabel('time [ms]')
axes[3].legend(loc='best', fontsize=9)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v15_ode_match_v2.png', dpi=130)
print("\nSaved: v15_ode_match_v2.png")

# ============================================================
# 비교 요약
# ============================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"{'metric':<20} {'MuJoCo (kin)':>15} {'ODE':>12} {'diff':>10}")
print("-"*70)
if T_w2_mj is not None and T_w2_ode is not None:
    print(f"{'T_w2 (ms)':<20} {T_w2_mj*1000:>15.1f} {T_w2_ode*1000:>12.1f} "
          f"{(T_w2_mj-T_w2_ode)*1000:>+10.1f}")
if beta_end_mj is not None and beta_end_ode is not None:
    print(f"{'beta_end (deg)':<20} {beta_end_mj:>15.3f} "
          f"{np.degrees(beta_end_ode):>12.3f} "
          f"{beta_end_mj - np.degrees(beta_end_ode):>+10.3f}")
if r_mj is not None and r_ode is not None:
    print(f"{'r':<20} {r_mj:>15.4f} {r_ode:>12.4f} "
          f"{r_mj-r_ode:>+10.4f}")
print()

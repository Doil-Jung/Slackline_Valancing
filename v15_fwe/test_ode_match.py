# -*- coding: utf-8 -*-
"""
ODE 정확 조건을 MuJoCo에 그대로 넣어서 비교
============================================
ODE robot_4phase_check.py 가 찾은 수렴 조합:
    eps = 11.5,  T_f = 20ms,  T_w1 = 340ms,  beta0 = 5deg
    -> ODE r = -0.34 (수렴!)

이 스크립트는:
1. MuJoCo + 같은 로봇 파라미터로 1 사이클 실행
2. ODE 같은 조건으로 동시 실행
3. (phi, beta) 궤적 오버레이 플롯
4. r_mujoco vs r_ode 비교

목적: MuJoCo와 ODE 사이의 차이를 정확히 보기 위함.
"""
import numpy as np
import mujoco
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 로봇 파라미터 (robot_4phase_check.py 와 일치)
# ============================================================
g = 9.81
M1 = 0.1; M2 = 0.25
L1 = 0.2; L2 = 0.2
BD = 0.10
R_TARGET = 0.3
GRAV = 9.81
DT = 0.001

# Servo
KP = 500.0
KD_S = 20.0
TAU_MAX = 10000.0  # 일부러 매우 크게 — 서보 한계 영향 배제

# 파생 (ODE 와 같이)
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
iM11 = M22/det_M
iM12 = -M12/det_M
iM22 = M11/det_M
gMR = g*Mt*R_TARGET
g_ps = g/ps
w_eff = np.sqrt(iM11*gMR)
lam_eff = np.sqrt(iM22*g_ps)
T_quarter = np.pi/(2*w_eff)

# ============================================================
# 시험 조건: ODE 가 찾은 수렴 조합
# ============================================================
eps = 11.5
T_f = 0.020
T_w1 = 0.340
beta0 = np.radians(5.0)

d_fold = h_com * (1+eps) * abs(beta0) / c_foot * np.sign(beta0)

print("="*70)
print("ROBOT params (no head)")
print("="*70)
print(f"  M1={M1}  M2={M2}  L1={L1}  L2={L2}  R={R_TARGET}")
print(f"  h_com = {h_com:.4f} m,  c_foot = {c_foot:.4f}")
print(f"  w_eff = {w_eff:.4f} rad/s  (T_quarter = {T_quarter*1000:.1f} ms)")
print(f"  lam_eff = {lam_eff:.4f} rad/s  (1/lam = {1000/lam_eff:.1f} ms)")
print()
print(f"TEST CONDITIONS")
print(f"  eps    = {eps}")
print(f"  T_f    = {T_f*1000:.0f} ms")
print(f"  T_w1   = {T_w1*1000:.0f} ms")
print(f"  beta0  = {np.degrees(beta0):.2f} deg")
print(f"  d_fold = {np.degrees(d_fold):.2f} deg")
print(f"  v_peak (triangle accel) = {2*np.degrees(d_fold)/T_f:.0f} deg/s")
print("="*70)
print()

# ============================================================
# 명령 프로파일: triangular acceleration -> smooth S-position
#   ODE 와 동일하게 가속도 a 가 +a, -a 두 펄스
# ============================================================
def fold_position(t_local, d, Tf):
    """0 -> d over [0, Tf], using triangular acceleration."""
    if t_local <= 0:
        return 0.0
    if t_local >= Tf:
        return d
    a = 4*abs(d)/Tf**2
    sign = np.sign(d)
    half = Tf/2
    if t_local < half:
        return sign * 0.5*a*t_local**2
    else:
        dt = t_local - half
        return sign * (0.5*a*half**2 + a*half*dt - 0.5*a*dt**2)

# ============================================================
# MuJoCo 셋업 (test_diverge.py 패턴)
# ============================================================
HP = 0.8 / 2.0
RH = HP - BD/2.0
RV = R_TARGET
POLE_H = 1.0
lw2 = L2/8; uw2 = L2/8; bd2 = BD/2
PR = max(0.005, L2*0.05)
RR = max(0.003, L2*0.015)
RM = 0.01
B_THETA = 0.01

xml = f"""<?xml version="1.0"?>
<mujoco model="v15_ode_match">
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

def compute_beta_mj(m, d):
    """발 기준 CoM 기울기 (MuJoCo 실제 좌표에서)"""
    foot = d.xpos[m.body('lower_body').id]
    cl = d.xipos[m.body('lower_body').id]
    cu = d.xipos[m.body('upper_body').id]
    com = (M1*cl + M2*cu) / Mt
    return np.arctan2(com[0]-foot[0], com[2]-foot[2])

# 초기 조건: beta0 = 5deg 기울임 (ankle 만 회전)
ankle_qadr = model.jnt_qposadr[model.joint('ankle').id]
hip_qadr = model.jnt_qposadr[model.joint('hip').id]
data.qpos[ankle_qadr] = beta0  # alpha = beta0 (theta=0 -> 같은 방향 기울임)
# 사실 ODE 는 beta0 만 신경쓰니까 alpha=theta=beta0 로 시작
data.qpos[hip_qadr] = 0.0  # hip 은 일단 0 (theta = alpha 가 되도록)
mujoco.mj_forward(model, data)

# 실제 beta 확인
beta_init_mj = compute_beta_mj(model, data)
phi_init = data.joint('phi_Y').qpos[0]
print(f"[MuJoCo INIT] beta = {np.degrees(beta_init_mj):.3f} deg,  phi = {np.degrees(phi_init):.3f} deg")
print()

# ============================================================
# MuJoCo 실행
# ============================================================
# 단계 시간:
#   FOLD: [0, T_f]
#   WAIT1: [T_f, T_f+T_w1]
#   EXTEND: [T_f+T_w1, 2*T_f+T_w1]
#   WAIT2: [2*T_f+T_w1, ~] -> phi=0 검출
t_e_start = T_f + T_w1
t_e_end = 2*T_f + T_w1
T_END = t_e_end + 1.5  # 충분히 길게

rec_t = []; rec_phi = []; rec_beta = []
rec_alpha = []; rec_theta = []
rec_d_cmd = []; rec_d_act = []
rec_phase = []

phi_prev = 0.0
phi_crossed_t = None
phase = 0  # 0=FOLD, 1=WAIT1, 2=EXTEND, 3=WAIT2

N = int(T_END / DT)
for i in range(N):
    t = data.time
    phi = data.joint('phi_Y').qpos[0]
    beta = compute_beta_mj(model, data)
    alpha = data.joint('ankle').qpos[0]
    hip = data.joint('hip').qpos[0]
    # theta in world: alpha + hip
    theta = alpha + hip

    # 현재 단계 결정 및 ctrl 계산
    if t < T_f:
        phase = 0
        d_cmd = fold_position(t, d_fold, T_f)
    elif t < t_e_start:
        phase = 1
        d_cmd = d_fold  # hold
    elif t < t_e_end:
        phase = 2
        # extend: d_fold -> 0
        d_cmd = d_fold - fold_position(t - t_e_start, d_fold, T_f)
    else:
        phase = 3
        d_cmd = 0.0
        # phi=0 교차 검출 (W2 시작 후 약간 시간 지난 후)
        if phi_crossed_t is None and (t - t_e_end) > 0.005:
            if phi_prev * phi < 0:
                phi_crossed_t = t

    rec_t.append(t)
    rec_phi.append(np.degrees(phi))
    rec_beta.append(np.degrees(beta))
    rec_alpha.append(np.degrees(alpha))
    rec_theta.append(np.degrees(theta))
    rec_d_cmd.append(np.degrees(d_cmd))
    rec_d_act.append(np.degrees(hip))
    rec_phase.append(phase)

    data.ctrl[0] = d_cmd
    phi_prev = phi
    mujoco.mj_step(model, data)

    # phi=0 교차했으면 약간만 더 진행하고 종료
    if phi_crossed_t is not None and (t - phi_crossed_t) > 0.1:
        break

# 결과 분석
if phi_crossed_t is not None:
    T_w2_mj = phi_crossed_t - t_e_end
    # phi=0 시점의 beta
    # 가장 가까운 인덱스
    closest = min(range(len(rec_t)), key=lambda k: abs(rec_t[k]-phi_crossed_t))
    beta_end_mj = rec_beta[closest]
    r_mj = beta_end_mj / np.degrees(beta0)
else:
    T_w2_mj = None
    beta_end_mj = rec_beta[-1]
    r_mj = beta_end_mj / np.degrees(beta0)

print(f"[MuJoCo RESULT]")
print(f"  phi=0 crossed: {phi_crossed_t}")
if phi_crossed_t:
    print(f"  T_w2 (measured) = {T_w2_mj*1000:.0f} ms")
print(f"  beta_end = {beta_end_mj:.3f} deg")
print(f"  r = beta_end / beta_0 = {r_mj:.4f}")
print()

# ============================================================
# ODE 시뮬레이션 (robot_4phase_check.py 와 같은 RHS)
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

    # phi=0 교차 검출
    t_w2_grid = np.linspace(t3_o+0.003, T_end_o, 8000)
    phi_w2 = np.array([sol.sol(t)[0] for t in t_w2_grid])
    cross_t = None
    for k in range(len(phi_w2)-1):
        if phi_w2[k]*phi_w2[k+1] < 0:
            from scipy.optimize import brentq
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

print(f"[ODE RESULT]")
print(f"  T_w2 = {T_w2_ode*1000:.0f} ms" if T_w2_ode else "  T_w2 = N/A")
print(f"  beta_end = {np.degrees(beta_end_ode):.3f} deg" if beta_end_ode is not None else "  beta_end = N/A")
print(f"  r = {r_ode:.4f}" if r_ode is not None else "  r = N/A")
print()

# ODE 궤적 (플롯용)
t_ode = np.linspace(0, min(rec_t[-1], sol_ode.t[-1]), 3000)
phi_ode = np.array([sol_ode.sol(t)[0] for t in t_ode])
beta_ode = np.array([sol_ode.sol(t)[2]/(Mt*h_com) for t in t_ode])

# ============================================================
# 비교 플롯
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# Phase 음영
phase_colors = ['#FFE08A', '#A8E6A1', '#9CC9F2', '#E2C6F2']
phase_labels = ['FOLD', 'WAIT1', 'EXTEND', 'WAIT2']
phase_bounds = [(0, T_f), (T_f, t_e_start), (t_e_start, t_e_end), (t_e_end, rec_t[-1])]
for ax in axes:
    for (a_, b_), c_ in zip(phase_bounds, phase_colors):
        ax.axvspan(a_*1000, b_*1000, alpha=0.25, color=c_)

# 1. beta
axes[0].plot([t*1000 for t in rec_t], rec_beta, 'r-', lw=2, label='MuJoCo')
axes[0].plot(t_ode*1000, np.degrees(beta_ode), 'r--', lw=1.5, label='ODE', alpha=0.7)
axes[0].axhline(0, color='k', lw=0.5)
if phi_crossed_t:
    axes[0].axvline(phi_crossed_t*1000, color='blue', ls=':', alpha=0.7,
                    label=f'MuJoCo phi=0 @ {phi_crossed_t*1000:.0f}ms')
if cross_t_ode:
    axes[0].axvline(cross_t_ode*1000, color='cyan', ls=':', alpha=0.7,
                    label=f'ODE phi=0 @ {cross_t_ode*1000:.0f}ms')
axes[0].set_ylabel('beta [deg]')
axes[0].set_title(f'eps={eps}, T_f={T_f*1000:.0f}ms, T_w1={T_w1*1000:.0f}ms, '
                  f'd_fold={np.degrees(d_fold):.0f}d  |  '
                  f'r_MJ={r_mj:.3f}  r_ODE={r_ode:.3f}' if r_ode else
                  f'eps={eps}, T_f={T_f*1000:.0f}ms, T_w1={T_w1*1000:.0f}ms  |  r_MJ={r_mj:.3f}')
axes[0].legend(loc='best', fontsize=9)
axes[0].grid(True, alpha=0.3)

# 2. phi
axes[1].plot([t*1000 for t in rec_t], rec_phi, 'b-', lw=2, label='MuJoCo')
axes[1].plot(t_ode*1000, np.degrees(phi_ode), 'b--', lw=1.5, label='ODE', alpha=0.7)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('phi [deg]')
axes[1].legend(loc='best', fontsize=9)
axes[1].grid(True, alpha=0.3)

# 3. delta cmd vs actual
axes[2].plot([t*1000 for t in rec_t], rec_d_cmd, 'g-', lw=2, label='cmd (target)')
axes[2].plot([t*1000 for t in rec_t], rec_d_act, 'm-', lw=1.5, label='actual hip')
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_ylabel('delta [deg]')
axes[2].legend(loc='best', fontsize=9)
axes[2].grid(True, alpha=0.3)

# 4. alpha, theta
axes[3].plot([t*1000 for t in rec_t], rec_alpha, 'orange', lw=1.5, label='alpha (lower)')
axes[3].plot([t*1000 for t in rec_t], rec_theta, 'purple', lw=1.5, label='theta (upper)')
axes[3].axhline(0, color='k', lw=0.5)
axes[3].set_ylabel('angle [deg]')
axes[3].set_xlabel('time [ms]')
axes[3].legend(loc='best', fontsize=9)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
out_path = 'v15_ode_match.png'
plt.savefig(out_path, dpi=130)
print(f"Saved: {out_path}")

# ============================================================
# 최종 비교 요약
# ============================================================
print()
print("="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"{'metric':<25} {'MuJoCo':>12} {'ODE':>12} {'diff':>12}")
print("-"*70)
if T_w2_mj is not None and T_w2_ode is not None:
    print(f"{'T_w2 (ms)':<25} {T_w2_mj*1000:>12.1f} {T_w2_ode*1000:>12.1f} "
          f"{(T_w2_mj-T_w2_ode)*1000:>+12.1f}")
if beta_end_mj is not None and beta_end_ode is not None:
    print(f"{'beta_end (deg)':<25} {beta_end_mj:>12.3f} "
          f"{np.degrees(beta_end_ode):>12.3f} "
          f"{beta_end_mj - np.degrees(beta_end_ode):>+12.3f}")
if r_mj is not None and r_ode is not None:
    print(f"{'r = beta_end/beta_0':<25} {r_mj:>12.4f} {r_ode:>12.4f} "
          f"{r_mj-r_ode:>+12.4f}")
print()
print(f"Conv criterion: |r| < 1")
print(f"  MuJoCo: |r|={abs(r_mj):.3f} {'OK' if abs(r_mj)<1 else 'DIVERGE'}")
if r_ode is not None:
    print(f"  ODE   : |r|={abs(r_ode):.3f} {'OK' if abs(r_ode)<1 else 'DIVERGE'}")
print()
print("Done.")

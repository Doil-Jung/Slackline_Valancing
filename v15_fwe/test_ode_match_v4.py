# -*- coding: utf-8 -*-
"""
ODE 정확 조건 + delta_balance 기준 제어
======================================
v3 의 통찰:
  beta (수직 기준) = 0 은 진짜 균형이 아님.
  발이 R·phi 만큼 옆으로 가있으면 몸도 (R/h)·phi 만큼 기울어야 균형.

  delta_balance = beta − (R/h)·phi
  → x_CoM = h · delta_balance
  → delta_balance = 0 ⇔ CoM 이 원점 위 (진짜 균형)

v4 변경:
  1. trigger / target / 종료조건 모두 delta_balance 기준
  2. d_fold = h_com * (1+eps) * |delta_balance_0| / c_foot * sign(...)
  3. 사이클 종료: delta_balance = 0 교차 검출
  4. 그래도 phi 도 같이 기록해서 비교
"""
import numpy as np
import mujoco
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 로봇 파라미터
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
delta_b0 = np.radians(5.0)   # ← delta_balance 초기값 (예전엔 beta_0)
d_fold = h_com * (1+eps) * abs(delta_b0) / c_foot * np.sign(delta_b0)

print("="*70)
print(f"  R={R_TARGET}  h_com={h_com:.4f}  R/h = {R_TARGET/h_com:.3f}")
print(f"  R/h*phi 환산: phi=5° → β 보정 {np.degrees(R_TARGET/h_com*np.radians(5)):.2f}°")
print(f"  delta_b0 = {np.degrees(delta_b0):.2f}°")
print(f"  d_fold   = {np.degrees(d_fold):.2f}°")
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
<mujoco model="v15_v4">
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

# 초기: 몸 통째로 5° 기울임 → δ_balance ≈ 5° (phi=0 이므로)
data.qpos[ankle_qadr] = delta_b0
data.qpos[hip_qadr] = 0.0
mujoco.mj_forward(model, data)

def measure(m, d):
    """phi, beta, x_CoM, delta_balance 모두 측정"""
    phi = d.joint('phi_Y').qpos[0]
    foot = d.xpos[m.body('lower_body').id]
    cl = d.xipos[m.body('lower_body').id]
    cu = d.xipos[m.body('upper_body').id]
    com = (M1*cl + M2*cu) / Mt
    # beta: 수직 기준 CoM 기울기 (발에서 본)
    beta = np.arctan2(com[0]-foot[0], com[2]-foot[2])
    # x_CoM: 원점 기준 CoM x좌표
    x_com_world = com[0]
    # delta_balance = x_CoM / h_com  (이게 진짜 균형의 척도)
    delta_b = x_com_world / h_com
    return phi, beta, x_com_world, delta_b

phi0_mj, beta0_mj, xcom0, db0 = measure(model, data)
print(f"[INIT] phi={np.degrees(phi0_mj):.2f}°  beta={np.degrees(beta0_mj):.2f}°  "
      f"x_CoM={xcom0*100:.2f}cm  δ_b={np.degrees(db0):.2f}°")
print()

# ============================================================
# 실행
# ============================================================
t_e_start = T_f + T_w1
t_e_end = 2*T_f + T_w1
T_END = t_e_end + 1.0

rec = {k: [] for k in ['t','phi','beta','x_com','delta_b',
                        'alpha','theta','d_cmd','d_act','phase']}

phi_prev = 0.0
db_prev = db0
phi_crossed_t = None
db_crossed_t = None

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
    alpha = data.joint('ankle').qpos[0]
    hip_act = data.joint('hip').qpos[0]
    theta = alpha + hip_act

    rec['t'].append(t)
    rec['phi'].append(np.degrees(phi))
    rec['beta'].append(np.degrees(beta))
    rec['x_com'].append(xcom*100)
    rec['delta_b'].append(np.degrees(db))
    rec['alpha'].append(np.degrees(alpha))
    rec['theta'].append(np.degrees(theta))
    rec['d_cmd'].append(np.degrees(ctrl))
    rec['d_act'].append(np.degrees(hip_act))
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
        print(f"NaN at t={t:.3f}s")
        break

    # 두 교차 모두 잡았으면 약간 더 진행 후 종료
    if (phi_crossed_t is not None and db_crossed_t is not None and
        (t - max(phi_crossed_t, db_crossed_t)) > 0.05):
        break

# ============================================================
# 결과 정리
# ============================================================
def value_at(t_target, key):
    if t_target is None: return None
    idx = min(range(len(rec['t'])), key=lambda k: abs(rec['t'][k]-t_target))
    return rec[key][idx]

print(f"[MuJoCo] 교차 검출:")
print(f"  phi=0 at t={phi_crossed_t*1000 if phi_crossed_t else 'N/A'} ms")
print(f"  δ_b=0 at t={db_crossed_t*1000 if db_crossed_t else 'N/A'} ms")
print()

if phi_crossed_t:
    print(f"[At phi=0]")
    print(f"  beta = {value_at(phi_crossed_t,'beta'):.3f}°")
    print(f"  δ_b  = {value_at(phi_crossed_t,'delta_b'):.3f}°")
    print(f"  r_beta = {value_at(phi_crossed_t,'beta')/np.degrees(delta_b0):.4f}")
    print(f"  r_δb   = {value_at(phi_crossed_t,'delta_b')/np.degrees(delta_b0):.4f}")
    print()

if db_crossed_t:
    print(f"[At δ_b=0]")
    print(f"  phi  = {value_at(db_crossed_t,'phi'):.3f}°")
    print(f"  beta = {value_at(db_crossed_t,'beta'):.3f}°")
    print(f"  T_w2(δ_b) = {(db_crossed_t-t_e_end)*1000:.1f} ms")
    print()

# 최종 시점 진폭 (수렴 평가)
final_db = rec['delta_b'][-1]
max_db_after_extend = max(abs(d) for t_, d in zip(rec['t'], rec['delta_b']) if t_ > t_e_end)
print(f"[Diagnostics]")
print(f"  δ_b max after EXTEND = {max_db_after_extend:.2f}°")
print(f"  δ_b final = {final_db:.2f}°")
print()

# ============================================================
# ODE 비교
# ============================================================
def run_ode():
    # ODE 의 sigma 는 Mt*h_com*beta. 초기 beta=delta_b0 라 가정
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
# ODE 도 delta_balance 계산 (선형 영역)
db_ode = beta_ode - (R_TARGET/h_com)*phi_ode

# ============================================================
# 플롯
# ============================================================
fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
phase_colors = ['#FFE08A', '#A8E6A1', '#9CC9F2', '#E2C6F2']
phase_bounds = [(0, T_f), (T_f, t_e_start), (t_e_start, t_e_end), (t_e_end, rec['t'][-1])]
for ax in axes:
    for (a_, b_), c_ in zip(phase_bounds, phase_colors):
        ax.axvspan(a_*1000, b_*1000, alpha=0.25, color=c_)

# 1. delta_balance ← 주요 척도
axes[0].plot([t*1000 for t in rec['t']], rec['delta_b'], 'darkgreen', lw=2, label='MuJoCo')
axes[0].plot(t_ode_grid*1000, np.degrees(db_ode), 'darkgreen', ls='--', lw=1.5, alpha=0.7, label='ODE')
axes[0].axhline(0, color='k', lw=0.5)
if db_crossed_t:
    axes[0].axvline(db_crossed_t*1000, color='red', ls=':',
                    label=f'δ_b=0 @ {db_crossed_t*1000:.0f}ms')
axes[0].set_ylabel('δ_balance [deg]')
axes[0].set_title(f'eps={eps}, T_f={T_f*1000:.0f}ms, T_w1={T_w1*1000:.0f}ms, '
                  f'd_fold={np.degrees(d_fold):.0f}°, R/h={R_TARGET/h_com:.2f}')
axes[0].legend(loc='best', fontsize=9); axes[0].grid(True, alpha=0.3)

# 2. beta
axes[1].plot([t*1000 for t in rec['t']], rec['beta'], 'r-', lw=2, label='MuJoCo')
axes[1].plot(t_ode_grid*1000, np.degrees(beta_ode), 'r--', lw=1.5, alpha=0.7, label='ODE')
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_ylabel('beta [deg]'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

# 3. phi
axes[2].plot([t*1000 for t in rec['t']], rec['phi'], 'b-', lw=2, label='MuJoCo')
axes[2].plot(t_ode_grid*1000, np.degrees(phi_ode), 'b--', lw=1.5, alpha=0.7, label='ODE')
axes[2].axhline(0, color='k', lw=0.5)
if phi_crossed_t:
    axes[2].axvline(phi_crossed_t*1000, color='blue', ls=':', alpha=0.5,
                    label=f'phi=0 @ {phi_crossed_t*1000:.0f}ms')
axes[2].set_ylabel('phi [deg]'); axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

# 4. x_CoM (cm)
axes[3].plot([t*1000 for t in rec['t']], rec['x_com'], 'purple', lw=2)
axes[3].axhline(0, color='k', lw=0.5)
axes[3].set_ylabel('x_CoM [cm]'); axes[3].grid(True, alpha=0.3)

# 5. hip
axes[4].plot([t*1000 for t in rec['t']], rec['d_cmd'], 'g-', lw=2, label='cmd')
axes[4].plot([t*1000 for t in rec['t']], rec['d_act'], 'm--', lw=1.5, label='actual')
axes[4].axhline(0, color='k', lw=0.5)
axes[4].set_ylabel('hip [deg]'); axes[4].legend(fontsize=9); axes[4].grid(True, alpha=0.3)

# 6. servo err
axes[5].plot([t*1000 for t in rec['t']],
             [c-a for c,a in zip(rec['d_cmd'], rec['d_act'])], 'k-', lw=1)
axes[5].axhline(0, color='k', lw=0.5)
axes[5].set_ylabel('servo err [deg]'); axes[5].set_xlabel('time [ms]'); axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v15_ode_match_v4.png', dpi=130)
print("Saved: v15_ode_match_v4.png")

# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
진단: 소형 로봇 MuJoCo에서 왜 모든 ε에서 발산하는지 확인
— 단일 케이스 시계열 추적 + 해석 궤적 비교
"""
import numpy as np, mujoco, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_robot_mujoco_band import (
    compute_params, mode_analysis, eps_star_gen,
    generate_small_mjcf, get_state, GRAV, DT
)

# 대표 케이스: L1=L2=25cm, R=35cm
L1, L2, R = 0.25, 0.25, 0.35
params = compute_params(L1, L2)
mode = mode_analysis(params, R)
h = params['h']

print("=" * 70)
print(f"  진단: L1={L1*100}cm L2={L2*100}cm R={R*100}cm")
print(f"  Mt={params['Mt']*1000:.0f}g, h={h*100:.1f}cm")
print(f"  M1={params['M1']*1000:.0f}g, M2_eff={params['M2_eff']*1000:.0f}g")
print(f"  LC1={params['LC1']*100:.1f}cm, LC2_eff={params['LC2_eff']*100:.1f}cm")
print(f"  ω={mode['omega']:.2f} rad/s, λ={mode['lam']:.2f} rad/s")
print(f"  T_osc={2*np.pi/mode['omega']:.3f}s, τ_decay={1/mode['lam']:.3f}s")
print("=" * 70)

# 정지 출발 테스트
beta_i = np.radians(2)
r_ratio = mode['v1'][0] / mode['v1'][1]
es = -h / (r_ratio * R + h)
print(f"\n  ε*_stat = {es:.4f}")

phi_after = h * (1 + es) * beta_i / R
beta_after = -es * beta_i
print(f"  접기 후: φ={np.degrees(phi_after):.2f}°, β={np.degrees(beta_after):.2f}°")
print(f"  x_CoM = R·φ + h·β = {(R*phi_after + h*beta_after)*1000:.1f}mm")

# MuJoCo 시뮬레이션 — 상세 시계열
xml = generate_small_mjcf(params, R)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 초기 상태 설정
phi_mj = -phi_after  # MuJoCo 부호 관례
alpha_mj = beta_after  # β ≈ α when δ=0

ra = model.jnt_qposadr[model.joint('phi_Y').id]
aa = model.jnt_qposadr[model.joint('ankle').id]
ha = model.jnt_qposadr[model.joint('hip').id]
try: rb = model.jnt_qposadr[model.joint('rope_b_Y').id]
except: rb = None

data.qpos[ra] = phi_mj
data.qpos[aa] = alpha_mj - phi_mj
data.qpos[ha] = 0
if rb is not None: data.qpos[rb] = phi_mj

mujoco.mj_forward(model, data)

print(f"\n  초기 qpos: phi_Y={np.degrees(data.qpos[ra]):.3f}° "
      f"ankle={np.degrees(data.qpos[aa]):.3f}° "
      f"hip={np.degrees(data.qpos[ha]):.3f}°")

# 시뮬 실행
times = []; phis = []; alphas = []; thetas = []; xcoms = []
deltas = []; ctrls = []; betas = []

p1 = params['p1']; p2 = params['p2']; Mt = params['Mt']

va = model.jnt_dofadr[model.joint('phi_Y').id]
va_a = model.jnt_dofadr[model.joint('ankle').id]
va_h = model.jnt_dofadr[model.joint('hip').id]

sim_t = 0.5  # 0.5초만 — 빠르게 확인
n = int(sim_t / DT)

print(f"\n  {'시간':>6} | {'φ':>7} {'α':>7} {'θ':>7} {'δ':>7} | {'β':>7} | {'x_CoM':>8} | {'ctrl':>8}")
print(f"  {'-'*75}")

for i in range(n):
    x = get_state(model, data)
    if np.any(np.isnan(x)):
        print(f"  NaN at t={data.time:.4f}!")
        break
    if abs(x[1]) > 1.4:
        print(f"  쓰러짐 at t={data.time:.4f}! α={np.degrees(x[1]):.1f}°")
        break
    
    phi_v = x[0]; alpha_v = x[1]; theta_v = x[2]
    delta_v = theta_v - alpha_v
    delta_d = x[5] - x[4]
    beta_v = (p1*alpha_v + p2*theta_v) / (Mt*h)
    xcom = (-R*phi_v + h*beta_v) * 1000
    
    # PD 제어
    ctrl = np.clip(5000*(0-delta_v) - 100*delta_d, -500, 500)
    data.ctrl[0] = ctrl
    
    if i % 50 == 0:  # 50ms 간격 출력
        times.append(data.time)
        phis.append(np.degrees(phi_v))
        alphas.append(np.degrees(alpha_v))
        thetas.append(np.degrees(theta_v))
        deltas.append(np.degrees(delta_v))
        betas.append(np.degrees(beta_v))
        xcoms.append(xcom)
        ctrls.append(ctrl)
        
        print(f"  {data.time:6.3f} | {np.degrees(phi_v):7.2f} {np.degrees(alpha_v):7.2f} "
              f"{np.degrees(theta_v):7.2f} {np.degrees(delta_v):7.2f} | "
              f"{np.degrees(beta_v):7.2f} | {xcom:8.1f} | {ctrl:8.1f}")
    
    mujoco.mj_step(model, data)

# 해석 궤적 비교
V = np.column_stack([mode['v1'], mode['v2']])
Vi = np.linalg.inv(V)
omega = mode['omega']; lam = mode['lam']

q0 = np.array([phi_after, beta_after])  # 가설 관례
c = Vi @ q0
t_an = np.array(times)
eta1 = c[0] * np.cos(omega * t_an)
eta2 = c[1] * np.cosh(lam * t_an)
phi_an = mode['v1'][0]*eta1 + mode['v2'][0]*eta2
beta_an = mode['v1'][1]*eta1 + mode['v2'][1]*eta2
xcom_an = (R*phi_an + h*beta_an) * 1000

print(f"\n  해석 모드 계수: c1={c[0]:.6f}, c2={c[1]:.2e}")
print(f"  c2가 0이면 ε*_stat 성공 — c2={c[1]:.2e}")

# 그래프
fig, axes = plt.subplots(3, 2, figsize=(16, 12))

ax = axes[0][0]
ax.plot(times, xcoms, 'r-', lw=2, label='MuJoCo')
ax.plot(times, xcom_an, 'b--', lw=1.5, label='해석')
ax.set_ylabel('x_CoM [mm]'); ax.set_title('x_CoM 비교', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2)

ax = axes[0][1]
ax.plot(times, deltas, 'r-', lw=2, label='δ (MuJoCo)')
ax.axhline(0, color='k', lw=0.3)
ax.set_ylabel('δ [deg]'); ax.set_title('접힘각 δ 유지', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2)

ax = axes[1][0]
ax.plot(times, phis, 'r-', lw=2, label='φ MuJoCo')
ax.plot(times, np.degrees(phi_an), 'b--', lw=1.5, label='φ 해석')
ax.set_ylabel('φ [deg]'); ax.set_title('줄 각도 φ', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2)

ax = axes[1][1]
ax.plot(times, betas, 'r-', lw=2, label='β MuJoCo')
ax.plot(times, np.degrees(beta_an), 'b--', lw=1.5, label='β 해석')
ax.set_ylabel('β [deg]'); ax.set_title('CoM 기울기 β', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2)

ax = axes[2][0]
ax.plot(times, ctrls, 'g-', lw=2)
ax.set_ylabel('토크 [N·m]'); ax.set_xlabel('시간 [s]')
ax.set_title('제어 토크', fontweight='bold'); ax.grid(True, alpha=0.2)

ax = axes[2][1]
ax.plot(times, alphas, '-', lw=1.5, label='α')
ax.plot(times, thetas, '-', lw=1.5, label='θ')
ax.set_ylabel('각도 [deg]'); ax.set_xlabel('시간 [s]')
ax.set_title('α, θ', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.2)

fig.suptitle(f'진단: L1={L1*100:.0f}cm L2={L2*100:.0f}cm R={R*100:.0f}cm\n'
             f'ε*_stat={es:.3f}, Mt={params["Mt"]*1000:.0f}g, h={h*100:.1f}cm',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save = os.path.join(os.path.dirname(__file__), '..', 'docs', 'small_robot_diagnostic.png')
plt.savefig(save, dpi=150, bbox_inches='tight')
print(f"\n  그래프 저장: {save}")

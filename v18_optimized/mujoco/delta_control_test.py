# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
δ 제어 영향 진단: 세 가지 조건 비교
1) δ=0 PD 고정 (현재 코드)
2) δ 자유 (ctrl=0, 힘 풀림)
3) δ=δ_fold PD 고정 (접힌 각도 유지)
"""
import numpy as np, mujoco, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_robot_scan_v2 import (
    make_robot, mode_analysis_param, eps_star_stat_param,
    generate_mjcf_param, get_state, GRAV, DT
)

# 테스트 케이스: 60cm 로봇
L1, L2, R = 0.30, 0.30, 0.40
robot = make_robot(L1, L2)
om, lam, v1, v2, u1, u2 = mode_analysis_param(robot, R)
h = robot['h']; p1 = robot['p1']; p2 = robot['p2']; Mt = robot['Mt']; ps = robot['ps']
es = eps_star_stat_param(robot, R)

print(f"로봇: L1={L1*100:.0f}cm L2={L2*100:.0f}cm R={R*100:.0f}cm")
print(f"Mt={Mt*1000:.0f}g h={h*100:.1f}cm ω={om:.2f} λ={lam:.2f} ε*_stat={es:.3f}")
print(f"p1={p1:.4f} p2={p2:.4f} ps={ps:.4f}")

# 정지 조건에서 ε*_stat 접기
beta_i = np.radians(2)
eps = es
phi_after = h * (1 + eps) * beta_i / R
beta_after = -eps * beta_i

# δ_fold 계산: β가 beta_i에서 beta_after로 바뀌려면
# β = (p1*α + p2*θ)/(Mt*h), δ = θ - α
# β_after = (p1*α + p2*(α+δ_fold))/(Mt*h) = α + (p2/ps)*δ_fold (approx)
# 접기 전: α = β_i (δ=0), 접기 후: β_after = β_i + (p2/ps)*δ_fold
# → δ_fold = (beta_after - beta_i) * ps / p2
delta_fold = (beta_after - beta_i) * ps / p2

print(f"\n접기 후 상태:")
print(f"  phi_after = {np.degrees(phi_after):.3f}°")
print(f"  beta_after = {np.degrees(beta_after):.3f}°")
print(f"  delta_fold = {np.degrees(delta_fold):.3f}°")

# MuJoCo 모델 생성
xml = generate_mjcf_param(R, robot['M1'], robot['M2'], robot['L1'], robot['L2'],
                           robot['BD'], robot['M_HEAD'], robot['HEAD_R'],
                           robot['LOWER_W'], robot['UPPER_W'], robot['LC1'], robot['I1'],
                           robot['L_POLE'], robot['POLE_H'], robot['B_THETA'],
                           robot['TAU_MAX'], robot['ROPE_MASS'], robot['POLE_RADIUS'],
                           robot['POLE_TOP_R'], robot['ROPE_RADIUS'],
                           robot['HIP_R'], robot['HIP_H'])
model = mujoco.MjModel.from_xml_string(xml)

KP = 10000; KD = 200; CTRL_MAX = 50000

# 세 가지 조건 실행
conditions = [
    ("delta=0 PD 고정 (현재)", 0, True, 0.0),
    ("delta 자유 (제어 없음)", 0, False, 0.0),
    ("delta=delta_fold PD 고정", delta_fold, True, delta_fold),
]

sim_t = 2.0; rec_dt = 0.002
all_traces = {}

for cond_name, init_delta, do_control, target_delta in conditions:
    data = mujoco.MjData(model)
    
    ra = model.jnt_qposadr[model.joint('phi_Y').id]
    aa = model.jnt_qposadr[model.joint('ankle').id]
    ha = model.jnt_qposadr[model.joint('hip').id]
    try: rb = model.jnt_qposadr[model.joint('rope_b_Y').id]
    except: rb = None
    va = model.jnt_dofadr[model.joint('phi_Y').id]
    va_a = model.jnt_dofadr[model.joint('ankle').id]
    
    # 초기 조건 — 현재 코드와 동일 방식
    phi_mj = -phi_after
    alpha_after_val = beta_after  # β ≈ α when δ=0
    
    if init_delta == 0:
        # 방식 A: 접기를 α 초기값에 인코딩, δ=0
        data.qpos[ra] = phi_mj
        data.qpos[aa] = alpha_after_val - phi_mj
        data.qpos[ha] = 0
    else:
        # 방식 B: α = β_i(접기 전), δ = δ_fold
        alpha_init = beta_i
        data.qpos[ra] = -phi_after  # phi는 동일
        data.qpos[aa] = alpha_init - (-phi_after)
        data.qpos[ha] = delta_fold
    
    if rb: data.qpos[rb] = phi_mj
    mujoco.mj_forward(model, data)
    
    n = int(sim_t / DT)
    rec = max(1, int(rec_dt / DT))
    times=[]; phis=[]; alphas=[]; thetas=[]; deltas=[]; betas=[]; xcoms=[]; ctrls=[]
    
    for i in range(n):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > 1.4:
            break
        
        phi_v=x[0]; alpha_v=x[1]; theta_v=x[2]
        delta_v = theta_v - alpha_v
        delta_d = x[5] - x[4]
        
        if do_control:
            ctrl = np.clip(KP*(target_delta - delta_v) - KD*delta_d, -CTRL_MAX, CTRL_MAX)
        else:
            ctrl = 0.0
        data.ctrl[0] = ctrl
        
        if i % rec == 0:
            beta_v = (p1*alpha_v + p2*theta_v) / (Mt*h)
            xcom = (-R*phi_v + h*beta_v) * 1000
            times.append(data.time)
            phis.append(np.degrees(phi_v))
            alphas.append(np.degrees(alpha_v))
            thetas.append(np.degrees(theta_v))
            deltas.append(np.degrees(delta_v))
            betas.append(np.degrees(beta_v))
            xcoms.append(xcom)
            ctrls.append(ctrl)
        
        mujoco.mj_step(model, data)
    
    all_traces[cond_name] = {
        't': np.array(times), 'phi': np.array(phis), 'alpha': np.array(alphas),
        'theta': np.array(thetas), 'delta': np.array(deltas), 'beta': np.array(betas),
        'xcom': np.array(xcoms), 'ctrl': np.array(ctrls),
    }
    
    survived = len(times) > 0 and times[-1] >= 1.9
    max_xcom = np.max(np.abs(xcoms)) if xcoms else 9999
    t_end = times[-1] if times else 0
    print(f"\n  {cond_name}:")
    print(f"    생존={survived} (t_end={t_end:.2f}s) max_xcom={max_xcom:.0f}mm")
    if len(deltas) > 0:
        print(f"    δ: 초기={deltas[0]:.2f}° 최종={deltas[-1]:.2f}° 최대={max(np.abs(deltas)):.2f}°")

# ============================================================
# 그래프: 6개 패널
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(18, 14))

colors = {'delta=0 PD 고정 (현재)': '#E63946',
          'delta 자유 (제어 없음)': '#264653', 
          'delta=delta_fold PD 고정': '#2A9D8F'}

panels = [
    ('phi', 'φ [deg]', '줄 각도 φ'),
    ('alpha', 'α [deg]', '하체 절대각 α'),
    ('theta', 'θ [deg]', '상체 절대각 θ'),
    ('delta', 'δ [deg]', '접힘각 δ = θ - α'),
    ('beta', 'β [deg]', 'CoM 기울기 β'),
    ('xcom', 'x_CoM [mm]', 'CoM 수평 변위'),
]

for idx, (key, ylabel, title) in enumerate(panels):
    ax = axes[idx//2][idx%2]
    for cond_name, tr in all_traces.items():
        if len(tr['t']) > 0:
            ax.plot(tr['t'], tr[key], color=colors[cond_name], lw=1.5, 
                    label=cond_name, alpha=0.9)
    ax.axhline(0, color='k', lw=0.3)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.15)
    if idx >= 4: ax.set_xlabel('시간 [s]', fontsize=10)

fig.suptitle(f'δ 제어 영향 비교: L1={L1*100:.0f}cm L2={L2*100:.0f}cm R={R*100:.0f}cm\n'
             f'정지 β_i=2°, ε*_stat={es:.3f}, δ_fold={np.degrees(delta_fold):.1f}°',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save = os.path.join(os.path.dirname(__file__), '..', 'docs', 'delta_control_comparison.png')
plt.savefig(save, dpi=150, bbox_inches='tight')
print(f"\n  그래프 저장: {save}")

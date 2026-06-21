# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
소형 로봇 히트맵 스캔 — 적응형 PD 게인 (wn=200, zeta=0.7)
대형 모델 scan_eps_gen_heatmap.py와 동일한 형태의 히트맵 출력
"""
import numpy as np, mujoco, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from small_robot_scan_v2 import (
    make_robot, generate_mjcf_param, get_state,
    mode_analysis_param, eps_star_stat_param, eps_star_gen_param,
    GRAV, DT
)

# ============================================================
# 적응형 PD 게인: wn=200, zeta=0.7 → Kp = J_tt*200^2, Kd = 2*0.7*200*J_tt
# ============================================================
WN = 200; ZETA = 0.7

def run_heatmap_trace(robot, R, eps, phi_i, beta_i, pdot_i, bdot_i, sim_t=2.0, rec_dt=0.005):
    """적응형 PD 게인으로 시뮬레이션, x_CoM 시계열 반환"""
    h = robot['h']; p1 = robot['p1']; p2 = robot['p2']; Mt = robot['Mt']
    J_tt = robot['J_tt']
    KP = J_tt * WN**2
    KD = 2 * ZETA * WN * J_tt
    
    xml = generate_mjcf_param(R, robot['M1'], robot['M2'], robot['L1'], robot['L2'],
            robot['BD'], robot['M_HEAD'], robot['HEAD_R'],
            robot['LOWER_W'], robot['UPPER_W'], robot['LC1'], robot['I1'],
            robot['L_POLE'], robot['POLE_H'], robot['B_THETA'],
            robot['TAU_MAX'], robot['ROPE_MASS'], robot['POLE_RADIUS'],
            robot['POLE_TOP_R'], robot['ROPE_RADIUS'], robot['HIP_R'], robot['HIP_H'])
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    
    ra = model.jnt_qposadr[model.joint('phi_Y').id]
    aa = model.jnt_qposadr[model.joint('ankle').id]
    ha = model.jnt_qposadr[model.joint('hip').id]
    try: rb = model.jnt_qposadr[model.joint('rope_b_Y').id]
    except: rb = None
    va = model.jnt_dofadr[model.joint('phi_Y').id]
    va_a = model.jnt_dofadr[model.joint('ankle').id]
    va_h = model.jnt_dofadr[model.joint('hip').id]
    try: vb = model.jnt_dofadr[model.joint('rope_b_Y').id]
    except: vb = None

    # 초기조건
    phi_after = phi_i + h*(1+eps)*beta_i/R
    beta_after = -eps*beta_i
    phi_mj = -phi_after; alpha_after = beta_after
    data.qpos[ra] = phi_mj; data.qpos[aa] = alpha_after - phi_mj; data.qpos[ha] = 0
    if rb: data.qpos[rb] = phi_mj
    phi_dot_mj = -pdot_i; alpha_dot = bdot_i
    data.qvel[va] = phi_dot_mj; data.qvel[va_a] = alpha_dot - phi_dot_mj
    data.qvel[va_h] = 0
    if vb: data.qvel[vb] = phi_dot_mj
    mujoco.mj_forward(model, data)

    CTRL_MAX = robot['TAU_MAX']
    rec_step = max(1, int(rec_dt/DT))
    n = int(sim_t/DT)
    times = []; xcoms = []
    for i in range(n):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > 1.4: break
        delta = x[2] - x[1]; delta_d = x[5] - x[4]
        data.ctrl[0] = np.clip(KP*(0-delta) - KD*delta_d, -CTRL_MAX, CTRL_MAX)
        if i % rec_step == 0:
            beta_v = (p1*x[1] + p2*x[2])/(Mt*h)
            xcom = (-R*x[0] + h*beta_v)*1000
            times.append(data.time); xcoms.append(xcom)
        mujoco.mj_step(model, data)
    return np.array(times), np.array(xcoms)


# ============================================================
# 히트맵 생성
# ============================================================
scan_cases = [
    # (L1, L2, R, label)
    (0.25, 0.25, 0.35, '50cm R=35cm'),
    (0.30, 0.30, 0.40, '60cm R=40cm'),
    (0.40, 0.40, 0.40, '80cm R=40cm'),
    (0.50, 0.50, 0.50, '100cm R=50cm'),
]

beta_i = np.radians(2)
# 정지 출발
phi_i = 0; pdot_i = 0; bdot_i = 0

fig, axes = plt.subplots(len(scan_cases), 1, figsize=(14, 5*len(scan_cases)))
if len(scan_cases) == 1: axes = [axes]

for ci, (L1, L2, R, label) in enumerate(scan_cases):
    robot = make_robot(L1, L2)
    es = eps_star_stat_param(robot, R)
    J_tt = robot['J_tt']
    KP = J_tt * WN**2; KD = 2*ZETA*WN*J_tt
    
    print(f'\n=== {label}: Mt={robot["Mt"]*1000:.0f}g, h={robot["h"]*100:.1f}cm, '
          f'eps*={es:.3f}, Kp={KP:.1f}, Kd={KD:.3f} ===')
    
    # eps 스캔 범위: eps* 중심으로 +-1.5
    eps_center = es
    eps_range = np.arange(max(0.1, eps_center-1.5), eps_center+1.5, 0.025)
    
    sim_t = 2.0; rec_dt = 0.005
    n_t = int(sim_t/rec_dt) + 1
    heatmap = np.full((len(eps_range), n_t), np.nan)
    
    for ei, eps in enumerate(eps_range):
        t, xc = run_heatmap_trace(robot, R, eps, phi_i, beta_i, pdot_i, bdot_i, sim_t, rec_dt)
        if len(xc) > 0:
            heatmap[ei, :len(xc)] = xc
            # 발산 후 NaN으로 채움
            if len(xc) < n_t:
                heatmap[ei, len(xc):] = np.nan
        survived = len(t) > 0 and t[-1] >= 1.9
        if ei % 20 == 0:
            status = 'ALIVE' if survived else f'DEAD@{t[-1]:.2f}s' if len(t)>0 else 'DEAD'
            print(f'  eps={eps:.3f}: {status}')
    
    # 히트맵 그리기
    ax = axes[ci]
    t_axis = np.linspace(0, sim_t, n_t)
    
    # NaN을 큰 값으로 (발산 표시)
    heatmap_vis = heatmap.copy()
    vmax = 300  # mm
    heatmap_vis = np.clip(heatmap_vis, -vmax, vmax)
    
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(t_axis, eps_range, heatmap_vis, cmap='RdBu_r', norm=norm, shading='auto')
    
    # eps* 라인
    ax.axhline(es, color='red', lw=2, ls='-', label=f'eps*_stat={es:.3f}')
    
    # 안정 띠 표시 (2초 생존한 eps 범위)
    survived_eps = []
    for ei, eps in enumerate(eps_range):
        if not np.any(np.isnan(heatmap[ei, -10:])):
            max_xcom = np.max(np.abs(heatmap[ei, :]))
            if max_xcom < 500:
                survived_eps.append(eps)
    
    if len(survived_eps) > 0:
        band_lo = min(survived_eps); band_hi = max(survived_eps)
        ax.axhline(band_lo, color='lime', lw=1, ls='--', alpha=0.7)
        ax.axhline(band_hi, color='lime', lw=1, ls='--', alpha=0.7)
        band_w = band_hi - band_lo
        band_label = f'band=[{band_lo:.2f},{band_hi:.2f}] w={band_w:.2f}'
    else:
        band_label = 'no survivors'
    
    ax.set_ylabel('eps')
    ax.set_title(f'{label} | Mt={robot["Mt"]*1000:.0f}g h={robot["h"]*100:.1f}cm | '
                 f'eps*={es:.3f} | {band_label} | Kp={KP:.1f}', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax, label='x_CoM [mm]', shrink=0.8)

axes[-1].set_xlabel('time [s]')
fig.suptitle(f'소형 로봇 x_CoM 히트맵 (PD: wn={WN}, zeta={ZETA}, 정지출발 bi=2deg)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save_path = os.path.join('..', 'docs', 'small_robot_heatmap_adaptive_pd.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'\n저장: {save_path}')

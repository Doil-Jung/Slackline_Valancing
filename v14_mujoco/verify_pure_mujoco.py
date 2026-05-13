# -*- coding: utf-8 -*-
"""
진짜 MuJoCo 검증: hip만 변경, 나머지는 물리에 맡김
가정: 순간 접기(hip만 변경) + MuJoCo 전물리
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.optimize import brentq
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (generate_mjcf, M1, M2_EFF, LC1, LC2_EFF, L1, GRAV, B_THETA)

g = GRAV; Mt = M1 + M2_EFF
p1 = M1*LC1 + M2_EFF*L1; p2 = M2_EFF*LC2_EFF
h = (p1 + p2) / Mt

def run_pure_mujoco(R, beta0, delta_fold):
    """
    순수 MuJoCo 테스트:
    1. 초기: α=β₀, hip=0, φ=0  (몸이 β₀ 기울어짐)
    2. 순간 접기: hip=δ_fold (다른 건 안 건드림!)
    3. 자유 진화: ctrl=0
    → x_CoM(t) 기록
    """
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    
    rope_adr = model.jnt_qposadr[model.joint('phi_Y').id]
    ankle_adr = model.jnt_qposadr[model.joint('ankle').id]
    hip_adr = model.jnt_qposadr[model.joint('hip').id]
    
    # 초기: β₀ 기울임
    data.qpos[rope_adr] = 0
    data.qpos[ankle_adr] = beta0
    data.qpos[hip_adr] = 0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    # 접기 전 x_CoM
    alpha = data.qpos[ankle_adr]; theta = alpha + data.qpos[hip_adr]
    xcom_before = R*data.qpos[rope_adr] + (p1*alpha + p2*theta)/Mt
    
    # 순간 접기: hip만 변경!
    data.qpos[hip_adr] = delta_fold
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    # 접기 후 x_CoM
    alpha = data.qpos[ankle_adr]; theta = alpha + data.qpos[hip_adr]
    xcom_after = R*data.qpos[rope_adr] + (p1*alpha + p2*theta)/Mt
    
    # 자유 진화
    w = np.sqrt(g/(R+h)); Tq = np.pi/(2*w)
    times = []; xcoms = []; phis = []; betas = []
    
    for i in range(int(Tq*1.5/dt)):
        phi = data.qpos[rope_adr]; alpha = data.qpos[ankle_adr]
        hip = data.qpos[hip_adr]; theta = alpha + hip
        beta = (p1*alpha + p2*theta)/(p1+p2)
        x_com = R*phi + (p1*alpha + p2*theta)/Mt
        
        times.append(data.time)
        xcoms.append(x_com)
        phis.append(phi); betas.append(beta)
        
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)
    
    return (np.array(times), np.array(xcoms), np.array(phis), np.array(betas),
            xcom_before, xcom_after)

# ============================================================
#  여러 R에서 테스트
# ============================================================
beta0 = np.radians(2)
R_list = [0.5, 0.7, 1.0, 1.5, 2.0]

# eps에 해당하는 delta_fold: 기존 접기량 사용
# delta_fold = h*(1+eps)*beta0/c_foot 대신
# 단순히 δ_fold = (1+eps)*beta0 로 설정 (정진자의 오버슈트량)
eps = 0.5
delta_fold = (1+eps)*beta0  # hip angle after fold

fig, axes = plt.subplots(len(R_list), 1, figsize=(14, 4*len(R_list)), sharex=False)

print(f"beta0 = {np.degrees(beta0):.1f}°, delta_fold = {np.degrees(delta_fold):.1f}°")
print(f"eps = {eps}")

for ax, R in zip(axes, R_list):
    w = np.sqrt(g/(R+h)); Tq = np.pi/(2*w)
    times, xcoms, phis, betas, xc_bef, xc_aft = run_pure_mujoco(R, beta0, delta_fold)
    
    ax.plot(times, xcoms*1000, 'b-', linewidth=2, label='x_CoM (MuJoCo)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=xc_bef*1000, color='gray', linestyle='--', linewidth=1, label=f'x_CoM before fold = {xc_bef*1000:.1f}mm')
    ax.axhline(y=xc_aft*1000, color='r', linestyle='--', linewidth=1, label=f'x_CoM after fold = {xc_aft*1000:.1f}mm')
    ax.axvline(x=Tq, color='orange', linestyle=':', linewidth=1, label=f'Tq = {Tq:.3f}s')
    
    # 0 교차 찾기
    crossings = []
    for i in range(len(xcoms)-1):
        if xcoms[i]*xcoms[i+1] < 0:
            t_cross = times[i] - xcoms[i]*(times[i+1]-times[i])/(xcoms[i+1]-xcoms[i])
            crossings.append(t_cross)
            ax.axvline(x=t_cross, color='lime', linewidth=2, alpha=0.7)
    
    cross_str = f", T_cross = {crossings[0]:.3f}s" if crossings else ", no crossing"
    ax.set_ylabel('x_CoM [mm]', fontsize=11)
    ax.set_title(f'R = {R}m  (Tq = {Tq:.3f}s{cross_str})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    
    print(f"R={R}: xCoM_before={xc_bef*1000:.2f}mm, xCoM_after={xc_aft*1000:.2f}mm" +
          (f", T_cross={crossings[0]:.4f}s" if crossings else ", no crossing"))

axes[-1].set_xlabel('Time [s]', fontsize=12)
plt.suptitle(f'Pure MuJoCo Test: Only hip changed, rest = physics\n'
             f'beta0={np.degrees(beta0):.0f}°, delta_fold={np.degrees(delta_fold):.1f}°',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'pure_mujoco_test.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

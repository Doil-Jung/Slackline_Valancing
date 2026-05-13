# -*- coding: utf-8 -*-
"""
메인 식 MuJoCo 독립 검증
각 ε에 대해 MuJoCo에서 φ_new=0이 되는 T를 찾고, 이론 T와 비교
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
from batch_experiment import (generate_mjcf, M1, M2_EFF, LC1, LC2_EFF, L1, DT, GRAV)

g = GRAV; Mt = M1 + M2_EFF
p1 = M1*LC1 + M2_EFF*L1; p2 = M2_EFF*LC2_EFF
h = (p1 + p2) / Mt; c_foot = 0.126
I_eff = M1*(LC1**2 + L1**2/12) + M2_EFF*((L1+LC2_EFF)**2)
lam = np.sqrt(Mt*g*h / I_eff)

def omega(R): return np.sqrt(g/(R+h))
def T_quarter(R): return np.pi/(2*omega(R))

def theory_T(R, eps):
    """메인 식에서 이론 T"""
    w = omega(R); Tq = T_quarter(R)
    def eq(T): return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    try: return brentq(eq, 1e-6, Tq*0.99)
    except: return None

def mujoco_xcom_vs_t(R, eps, beta0, t_max):
    """MuJoCo: 순간 접기 후 자유 진화, 매 시간에서 x_CoM 기록"""
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    
    rope_adr = model.jnt_qposadr[model.joint('phi_Y').id]
    ankle_adr = model.jnt_qposadr[model.joint('ankle').id]
    hip_adr = model.jnt_qposadr[model.joint('hip').id]
    
    # 순간 접기 설정
    delta_fold = h*(1+eps)*beta0/c_foot
    phi0 = h*(1+eps)*beta0/R
    
    # H1 유지: α 조정하여 CoM 불변
    # β = α + p2*δ/(p1+p2), 원래 β=β₀, 접기 후 β=-ε*β₀
    alpha_new = -eps*beta0 - p2*delta_fold/(p1+p2)
    
    data.qpos[rope_adr] = phi0
    data.qpos[ankle_adr] = alpha_new
    data.qpos[hip_adr] = delta_fold
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    # 자유 진화 기록
    times = []; xcoms = []
    steps = int(t_max / dt)
    for i in range(steps):
        phi = data.qpos[rope_adr]
        alpha = data.qpos[ankle_adr]
        hip = data.qpos[hip_adr]
        theta = alpha + hip
        beta = (p1*alpha + p2*theta)/(p1+p2)
        x_com = R*np.sin(phi) + h*np.sin(beta)
        
        times.append(data.time)
        xcoms.append(x_com)
        
        data.ctrl[0] = 0  # 토크 없음
        mujoco.mj_step(model, data)
    
    return np.array(times), np.array(xcoms)

def find_zero_crossing(times, values):
    """x_CoM이 0을 지나는 시점 찾기"""
    crossings = []
    for i in range(len(values)-1):
        if values[i]*values[i+1] < 0:
            # 선형 보간
            t_cross = times[i] - values[i]*(times[i+1]-times[i])/(values[i+1]-values[i])
            crossings.append(t_cross)
    return crossings

# ============================================================
#  메인 계산
# ============================================================
R_list = [0.7, 1.0, 1.5]
beta0 = np.radians(5)
eps_range = np.linspace(0.1, 4.0, 40)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, R in zip(axes, R_list):
    w = omega(R); Tq = T_quarter(R)
    print(f"\n=== R = {R}m (Tq = {Tq:.3f}s) ===")
    
    T_theory_list = []; eps_theory_list = []
    T_mujoco_list = []; eps_mujoco_list = []
    
    for eps in eps_range:
        # 이론 T
        Tth = theory_T(R, eps)
        if Tth:
            T_theory_list.append(Tth)
            eps_theory_list.append(eps)
        
        # MuJoCo T
        try:
            times, xcoms = mujoco_xcom_vs_t(R, eps, beta0, Tq*0.99)
            crossings = find_zero_crossing(times, xcoms)
            if crossings:
                T_mujoco_list.append(crossings[0])  # 첫 번째 교차
                eps_mujoco_list.append(eps)
        except:
            pass
    
    print(f"  이론: {len(eps_theory_list)}개, MuJoCo: {len(eps_mujoco_list)}개")
    
    # 그래프
    ax.plot(eps_theory_list, T_theory_list, 'c-', linewidth=3,
            label='Theory (main eq)')
    ax.plot(eps_mujoco_list, T_mujoco_list, 'o', color='lime', markersize=8,
            markeredgecolor='darkgreen', markeredgewidth=1.5,
            label='MuJoCo (x_CoM=0)')
    
    # 수치 비교 출력
    for eps_m, t_m in zip(eps_mujoco_list, T_mujoco_list):
        t_th = theory_T(R, eps_m)
        if t_th:
            err = (t_m - t_th)/t_th*100
            if abs(eps_m - round(eps_m)) < 0.15:
                print(f"  ε={eps_m:.1f}: T_theory={t_th:.4f}, T_MuJoCo={t_m:.4f}, err={err:.1f}%")
    
    ax.set_xlabel('epsilon', fontsize=13)
    ax.set_ylabel('T [s]', fontsize=13)
    ax.set_title(f'R = {R}m', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0, Tq)

plt.suptitle('Main Equation Independent Verification\n'
             'Cyan = Theory prediction | Green = MuJoCo full physics (x_CoM=0 crossing)',
             fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'main_eq_mujoco_verify.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

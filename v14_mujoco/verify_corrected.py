# -*- coding: utf-8 -*-
"""
메인 식 MuJoCo 검증 (수정판)
- c_foot 올바르게 계산: p2/Mt
- 접기 후 상태: α_new = β₀ (ankle 불변), φ와 hip을 CoM 보존에서 계산
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (generate_mjcf, M1, M2_EFF, LC1, LC2_EFF, L1,
                               I2_EFF, GRAV, B_THETA)

g = GRAV; Mt = M1 + M2_EFF
I1 = M1*L1**2/12
p1 = M1*LC1 + M2_EFF*L1; p2 = M2_EFF*LC2_EFF
h = (p1 + p2) / Mt
c_foot = p2 / Mt   # 올바른 값!

print(f"=== 파라미터 ===")
print(f"M1={M1}, M2_EFF={M2_EFF:.1f}, Mt={Mt:.1f}")
print(f"p1={p1:.3f}, p2={p2:.3f}, h={h:.4f}m")
print(f"c_foot = p2/Mt = {c_foot:.4f}m  (이전에 0.126 사용했음)")

lam = np.sqrt(Mt*g*h / (M1*(LC1**2+I1/M1) + M2_EFF*((L1+LC2_EFF)**2)))
print(f"lambda = {lam:.4f}")

def omega(R): return np.sqrt(g/(R+h))

def fold_state_correct(R, eps, beta0):
    """올바른 접기 후 상태 (H1 = CoM 보존, ankle 불변)"""
    phi_new = (1+eps)*h*beta0/R
    alpha_new = beta0   # ankle은 변하지 않음!
    # CoM 보존: R*φ + (p1*α + p2*θ)/Mt = h*β₀
    theta_new = (-eps*h*Mt*beta0 - p1*beta0) / p2
    hip_new = theta_new - alpha_new
    return phi_new, alpha_new, hip_new, theta_new

def mujoco_xcom_trace(R, eps, beta0):
    """MuJoCo 시계열: 올바른 초기 조건으로"""
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep; w = omega(R); Tq = np.pi/(2*w)
    
    rope_adr = model.jnt_qposadr[model.joint('phi_Y').id]
    ankle_adr = model.jnt_qposadr[model.joint('ankle').id]
    hip_adr = model.jnt_qposadr[model.joint('hip').id]
    
    phi0, alpha0, hip0, theta0 = fold_state_correct(R, eps, beta0)
    data.qpos[rope_adr] = phi0
    data.qpos[ankle_adr] = alpha0
    data.qpos[hip_adr] = hip0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    # 초기 x_CoM 확인
    xcom0 = R*phi0 + (p1*alpha0 + p2*theta0)/Mt
    
    times = []; xcoms = []
    for i in range(int(Tq*1.0/dt)):
        phi = data.qpos[rope_adr]; alpha = data.qpos[ankle_adr]
        hip = data.qpos[hip_adr]; theta = alpha + hip
        x_com = R*phi + (p1*alpha + p2*theta)/Mt
        times.append(data.time); xcoms.append(x_com)
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)
    
    return np.array(times), np.array(xcoms), xcom0

def theory_xcom_decoupled(R, eps, beta0, times):
    """분리 이론 x_CoM(t)"""
    w = omega(R)
    return h*beta0*((1+eps)*np.cos(w*times) - eps*np.cosh(lam*times))

def theory_xcom_coupled(R, eps, beta0, times):
    """결합 이론 x_CoM(t) = e_com · exp(At) · x0"""
    M_eq = np.array([
        [Mt*R**2, R*p1, R*p2],
        [R*p1, M1*LC1**2+I1+M2_EFF*L1**2, M2_EFF*L1*LC2_EFF],
        [R*p2, M2_EFF*L1*LC2_EFF, M2_EFF*LC2_EFF**2+I2_EFF]])
    G_mat = np.array([[-Mt*g*R, 0, 0], [0, p1*g, 0], [0, 0, p2*g]])
    D_mat = np.diag([0, 0, -B_THETA])
    Mi = np.linalg.inv(M_eq)
    A = np.zeros((6,6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = Mi @ G_mat; A[3:, 3:] = Mi @ D_mat
    
    phi0, alpha0, hip0, theta0 = fold_state_correct(R, eps, beta0)
    x0 = np.array([phi0, alpha0, theta0, 0, 0, 0])
    ec = np.array([R, p1/Mt, p2/Mt, 0, 0, 0])
    
    xcoms = []
    for t in times:
        xT = expm(A*t) @ x0
        xcoms.append(ec @ xT)
    return np.array(xcoms)

# ============================================================
#  메인 계산
# ============================================================
beta0 = np.radians(1)
eps = 0.5
R_list = [0.5, 0.7, 1.0, 1.5, 2.0]

fig, axes = plt.subplots(len(R_list), 1, figsize=(14, 4*len(R_list)))

print(f"\nbeta0 = {np.degrees(beta0):.1f}°, eps = {eps}")

for ax, R in zip(axes, R_list):
    w = omega(R); Tq = np.pi/(2*w)
    phi0, alpha0, hip0, theta0 = fold_state_correct(R, eps, beta0)
    print(f"\nR={R}: phi0={np.degrees(phi0):.2f}°, alpha0={np.degrees(alpha0):.2f}°, "
          f"hip0={np.degrees(hip0):.2f}°, theta0={np.degrees(theta0):.2f}°")
    
    times_mj, xcoms_mj, xcom0 = mujoco_xcom_trace(R, eps, beta0)
    print(f"  xCoM(0) = {xcom0*1000:.2f}mm (should be {h*beta0*1000:.2f}mm)")
    
    xcom_dec = theory_xcom_decoupled(R, eps, beta0, times_mj)
    xcom_coup = theory_xcom_coupled(R, eps, beta0, times_mj)
    
    ax.plot(times_mj, xcoms_mj*1000, 'b-', linewidth=2.5, label='MuJoCo')
    ax.plot(times_mj, xcom_coup*1000, 'g--', linewidth=2, label='Coupled (exp(AT))')
    ax.plot(times_mj, xcom_dec*1000, 'r:', linewidth=2, label='Decoupled (old)')
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    # 0 교차
    for name, data_arr, color in [('MuJoCo', xcoms_mj, 'blue'),
                                    ('Coupled', xcom_coup, 'green'),
                                    ('Decoupled', xcom_dec, 'red')]:
        for i in range(len(data_arr)-1):
            if data_arr[i]*data_arr[i+1] < 0:
                t_c = times_mj[i] - data_arr[i]*(times_mj[i+1]-times_mj[i])/(data_arr[i+1]-data_arr[i])
                ax.axvline(x=t_c, color=color, linewidth=1, alpha=0.5)
                print(f"  {name}: T_cross = {t_c:.4f}s")
                break
    
    ax.set_ylabel('x_CoM [mm]', fontsize=11)
    ax.set_title(f'R = {R}m (Tq = {Tq:.3f}s)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time [s]', fontsize=12)
plt.suptitle(f'Main Eq Verification (corrected c_foot={c_foot:.3f})\n'
             f'Blue=MuJoCo, Green=Coupled, Red=Decoupled',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'main_eq_corrected.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

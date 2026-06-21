# -*- coding: utf-8 -*-
"""
메인 식 검증 (부호 수정): x_foot = -R*phi
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

lam_sq = Mt*g*h / (M1*(LC1**2+L1**2/12) + M2_EFF*((L1+LC2_EFF)**2))
lam = np.sqrt(lam_sq)
def omega(R): return np.sqrt(g/(R+h))

def xcom_mujoco(phi, alpha, hip, R):
    """올바른 x_CoM = -R*phi + (p1*α + p2*θ)/Mt"""
    theta = alpha + hip
    return -R*phi + (p1*alpha + p2*theta)/Mt

def fold_state(R, eps, beta0):
    """
    접기 후 상태. 
    x_CoM 보존: -R*φ + (p1*α + p2*θ)/Mt = h*β₀
    ankle 불변: α = β₀
    발 오버슈트: x_foot = (1+ε)*h*β₀ → -R*φ = (1+ε)*h*β₀ → φ = -(1+ε)*h*β₀/R
    """
    phi_new = -(1+eps)*h*beta0/R   # 부호 수정!
    alpha_new = beta0
    # CoM 보존: -R*φ + (p1*β₀ + p2*θ)/Mt = h*β₀
    # (1+ε)*h*β₀ + (p1*β₀ + p2*θ)/Mt = h*β₀
    # p2*θ/Mt = -ε*h*β₀ - p1*β₀/Mt
    theta_new = (-eps*h*Mt*beta0 - p1*beta0) / p2
    hip_new = theta_new - alpha_new
    return phi_new, alpha_new, hip_new, theta_new

def run_mujoco_trace(R, eps, beta0):
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep; w = omega(R); Tq = np.pi/(2*w)
    
    rope_adr = model.jnt_qposadr[model.joint('phi_Y').id]
    ankle_adr = model.jnt_qposadr[model.joint('ankle').id]
    hip_adr = model.jnt_qposadr[model.joint('hip').id]
    
    phi0, alpha0, hip0, theta0 = fold_state(R, eps, beta0)
    data.qpos[rope_adr] = phi0
    data.qpos[ankle_adr] = alpha0
    data.qpos[hip_adr] = hip0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    xcom0 = xcom_mujoco(phi0, alpha0, hip0, R)
    
    times = []; xcoms = []; phis = []; betas_arr = []
    for i in range(int(Tq*1.0/dt)):
        phi = data.qpos[rope_adr]; alpha = data.qpos[ankle_adr]
        hip = data.qpos[hip_adr]; theta = alpha + hip
        xc = -R*phi + (p1*alpha + p2*theta)/Mt
        beta = (p1*alpha + p2*theta)/(p1+p2)
        
        times.append(data.time); xcoms.append(xc)
        phis.append(phi); betas_arr.append(beta)
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)
    
    return np.array(times), np.array(xcoms), np.array(phis), np.array(betas_arr), xcom0

def theory_decoupled(R, eps, beta0, times):
    """분리 이론 (부호 수정)"""
    w = omega(R)
    # x_foot(t) = (1+ε)*h*β₀*cos(ωt)  (발이 돌아옴)
    # h*β(t) = -ε*h*β₀*cosh(λt)       (몸이 쓰러짐)
    # x_CoM = x_foot + h*β
    return h*beta0*((1+eps)*np.cos(w*times) - eps*np.cosh(lam*times))

def theory_coupled(R, eps, beta0, times):
    """결합 이론 (부호 수정)"""
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
    
    phi0, alpha0, hip0, theta0 = fold_state(R, eps, beta0)
    x0 = np.array([phi0, alpha0, theta0, 0, 0, 0])
    # 부호 수정: x_CoM = -R*φ + (p1*α + p2*θ)/Mt
    ec = np.array([-R, p1/Mt, p2/Mt, 0, 0, 0])
    
    xcoms = [ec @ expm(A*t) @ x0 for t in times]
    return np.array(xcoms)

# ============================================================
beta0 = np.radians(1); eps = 0.5
R_list = [0.5, 0.7, 1.0, 1.5, 2.0]

fig, axes = plt.subplots(len(R_list), 1, figsize=(14, 4*len(R_list)))

print(f"beta0={np.degrees(beta0):.1f}°, eps={eps}")

for ax, R in zip(axes, R_list):
    w = omega(R); Tq = np.pi/(2*w)
    phi0, alpha0, hip0, theta0 = fold_state(R, eps, beta0)
    
    times_mj, xcoms_mj, phis_mj, betas_mj, xcom0 = run_mujoco_trace(R, eps, beta0)
    xcom_dec = theory_decoupled(R, eps, beta0, times_mj)
    xcom_coup = theory_coupled(R, eps, beta0, times_mj)
    
    print(f"\nR={R}: phi0={np.degrees(phi0):.2f}°, hip={np.degrees(hip0):.2f}°")
    print(f"  xCoM(0)={xcom0*1000:.2f}mm (expected {h*beta0*1000:.2f}mm)")
    
    ax.plot(times_mj, xcoms_mj*1000, 'b-', linewidth=2.5, label='MuJoCo')
    ax.plot(times_mj, xcom_coup*1000, 'g--', linewidth=2, label='Coupled (exp(AT))')
    ax.plot(times_mj, xcom_dec*1000, 'r:', linewidth=2, label='Decoupled')
    ax.axhline(y=0, color='k', linewidth=0.5)
    
    for name, xc_arr, color in [('MuJoCo', xcoms_mj, 'blue'),
                                  ('Coupled', xcom_coup, 'green'),
                                  ('Decoupled', xcom_dec, 'red')]:
        for i in range(len(xc_arr)-1):
            if xc_arr[i]*xc_arr[i+1] < 0:
                tc = times_mj[i]-xc_arr[i]*(times_mj[i+1]-times_mj[i])/(xc_arr[i+1]-xc_arr[i])
                ax.axvline(x=tc, color=color, linewidth=2, alpha=0.6)
                print(f"  {name}: T_cross = {tc:.4f}s")
                break
    
    ax.set_ylabel('x_CoM [mm]', fontsize=11)
    ax.set_title(f'R = {R}m (Tq = {Tq:.3f}s)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time [s]', fontsize=12)
plt.suptitle('Main Eq Verification (SIGN FIXED: x_foot = -R*phi)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'main_eq_sign_fixed.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

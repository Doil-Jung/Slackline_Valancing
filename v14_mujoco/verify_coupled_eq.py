# -*- coding: utf-8 -*-
"""
결합 메인 식 검증: 분리 가정 없이 행렬 지수함수로 풀기
- 결합 모델: x(T) = exp(A·T)·x₀ → x_CoM(T) = 0 인 T 찾기
- MuJoCo와 직접 비교
- 분리 모델(구 메인 식)과도 비교
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
                               I2_EFF, DT, GRAV, B_THETA)

g = GRAV; Mt = M1 + M2_EFF
I1 = M1*L1**2/12
m2e = M2_EFF; lc2e = LC2_EFF; i2e = I2_EFF
p1 = M1*LC1 + m2e*L1; p2 = m2e*lc2e; p3 = m2e*L1*lc2e
h = (p1 + p2) / Mt; c_foot = 0.126
I_eff = M1*(LC1**2 + L1**2/12) + M2_EFF*((L1+LC2_EFF)**2)
lam = np.sqrt(Mt*g*h / I_eff)

def omega(R): return np.sqrt(g/(R+h))

# ============================================================
#  결합 상태 행렬 구성 (q = [φ, α, θ])
# ============================================================
def build_A(R):
    """6x6 상태 행렬 (분리 가정 없음)"""
    M_eq = np.array([
        [Mt*R**2, R*p1, R*p2],
        [R*p1, M1*LC1**2+I1+m2e*L1**2, p3],
        [R*p2, p3, m2e*lc2e**2+i2e]])
    G_mat = np.array([[-Mt*g*R, 0, 0], [0, p1*g, 0], [0, 0, p2*g]])
    D_mat = np.diag([0, 0, -B_THETA])
    
    Mi = np.linalg.inv(M_eq)
    A = np.zeros((6,6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = Mi @ G_mat
    A[3:, 3:] = Mi @ D_mat
    return A

# x_CoM 추출 벡터: x_CoM = R*φ + h*β, β = (p1*α + p2*θ)/Mt/(h/h) 
# 정확히는: x_CoM = R*φ + (p1*α + p2*θ)/Mt * ... 
# x_CoM = R*φ + h*β, β = (p1*α + p2*θ)/(p1+p2)
# → x_CoM = R*φ + (p1*α + p2*θ)/Mt  (since h = (p1+p2)/Mt)
# → x_CoM = R*φ + p1/Mt*α + p2/Mt*θ
def e_com(R):
    """x_CoM = e · x (위치만, 처음 3 성분)"""
    return np.array([R, p1/Mt, p2/Mt, 0, 0, 0])

# ============================================================
#  초기 상태 (접기 직후)
# ============================================================
def fold_state(R, eps, beta0):
    """순간 접기 후 상태: [φ₀, α₀, θ₀, 0, 0, 0]"""
    delta_fold = h*(1+eps)*beta0/c_foot
    phi0 = h*(1+eps)*beta0/R
    # H1 유지하도록 α 조정 (이전과 동일)
    frac_p2 = p2/(p1+p2)
    alpha0 = beta0 - frac_p2*delta_fold
    theta0 = alpha0 + delta_fold
    return np.array([phi0, alpha0, theta0, 0, 0, 0])

# ============================================================
#  T 찾기: 결합 모델
# ============================================================
def find_T_coupled(R, eps, beta0):
    """결합 모델에서 x_CoM(T) = 0 인 T 찾기"""
    A = build_A(R)
    x0 = fold_state(R, eps, beta0)
    ec = e_com(R)
    Tq = np.pi/(2*omega(R))
    
    def f(T):
        xT = expm(A*T) @ x0
        return ec @ xT
    
    # 0에서의 부호
    f0 = f(0.001)
    if abs(f0) < 1e-15: return None
    
    # T를 스윕하여 부호 변화 찾기
    ts = np.linspace(0.001, Tq*0.95, 500)
    fs = [f(t) for t in ts]
    
    crossings = []
    for i in range(len(fs)-1):
        if fs[i]*fs[i+1] < 0:
            T_cross = brentq(f, ts[i], ts[i+1])
            crossings.append(T_cross)
    return crossings[0] if crossings else None

# ============================================================
#  T 찾기: 분리 모델 (구 메인 식)
# ============================================================
def find_T_decoupled(R, eps):
    w = omega(R); Tq = np.pi/(2*w)
    def eq(T): return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    try: return brentq(eq, 1e-6, Tq*0.99)
    except: return None

# ============================================================
#  T 찾기: MuJoCo
# ============================================================
def find_T_mujoco(R, eps, beta0):
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    Tq = np.pi/(2*omega(R))
    
    rope_adr = model.jnt_qposadr[model.joint('phi_Y').id]
    ankle_adr = model.jnt_qposadr[model.joint('ankle').id]
    hip_adr = model.jnt_qposadr[model.joint('hip').id]
    
    x0 = fold_state(R, eps, beta0)
    data.qpos[rope_adr] = x0[0]
    data.qpos[ankle_adr] = x0[1]
    data.qpos[hip_adr] = x0[2] - x0[1]  # hip = θ - α
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    times = []; xcoms = []
    for i in range(int(Tq*0.95/dt)):
        phi = data.qpos[rope_adr]; alpha = data.qpos[ankle_adr]
        hip = data.qpos[hip_adr]; theta = alpha + hip
        x_com = R*phi + (p1*alpha + p2*theta)/Mt
        times.append(data.time); xcoms.append(x_com)
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)
    
    # 0 교차 찾기
    for i in range(len(xcoms)-1):
        if xcoms[i]*xcoms[i+1] < 0:
            t_cross = times[i] - xcoms[i]*(times[i+1]-times[i])/(xcoms[i+1]-xcoms[i])
            return t_cross
    return None

# ============================================================
#  메인 계산
# ============================================================
beta0 = np.radians(1)
R_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5]
eps = 0.5

print(f"beta0 = {np.degrees(beta0):.1f}°, eps = {eps}")
print(f"\n{'R':>5} {'T_decoup':>10} {'T_coupled':>10} {'T_MuJoCo':>10} {'err_dec%':>9} {'err_coup%':>9}")
print("=" * 60)

Rs_ok = []; T_dec = []; T_coup = []; T_mj = []

for R in R_list:
    td = find_T_decoupled(R, eps)
    tc = find_T_coupled(R, eps, beta0)
    tm = find_T_mujoco(R, eps, beta0)
    
    if td and tc and tm:
        err_d = abs(td-tm)/tm*100
        err_c = abs(tc-tm)/tm*100
        print(f"{R:5.1f} {td:10.4f} {tc:10.4f} {tm:10.4f} {err_d:9.1f} {err_c:9.1f}")
        Rs_ok.append(R); T_dec.append(td); T_coup.append(tc); T_mj.append(tm)
    else:
        print(f"{R:5.1f} {'N/A':>10} {tc or 'N/A':>10} {tm or 'N/A':>10}")

# ============================================================
#  그래프
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax = axes[0]
ax.plot(Rs_ok, T_mj, 'o', color='#e74c3c', markersize=12, markeredgecolor='black',
        markeredgewidth=2, label='MuJoCo (full physics)', zorder=5)
ax.plot(Rs_ok, T_coup, 's-', color='#3498db', markersize=8, linewidth=2.5,
        label='Coupled model (exp(AT))')
ax.plot(Rs_ok, T_dec, '^--', color='#95a5a6', markersize=8, linewidth=2,
        label='Decoupled (old main eq)', alpha=0.7)
ax.set_xlabel('R [m]', fontsize=14); ax.set_ylabel('T [s]', fontsize=14)
ax.set_title(f'x_CoM=0 Crossing Time\neps={eps}, beta0={np.degrees(beta0):.0f}°',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

ax = axes[1]
err_d = [abs(td-tm)/tm*100 for td,tm in zip(T_dec, T_mj)]
err_c = [abs(tc-tm)/tm*100 for tc,tm in zip(T_coup, T_mj)]
ax.bar(np.array(Rs_ok)-0.02, err_d, width=0.04, color='#95a5a6', label='Decoupled error', alpha=0.7)
ax.bar(np.array(Rs_ok)+0.02, err_c, width=0.04, color='#3498db', label='Coupled error')
ax.set_xlabel('R [m]', fontsize=14); ax.set_ylabel('Error vs MuJoCo [%]', fontsize=14)
ax.set_title('Error Comparison\nCoupled vs Decoupled', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.suptitle('Coupled Main Equation: No H1/H3/Decoupling Assumptions\nOnly: instantaneous fold + linearization',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'coupled_main_eq.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")

# -*- coding: utf-8 -*-
"""
메인 식 정성적 검증:
  예측4: β₀ 소거 (T_cross가 β₀에 무관)
  예측5: R_max 존재 (큰 R에서 복원 불가)
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
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

def fold_state(R, eps, beta0):
    phi = -(1+eps)*h*beta0/R
    alpha = beta0
    theta = (-eps*h*Mt*beta0 - p1*beta0) / p2
    hip = theta - alpha
    return phi, alpha, hip, theta

def find_T_cross_mujoco(R, eps, beta0, max_time=None):
    """MuJoCo에서 x_CoM=0 교차 시간 찾기"""
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    dt = model.opt.timestep
    w = omega(R); Tq = np.pi/(2*w)
    if max_time is None:
        max_time = Tq * 1.0
    
    rope_adr = model.jnt_qposadr[model.joint('phi_Y').id]
    ankle_adr = model.jnt_qposadr[model.joint('ankle').id]
    hip_adr = model.jnt_qposadr[model.joint('hip').id]
    
    phi0, alpha0, hip0, theta0 = fold_state(R, eps, beta0)
    
    # 너무 큰 각도면 스킵
    if abs(hip0) > np.radians(80):
        return None
    
    data.qpos[rope_adr] = phi0
    data.qpos[ankle_adr] = alpha0
    data.qpos[hip_adr] = hip0
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    
    prev_xcom = None
    for i in range(int(max_time/dt)):
        phi = data.qpos[rope_adr]; alpha = data.qpos[ankle_adr]
        hip = data.qpos[hip_adr]; theta = alpha + hip
        xcom = -R*phi + (p1*alpha + p2*theta)/Mt
        
        if prev_xcom is not None and prev_xcom * xcom < 0:
            # 선형 보간으로 정확한 교차 시점
            t_prev = (i-1)*dt; t_now = i*dt
            t_cross = t_prev - prev_xcom*(t_now-t_prev)/(xcom-prev_xcom)
            return t_cross
        
        # 발산 체크
        if abs(alpha) > np.radians(60):
            return None
        
        prev_xcom = xcom
        data.ctrl[0] = 0
        mujoco.mj_step(model, data)
    
    return None

# ============================================================
#  예측 4: β₀ 소거 검증
# ============================================================
print("=" * 60)
print("  예측 4: T_cross는 β₀에 무관한가?")
print("=" * 60)

R_test = [0.5, 0.7, 1.0]
eps = 0.5
beta0_list = [0.5, 1, 2, 3, 5, 7, 10]  # degrees

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, R in zip(axes, R_test):
    T_crosses = []
    b0_ok = []
    for b0_deg in beta0_list:
        beta0 = np.radians(b0_deg)
        tc = find_T_cross_mujoco(R, eps, beta0)
        if tc is not None:
            T_crosses.append(tc)
            b0_ok.append(b0_deg)
            print(f"  R={R}, β₀={b0_deg:5.1f}°: T_cross = {tc:.4f}s")
        else:
            print(f"  R={R}, β₀={b0_deg:5.1f}°: no crossing")
    
    if T_crosses:
        ax.plot(b0_ok, T_crosses, 'o-', color='#3498db', markersize=8, linewidth=2)
        # 이론값 (β₀와 무관해야 함)
        ax.axhline(y=T_crosses[0], color='red', linestyle='--', linewidth=1.5,
                   label=f'T(β₀→0) = {T_crosses[0]:.4f}s')
        
        # 변동 계산
        if len(T_crosses) > 1:
            variation = (max(T_crosses)-min(T_crosses))/np.mean(T_crosses)*100
            ax.set_title(f'R = {R}m\n변동: {variation:.1f}%',
                        fontsize=13, fontweight='bold',
                        color='green' if variation < 10 else 'orange' if variation < 30 else 'red')
        else:
            ax.set_title(f'R = {R}m', fontsize=13, fontweight='bold')
    
    ax.set_xlabel('β₀ [deg]', fontsize=12)
    ax.set_ylabel('T_cross [s]', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('예측 4 검증: T_cross는 β₀에 무관한가?\n'
             '메인 식: β₀가 소거되므로 T는 β₀에 의존하지 않아야 함',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save1 = os.path.join(os.path.dirname(__file__), '..', 'docs', 'pred4_beta0_independence.png')
plt.savefig(save1, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save1}")

# ============================================================
#  예측 5: R_max 존재 검증
# ============================================================
print("\n" + "=" * 60)
print("  예측 5: 큰 R에서 복원 불가한가?")
print("=" * 60)

R_sweep = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
eps = 0.5
beta0 = np.radians(1)

T_crosses_R = []
R_ok = []

for R in R_sweep:
    w = omega(R); Tq = np.pi/(2*w)
    tc = find_T_cross_mujoco(R, eps, beta0, max_time=Tq*1.5)
    if tc is not None:
        T_crosses_R.append(tc)
        R_ok.append(R)
        status = "✓"
    else:
        status = "✗ (no crossing)"
    print(f"  R={R:5.1f}m: Tq={Tq:.3f}s, T_cross={tc:.4f}s {status}" if tc else
          f"  R={R:5.1f}m: Tq={Tq:.3f}s, {status}")

# 이론 R_max: 메인 식의 해가 존재하지 않는 최소 R
from scipy.optimize import brentq
def has_solution(R, eps):
    w = omega(R)
    Tq = np.pi/(2*w)
    def eq(T): return (1+eps)*np.cos(w*T) - eps*np.cosh(lam*T)
    # t=0에서 양수, Tq에서의 부호 확인
    try:
        val_end = eq(Tq*0.99)
        if val_end < 0:
            return True, brentq(eq, 1e-6, Tq*0.99)
        return False, None
    except:
        return False, None

R_fine = np.linspace(0.3, 15, 200)
R_theory_ok = []; T_theory = []
for R in R_fine:
    ok, T = has_solution(R, eps)
    if ok:
        R_theory_ok.append(R); T_theory.append(T)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# T_cross vs R
ax1.plot(R_ok, T_crosses_R, 'o-', color='#3498db', markersize=8, linewidth=2,
         label='MuJoCo T_cross')
if R_theory_ok:
    ax1.plot(R_theory_ok, T_theory, '--', color='#e74c3c', linewidth=2,
             label='Decoupled T (이론)')
ax1.set_xlabel('R [m]', fontsize=13)
ax1.set_ylabel('T_cross [s]', fontsize=13)
ax1.set_title('T_cross vs R\n(MuJoCo vs 분리 이론)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# T_cross / Tq 비율 (안전 여유도)
ratios = [tc / (np.pi/(2*omega(R))) for tc, R in zip(T_crosses_R, R_ok)]
ax2.plot(R_ok, ratios, 's-', color='#2ecc71', markersize=8, linewidth=2)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='T = Tq (한계)')
ax2.set_xlabel('R [m]', fontsize=13)
ax2.set_ylabel('T_cross / Tq', fontsize=13)
ax2.set_title('타이밍 여유도\n(1.0 이하여야 복원 가능)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.suptitle('예측 5 검증: R_max가 존재하는가?',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save2 = os.path.join(os.path.dirname(__file__), '..', 'docs', 'pred5_Rmax.png')
plt.savefig(save2, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save2}")
print("\n✅ 검증 완료!")

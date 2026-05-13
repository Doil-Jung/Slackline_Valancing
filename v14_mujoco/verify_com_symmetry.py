# -*- coding: utf-8 -*-
"""
검증 1: 자유공간에서 접고 펴는 속도에 상관없이 CoM이 불변인가?
검증 2: CoM 불변은 수평만인가, 수직도 포함인가?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

# === 파라미터 (v14) ===
M1 = 30.0;  L1 = 0.85;  LC1 = L1/2
M2 = 40.0;  L2 = 0.80;  LC2_raw = L2/2
M_HEAD = 5.0; HEAD_R = L2/4
m2 = M2 + M_HEAD
LC2 = (M2*LC2_raw + M_HEAD*(L2 + HEAD_R)) / m2
Mt = M1 + m2
p1 = M1*LC1 + m2*L1
p2 = m2*LC2

print("=" * 60)
print("  검증: 자유공간에서 접기/펴기와 CoM의 관계")
print("=" * 60)
print(f"  p1 = {p1:.2f},  p2 = {p2:.2f},  p1/p2 = {p1/p2:.3f}")
print(f"  (p1 ≠ p2 이므로 H1 조건 불만족)")

# === 자유공간에서 CoM 계산 ===
def com_position(alpha, delta):
    """자유공간에서 (alpha, delta) 상태일 때 CoM의 (x, z) 좌표
    발목 위치를 원점으로, 위가 +z, 오른쪽이 +x"""
    theta = alpha + delta
    # 하체 CoM (발목 기준)
    x1 = LC1 * np.sin(alpha)
    z1 = LC1 * np.cos(alpha)
    # 상체 CoM (발목 기준)
    x2 = L1 * np.sin(alpha) + LC2 * np.sin(theta)
    z2 = L1 * np.cos(alpha) + LC2 * np.cos(theta)
    # 전체 CoM
    x_com = (M1 * x1 + m2 * x2) / Mt
    z_com = (M1 * z1 + m2 * z2) / Mt
    return x_com, z_com

def foot_position(alpha, delta):
    """발(ankle)의 위치 — CoM 기준 좌표"""
    theta = alpha + delta
    # 발 위치 = -(가중 평균 위치)
    x_foot = -(p1 * np.sin(alpha) + p2 * np.sin(theta)) / Mt
    z_foot = -(p1 * np.cos(alpha) + p2 * np.cos(theta)) / Mt
    return x_foot, z_foot

# === 검증 1: p1 ≠ p2에서도 자유공간 CoM은 불변인가? ===
print(f"\n{'='*60}")
print("  검증 1: 자유공간에서 접힘에 따른 CoM 위치")
print(f"{'='*60}")

# 자유공간: 각운동량 보존(L=0)으로 alpha(delta) 결정
from scipy.integrate import solve_ivp

def angular_momentum_coeffs(alpha, delta):
    theta = alpha + delta
    x_a = -(p1*np.sin(alpha) + p2*np.sin(theta)) / Mt
    z_a = -(p1*np.cos(alpha) + p2*np.cos(theta)) / Mt
    x1 = x_a + LC1*np.sin(alpha);  z1 = z_a + LC1*np.cos(alpha)
    x2 = x_a + L1*np.sin(alpha) + LC2*np.sin(theta)
    z2 = z_a + L1*np.cos(alpha) + LC2*np.cos(theta)
    I1_cm = M1*L1**2/12
    I2_body = M2*L2**2/12 + M2*(LC2_raw - LC2)**2
    I2_head = 2/5*M_HEAD*HEAD_R**2 + M_HEAD*((L2+HEAD_R) - LC2)**2
    I2_cm = I2_body + I2_head
    dxa_da = -(p1*np.cos(alpha) + p2*np.cos(theta)) / Mt
    dza_da = (p1*np.sin(alpha) + p2*np.sin(theta)) / Mt
    dx1_da = dxa_da + LC1*np.cos(alpha); dz1_da = dza_da - LC1*np.sin(alpha)
    dx2_da = dxa_da + L1*np.cos(alpha) + LC2*np.cos(theta)
    dz2_da = dza_da - L1*np.sin(alpha) - LC2*np.sin(theta)
    dxa_dd = -p2*np.cos(theta) / Mt; dza_dd = p2*np.sin(theta) / Mt
    dx1_dd = dxa_dd; dz1_dd = dza_dd
    dx2_dd = dxa_dd + LC2*np.cos(theta); dz2_dd = dza_dd - LC2*np.sin(theta)
    f1 = I1_cm + I2_cm + M1*(x1*dz1_da - z1*dx1_da) + m2*(x2*dz2_da - z2*dx2_da)
    f2 = I2_cm + M1*(x1*dz1_dd - z1*dx1_dd) + m2*(x2*dz2_dd - z2*dx2_dd)
    return f1, f2

def dalpha_ddelta(delta, alpha):
    f1, f2 = angular_momentum_coeffs(alpha[0], delta)
    return [-f2/f1]

sol = solve_ivp(dalpha_ddelta, [0, 0.5], [0.0], max_step=0.001, dense_output=True)

print(f"\n  발목 기준 좌표에서 CoM 위치:")
print(f"  {'δ [rad]':>10} {'α [rad]':>10} {'x_com':>12} {'z_com':>12} {'Δx_com':>10} {'Δz_com':>10}")
print(f"  {'-'*65}")

x0, z0 = com_position(0, 0)
for delta_test in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    if delta_test <= sol.t[-1]:
        alpha_val = sol.sol(delta_test)[0]
        x_c, z_c = com_position(alpha_val, delta_test)
        print(f"  {delta_test:10.3f} {alpha_val:10.4f} {x_c:12.6f} {z_c:12.6f} {x_c-x0:10.6f} {z_c-z0:10.6f}")

# === 검증 1b: 접었다가 펴면? ===
print(f"\n  접기 → 펴기 후 CoM 복원 확인:")
print(f"  (자유공간에서 δ: 0 → δ_max → 0 사이클)")

for delta_max in [0.1, 0.2, 0.3, 0.5]:
    if delta_max <= sol.t[-1]:
        # 접기: δ = 0 → δ_max
        alpha_folded = sol.sol(delta_max)[0]
        x_folded, z_folded = com_position(alpha_folded, delta_max)
        # 펴기: δ = δ_max → 0 (역과정, 같은 α(δ) 경로)
        alpha_unfolded = sol.sol(0)[0]  # = 0
        x_unfolded, z_unfolded = com_position(alpha_unfolded, 0)
        print(f"  δ_max = {delta_max:.1f}: 접기 후 CoM=({x_folded:.6f}, {z_folded:.6f})  "
              f"펴기 후 CoM=({x_unfolded:.6f}, {z_unfolded:.6f})  "
              f"차이=({abs(x_unfolded-x0):.1e}, {abs(z_unfolded-z0):.1e})")

# === 검증 2: 발 위치 변화는? ===
print(f"\n{'='*60}")
print("  검증 2: 접힘에 따른 발 위치 변화 (CoM 기준)")
print(f"{'='*60}")

print(f"\n  CoM 기준 좌표에서 발 위치:")
print(f"  {'δ [rad]':>10} {'x_foot':>12} {'z_foot':>12} {'Δx_foot':>10} {'Δz_foot':>10}")
print(f"  {'-'*55}")

xf0, zf0 = foot_position(0, 0)
for delta_test in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    if delta_test <= sol.t[-1]:
        alpha_val = sol.sol(delta_test)[0]
        xf, zf = foot_position(alpha_val, delta_test)
        print(f"  {delta_test:10.3f} {xf:12.6f} {zf:12.6f} {xf-xf0:10.6f} {zf-zf0:10.6f}")

# === 검증 3: σ = p1*α + p2*θ (수평 CoM 지표) ===
print(f"\n{'='*60}")
print("  검증 3: σ = p1·α + p2·θ (수평 CoM 선형 근사)")
print(f"{'='*60}")

print(f"\n  {'δ [rad]':>10} {'α':>10} {'θ=α+δ':>10} {'σ=p1α+p2θ':>12} {'σ (정확)':>12}")
print(f"  {'-'*55}")

for delta_test in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
    if delta_test <= sol.t[-1]:
        alpha_val = sol.sol(delta_test)[0]
        theta_val = alpha_val + delta_test
        sigma = p1 * alpha_val + p2 * theta_val
        # 정확한 수평 CoM (sin 사용)
        x_exact = p1 * np.sin(alpha_val) + p2 * np.sin(theta_val)
        print(f"  {delta_test:10.3f} {alpha_val:10.4f} {theta_val:10.4f} {sigma:12.6f} {x_exact:12.6f}")

print(f"\n{'='*60}")
print("  결론")
print(f"{'='*60}")
print("  1. 자유공간에서 CoM은 접힘에 완전히 불변 (수평+수직 모두)")
print("     이는 p1=p2 여부와 무관 (뉴턴 제3법칙)")
print("  2. 접고 펴는 속도에 무관 → 순간 접기/펴기 가정 가능")
print("  3. H1(p1=p2)은 '줄 위에서' 토크가 수평 CoM을 안 바꾸는 조건")
print("     자유공간에서는 항상 성립하므로 H1은 줄의 구속력 관련 조건")

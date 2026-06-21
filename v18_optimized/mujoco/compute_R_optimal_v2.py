# -*- coding: utf-8 -*-
"""
R_optimal 계산 v2: H2 시간 커플링 조건으로부터 최적 R 유도

핵심 아이디어:
  - 몸을 접어서 발을 이동 → 기울기 회복을 기다림 → 몸을 펴기
  - "기다리는 시간" = 몸 복원 시간 τ_body
  - 정진자의 1/4 주기 = T_rope/4
  - 매칭 조건: τ_body = T_rope/4

유도:
  1. 기울기 β에 비례하여 접는다: δ = k·β
  2. 접으면 발이 c·δ = c·k·β 만큼 이동
  3. 이동한 발 위에서 중력이 복원 토크 생성: τ = M·g·c·k·β
  4. 복원 방정식: I·β̈ = -M·g·c·k·β  (단순조화진동!)
  5. 복원 시간: τ_body = (π/2)·√(I/(M·g·c·k))
  6. 정진자: T_rope/4 = (π/2)·√((R + h)/g)
  7. 매칭: R + h = I/(M·c·k)

  자연스러운 k의 선택: 발을 CoM 바로 아래로 이동시키는 최소 접힘량
    c·k·β = h·β  →  k = h/c
  이를 대입하면:
    R + h = I/(M·c·(h/c)) = I/(M·h)
    R = I/(M·h) - h
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from scipy.integrate import solve_ivp

print("=" * 60)
print("  H2 시간 커플링 조건에 의한 R_optimal 계산")
print("=" * 60)

# ============================================================
#  1단계: 파라미터 설정 (v14 human-scale)
# ============================================================
M1 = 30.0       # 하체 질량 [kg]
L1 = 0.85       # 하체 길이 (발목~힙) [m]
LC1 = L1 / 2    # 하체 CoM 위치 [m]

M2 = 40.0       # 상체 질량 [kg]
L2 = 0.80       # 상체 길이 [m]
LC2_raw = L2 / 2

M_HEAD = 5.0    # 머리 질량 [kg]
HEAD_R = L2 / 4 # 머리 반경 [m]

# 유효 상체 (상체 + 머리)
m2 = M2 + M_HEAD
d2 = LC2_raw
d_h = L2 + HEAD_R
LC2 = (M2 * d2 + M_HEAD * d_h) / m2

# 관성모멘트 (CoM 기준)
I1_cm = M1 * L1**2 / 12
I2_body = M2 * L2**2 / 12 + M2 * (d2 - LC2)**2
I2_head = 2/5 * M_HEAD * HEAD_R**2 + M_HEAD * (d_h - LC2)**2
I2_cm = I2_body + I2_head

# 통합 변수
Mt = M1 + m2    # 전체 질량
p1 = M1 * LC1 + m2 * L1
p2 = m2 * LC2
h_CoM = (p1 + p2) / Mt  # 발목 위 무게중심 높이

g = 9.81

print(f"\n[1단계] 파라미터")
print(f"  M1 = {M1} kg, L1 = {L1} m, LC1 = {LC1} m")
print(f"  m2_eff = {m2} kg, LC2_eff = {LC2:.4f} m")
print(f"  M_total = {Mt} kg")
print(f"  p1 = {p1:.2f}, p2 = {p2:.2f}")
print(f"  h_CoM = {h_CoM:.4f} m")

# ============================================================
#  2단계: 발목 기준 관성모멘트 I_foot 계산
# ============================================================
# 직립 상태에서 발목(foot) 기준 관성모멘트
# 평행축 정리: I_foot = I_cm + M·d²

# 하체: CoM이 발목에서 LC1 위에 있음
I1_foot = I1_cm + M1 * LC1**2

# 상체: CoM이 발목에서 (L1 + LC2) 위에 있음
d_upper = L1 + LC2   # 발목에서 상체 CoM까지 거리
I2_foot = I2_cm + m2 * d_upper**2

I_foot = I1_foot + I2_foot

print(f"\n[2단계] 관성모멘트 (발목 기준)")
print(f"  I1_cm = {I1_cm:.4f} kg·m²")
print(f"  I1_foot = I1_cm + M1·LC1² = {I1_cm:.4f} + {M1*LC1**2:.4f} = {I1_foot:.4f} kg·m²")
print(f"  I2_cm = {I2_cm:.4f} kg·m²")
print(f"  d_upper = L1 + LC2 = {L1} + {LC2:.4f} = {d_upper:.4f} m")
print(f"  I2_foot = I2_cm + m2·d² = {I2_cm:.4f} + {m2}×{d_upper:.4f}² = {I2_cm:.4f} + {m2*d_upper**2:.4f} = {I2_foot:.4f} kg·m²")
print(f"  I_foot = I1 + I2 = {I1_foot:.4f} + {I2_foot:.4f} = {I_foot:.4f} kg·m²")

# ============================================================
#  3단계: 발 이동 계수 c 계산
# ============================================================
# 자유공간에서 접힘 δ당 발의 수평 이동량
# c = L1 · |dα/dδ| at δ=0

# dα/dδ는 각운동량 보존 + CoM 고정 조건에서 유도
def angular_momentum_coeffs(alpha, delta):
    """L = f1·α̇ + f2·δ̇ 에서 f1, f2 계산"""
    theta = alpha + delta
    # 발 위치 (CoM 원점)
    x_a = -(p1 * np.sin(alpha) + p2 * np.sin(theta)) / Mt
    z_a = -(p1 * np.cos(alpha) + p2 * np.cos(theta)) / Mt
    # 각 질점 위치
    x1 = x_a + LC1 * np.sin(alpha)
    z1 = z_a + LC1 * np.cos(alpha)
    x2 = x_a + L1 * np.sin(alpha) + LC2 * np.sin(theta)
    z2 = z_a + L1 * np.cos(alpha) + LC2 * np.cos(theta)
    # dα 미분
    dxa_da = -(p1 * np.cos(alpha) + p2 * np.cos(theta)) / Mt
    dza_da = (p1 * np.sin(alpha) + p2 * np.sin(theta)) / Mt
    dx1_da = dxa_da + LC1 * np.cos(alpha)
    dz1_da = dza_da - LC1 * np.sin(alpha)
    dx2_da = dxa_da + L1 * np.cos(alpha) + LC2 * np.cos(theta)
    dz2_da = dza_da - L1 * np.sin(alpha) - LC2 * np.sin(theta)
    # dδ 미분
    dxa_dd = -p2 * np.cos(theta) / Mt
    dza_dd = p2 * np.sin(theta) / Mt
    dx1_dd = dxa_dd
    dz1_dd = dza_dd
    dx2_dd = dxa_dd + LC2 * np.cos(theta)
    dz2_dd = dza_dd - LC2 * np.sin(theta)
    # 각운동량 계수
    f1 = (I1_cm + I2_cm
          + M1 * (x1 * dz1_da - z1 * dx1_da)
          + m2 * (x2 * dz2_da - z2 * dx2_da))
    f2 = (I2_cm
          + M1 * (x1 * dz1_dd - z1 * dx1_dd)
          + m2 * (x2 * dz2_dd - z2 * dx2_dd))
    return f1, f2

f1_0, f2_0 = angular_momentum_coeffs(0, 0)
dAlpha_dDelta = -f2_0 / f1_0  # 각운동량 보존: L=0 → α̇ = -(f2/f1)·δ̇

# 발 이동 계수: 접힘 δ당 발의 수평 이동
# 발 위치 x_foot = -L1·sin(α) (발목에서 발까지)
# dx_foot/dδ = -L1·cos(α)·(dα/dδ) at α=0
c = L1 * abs(dAlpha_dDelta)

print(f"\n[3단계] 발 이동 계수")
print(f"  f1(0,0) = {f1_0:.4f}")
print(f"  f2(0,0) = {f2_0:.4f}")
print(f"  dα/dδ at δ=0 = -{f2_0:.4f}/{f1_0:.4f} = {dAlpha_dDelta:.6f}")
print(f"  c = L1·|dα/dδ| = {L1}×{abs(dAlpha_dDelta):.4f} = {c:.4f} m/rad")
print(f"  (접힘 1 rad당 발이 {c:.2f} m 이동)")

# ============================================================
#  4단계: 자연스러운 제어 이득 k 계산
# ============================================================
# k: 기울기 β당 접힘량 δ
# "발을 CoM 아래로 이동" 조건: c·k·β = h_CoM·β
k_natural = h_CoM / c

print(f"\n[4단계] 자연 제어 이득 k")
print(f"  조건: c·k = h_CoM (발이 CoM 바로 아래로 이동)")
print(f"  k = h_CoM / c = {h_CoM:.4f} / {c:.4f} = {k_natural:.4f} rad/rad")
print(f"  (기울기 1 rad당 {k_natural:.1f} rad 접힘)")

# ============================================================
#  5단계: 복원 시간 계산
# ============================================================
# τ_body = (π/2)·√(I/(M·g·c·k))
# k = h/c를 대입하면:
# τ_body = (π/2)·√(I/(M·g·h))

omega_body = np.sqrt(Mt * g * h_CoM / I_foot)
tau_body = np.pi / (2 * omega_body)
T_body_full = 2 * np.pi / omega_body

print(f"\n[5단계] 몸 복원 시간")
print(f"  β̈ = -(M·g·c·k / I)·β = -(M·g·h / I)·β")
print(f"  ω_body = √(M·g·h / I)")
print(f"         = √({Mt}×{g}×{h_CoM:.4f} / {I_foot:.4f})")
print(f"         = √({Mt*g*h_CoM:.4f} / {I_foot:.4f})")
print(f"         = √({Mt*g*h_CoM/I_foot:.6f})")
print(f"         = {omega_body:.4f} rad/s")
print(f"  τ_body = π/(2·ω) = {tau_body:.4f} s")
print(f"  T_body = 2π/ω = {T_body_full:.4f} s")

# ============================================================
#  6단계: R_optimal 계산
# ============================================================
# 매칭 조건: T_rope/4 = τ_body
# (π/2)·√((R+h)/g) = (π/2)·√(I/(M·g·h))
# R + h = I/(M·h)
# R = I/(M·h) - h

R_optimal = I_foot / (Mt * h_CoM) - h_CoM

# 검증: 이 R에서의 정진자 주기
T_rope = 2 * np.pi * np.sqrt((R_optimal + h_CoM) / g)
T_rope_quarter = T_rope / 4

print(f"\n[6단계] R_optimal (H2 매칭 조건)")
print(f"  R + h = I / (M·h)")
print(f"  R = I/(M·h) - h")
print(f"    = {I_foot:.4f} / ({Mt}×{h_CoM:.4f}) - {h_CoM:.4f}")
print(f"    = {I_foot:.4f} / {Mt*h_CoM:.4f} - {h_CoM:.4f}")
print(f"    = {I_foot/(Mt*h_CoM):.4f} - {h_CoM:.4f}")
print(f"    = {R_optimal:.4f} m")
print(f"")
print(f"  R_optimal = {R_optimal:.4f} m")
print(f"  R/h_CoM = {R_optimal/h_CoM:.3f}")

# ============================================================
#  7단계: 검증 — 두 시간 스케일 비교
# ============================================================
print(f"\n[7단계] 검증")
print(f"  τ_body    = {tau_body:.4f} s (몸 복원 1/4주기)")
print(f"  T_rope/4  = {T_rope_quarter:.4f} s (줄 진동 1/4주기)")
print(f"  차이      = {abs(tau_body - T_rope_quarter):.6f} s")
print(f"  → {'일치!' if abs(tau_body - T_rope_quarter) < 0.001 else '불일치'}")
print(f"")
print(f"  T_rope 전체 = {T_rope:.4f} s")
print(f"  T_body 전체 = {T_body_full:.4f} s")

# ============================================================
#  8단계: R에 따른 시간 매칭 그래프 데이터
# ============================================================
print(f"\n[8단계] R 변화에 따른 시간 비교")
print(f"{'R [m]':>8} {'R/h':>6} {'T_rope/4 [s]':>12} {'τ_body [s]':>12} {'차이 [s]':>10}")
print("-" * 55)
for R_test in [0.1, 0.2, 0.3, R_optimal, 0.5, 0.7, 0.96, 1.0, 1.5, 2.0, 3.0]:
    T_r = 2 * np.pi * np.sqrt((R_test + h_CoM) / g) / 4
    marker = " ◀ R_opt" if abs(R_test - R_optimal) < 0.001 else ""
    print(f"{R_test:8.3f} {R_test/h_CoM:6.2f} {T_r:12.4f} {tau_body:12.4f} {abs(T_r-tau_body):10.4f}{marker}")

# ============================================================
#  9단계: 물리적 의미 요약
# ============================================================
print(f"\n{'='*60}")
print(f"  물리적 의미 요약")
print(f"{'='*60}")
print(f"  R_optimal = {R_optimal:.4f} m = {R_optimal/h_CoM:.3f} × h_CoM")
print(f"")
print(f"  이 R에서:")
print(f"  - 기울기에 비례하여 몸을 접으면 (δ = {k_natural:.1f}·β)")
print(f"  - 몸이 세워지는 시간 = {tau_body:.3f} s")
print(f"  - 줄이 중앙으로 돌아오는 시간 = {T_rope_quarter:.3f} s")
print(f"  - 두 시간이 일치하여 한 번의 접기-펴기로 균형 복원 가능")
print(f"")
print(f"  핵심: 이 조건은 기울기 크기(β)에 무관 (진폭 독립성)")
print(f"  → 작은 외란이든 큰 외란이든 같은 타이밍으로 제어 가능")

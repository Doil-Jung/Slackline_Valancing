# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
소형 로봇 파라미터 스캔: ε*_gen 실현 가능성 + 안정 마진 평가
==============================================================
독립 변수: L1 (하체 길이), L2 (상체 길이), R (줄 처짐)
질량: 길이에 비례 + 고정 오버헤드 (서보, 전자장치)
크기 제한: L1+L2 ≤ 1.0m, R ≤ 0.6m
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import os

GRAV = 9.81

# ============================================================
# 질량 모델: 길이 비례 + 고정 오버헤드
# ============================================================
RHO_LINEAR = 0.30      # kg/m — 구조재 선밀도 (PLA 3D프린트 기준)
M_SERVO = 0.067         # kg — STS3215 1개
M_ELECTRONICS = 0.10    # kg — ESP32 + IMU + 배터리 + 배선 (상체 탑재)

def compute_params(L1, L2):
    """L1, L2로부터 모든 물리 파라미터 계산"""
    # 질량
    M1 = RHO_LINEAR * L1 + M_SERVO        # 하체 + 발목 서보
    M2_body = RHO_LINEAR * L2 + M_SERVO   # 상체 로드 + 힙 서보
    M_head = 0.03 * (L2 / 0.25)           # 머리 (L2에 비례)
    HEAD_R = L2 / 4
    
    m2_eff = M2_body + M_head + M_ELECTRONICS
    Mt = M1 + m2_eff
    
    # CoM 위치
    LC1 = L1 / 2  # 하체 균일 막대 근사
    I1 = M1 * L1**2 / 12
    
    # 상체 유효 CoM (로드 + 머리 + 전자장치)
    d_body = L2 / 2                         # 로드 CoM
    d_head = L2 + HEAD_R                    # 머리 CoM
    d_elec = L2 * 0.6                       # 전자장치 CoM (상체 중상부)
    
    LC2_eff = (M2_body * d_body + M_head * d_head + M_ELECTRONICS * d_elec) / m2_eff
    
    # 상체 유효 관성모멘트
    I2_body = M2_body * L2**2 / 12 + M2_body * (d_body - LC2_eff)**2
    I2_head = 2/5 * M_head * HEAD_R**2 + M_head * (d_head - LC2_eff)**2
    I2_elec = M_ELECTRONICS * (d_elec - LC2_eff)**2
    I2_eff = I2_body + I2_head + I2_elec
    
    # 정적 모멘트
    p1 = M1 * LC1 + m2_eff * L1
    p2 = m2_eff * LC2_eff
    p3 = m2_eff * L1 * LC2_eff
    ps = p1 + p2
    h = ps / Mt
    
    # 관성 텐서 요소
    J_aa = M1 * LC1**2 + I1 + m2_eff * L1**2
    J_tt = m2_eff * LC2_eff**2 + I2_eff
    J_at = p3
    
    return {
        'M1': M1, 'M2_eff': m2_eff, 'Mt': Mt,
        'L1': L1, 'L2': L2,
        'LC1': LC1, 'LC2_eff': LC2_eff, 'I1': I1, 'I2_eff': I2_eff,
        'p1': p1, 'p2': p2, 'p3': p3, 'ps': ps, 'h': h,
        'J_aa': J_aa, 'J_tt': J_tt, 'J_at': J_at,
        'M_head': M_head, 'HEAD_R': L2/4,
    }

def mode_analysis(params, R):
    """2×2 결합 모드 분석"""
    Mt = params['Mt']; p1 = params['p1']; p2 = params['p2']
    ps = params['ps']; h = params['h']
    J_aa = params['J_aa']; J_tt = params['J_tt']; J_at = params['J_at']
    
    T_inv = np.array([[1, 0, 0],
                       [0, Mt*h/ps, -p2/ps],
                       [0, Mt*h/ps,  p1/ps]])
    
    M = np.array([[Mt*R**2, R*p1, R*p2],
                   [R*p1, J_aa, J_at],
                   [R*p2, J_at, J_tt]])
    G = np.diag([-Mt*GRAV*R, p1*GRAV, p2*GRAV])
    
    Mp = T_inv.T @ M @ T_inv
    Gp = T_inv.T @ G @ T_inv
    
    M22 = Mp[:2, :2]; G22 = Gp[:2, :2]
    A2 = np.linalg.solve(M22, G22)
    
    ev, V = np.linalg.eig(A2)
    idx = np.argsort(ev.real); ev = ev[idx].real; V = V[:,idx].real
    Vi = np.linalg.inv(V)
    
    # 안정 모드 (ev[0] < 0) 확인
    if ev[0] >= 0 or ev[1] <= 0:
        return None  # 올바른 모드 구조가 아님
    
    return {
        'omega': np.sqrt(abs(ev[0])),
        'lam': np.sqrt(abs(ev[1])),
        'v1': V[:,0], 'v2': V[:,1],
        'u1': Vi[0,:], 'u2': Vi[1,:],
        'h': h,
    }

def eps_star_gen(mode, h, R, phi_i, beta_i, pdot_i, bdot_i):
    """ε*_gen 계산"""
    u2 = mode['u2']; lam = mode['lam']
    q_pred = np.array([phi_i + h*beta_i/R + pdot_i/lam, bdot_i/lam])
    e = np.array([h/R, -1.0])
    num = u2 @ q_pred; den = beta_i * (u2 @ e)
    if abs(den) < 1e-15: return np.nan
    return -num / den

def stability_band_width(mode, h, R, beta_i):
    """안정 띠 폭 해석적 추정
    
    Δε ≈ 2·A_max / |β_i · u₂ᵀ · e|
    A_max: e^{2λ}·|A| < x_threshold 를 만족하는 최대 A
    x_threshold = R·|v₂₁| + h·|v₂₂| 로 정규화
    """
    u2 = mode['u2']; lam = mode['lam']
    v2 = mode['v2']
    e = np.array([h/R, -1.0])
    
    # ε 민감도: dc₂/dε = β_i · u₂ᵀ · e
    sensitivity = abs(beta_i * (u2 @ e))
    if sensitivity < 1e-15: return 0.0
    
    # x_CoM에 대한 v₂ 기여: x = R·v₂[0]·η₂ + h·v₂[1]·η₂
    x_scale = abs(R * v2[0] + h * v2[1])
    if x_scale < 1e-15: x_scale = 1e-3
    
    # 2초 후 허용 x_CoM_max = 0.05m (50mm, 소형 로봇 기준)
    x_threshold = 0.05  # m
    A_max = x_threshold / (x_scale * np.exp(2 * lam))
    
    # 띠 폭 = 2 × A_max / (sensitivity/2) = 4·A_max / sensitivity
    band = 4 * A_max / sensitivity
    return band

# ============================================================
# 스캔 실행
# ============================================================
print("=" * 70)
print("  소형 로봇 파라미터 스캔")
print("=" * 70)

# 스캔 격자
L1_range = np.arange(0.10, 0.55, 0.02)   # 10cm ~ 52cm, 2cm 간격
L2_range = np.arange(0.10, 0.55, 0.02)
R_range  = np.arange(0.05, 0.65, 0.02)

# 비정지 초기조건 (소형 기준)
beta_i = np.radians(2)
phi_i  = np.radians(0.5)
pdot_i = np.radians(5)
bdot_i = np.radians(3)

# 결과 저장
results = []

total = 0; valid = 0
for L1 in L1_range:
    for L2 in L2_range:
        if L1 + L2 > 1.0: continue  # 크기 제한
        
        params = compute_params(L1, L2)
        
        for R in R_range:
            total += 1
            mode = mode_analysis(params, R)
            if mode is None: continue
            
            h = params['h']
            
            # 정지 ε*
            r = mode['v1'][0] / mode['v1'][1]
            es_stat = -h / (r * R + h)
            
            # 비정지 ε*_gen
            eg = eps_star_gen(mode, h, R, phi_i, beta_i, pdot_i, bdot_i)
            if np.isnan(eg): continue
            
            # 접기 후 φ₀
            phi_after = phi_i + h * (1 + eg) * beta_i / R
            
            # 안정 띠 폭
            band = stability_band_width(mode, h, R, beta_i)
            
            # 감쇠 시간
            tau_decay = 1 / mode['lam']
            
            # 진동 주기
            T_osc = 2 * np.pi / mode['omega']
            
            # x_CoM max (순수 안정 모드 진폭)
            v1 = mode['v1']
            # c₁ from ε*_gen fold
            Vi = np.linalg.inv(np.column_stack([mode['v1'], mode['v2']]))
            phi0 = phi_i + h*(1+eg)*beta_i/R
            beta0 = -eg*beta_i
            c1 = Vi[0,:] @ np.array([phi0, beta0])
            xcom_amp = abs(c1) * abs(R*v1[0] + h*v1[1])
            
            valid += 1
            results.append({
                'L1': L1, 'L2': L2, 'R': R,
                'M1': params['M1'], 'M2': params['M2_eff'], 'Mt': params['Mt'],
                'h': h,
                'es_stat': es_stat, 'eg': eg,
                'phi_after_deg': np.degrees(phi_after),
                'band': band,
                'tau_decay': tau_decay, 'T_osc': T_osc,
                'omega': mode['omega'], 'lam': mode['lam'],
                'xcom_amp_mm': xcom_amp * 1000,
            })

print(f"  총 조합: {total}, 유효: {valid}")

# numpy 배열로 변환
data = {k: np.array([r[k] for r in results]) for k in results[0].keys()}

# ============================================================
# 필터링: 실현 가능 조건
# ============================================================
feasible = (
    (np.abs(data['phi_after_deg']) < 15) &      # 소각근사 유효
    (data['eg'] > 0) & (data['eg'] < 10) &      # ε*_gen 합리 범위
    (data['band'] > 0.01) &                      # 안정 띠 존재
    (data['xcom_amp_mm'] < 200)                  # x_CoM 진폭 합리적
)

print(f"  실현 가능 조합: {np.sum(feasible)} / {valid}")

# ============================================================
# 그래프 1: L1+L2 고정, R 변화에 따른 주요 지표
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# 대표 (L1, L2) 조합들
combos = [
    (0.20, 0.20, "L1=L2=20cm (총 40cm)"),
    (0.25, 0.25, "L1=L2=25cm (총 50cm)"),
    (0.30, 0.30, "L1=L2=30cm (총 60cm)"),
    (0.20, 0.30, "L1=20, L2=30cm"),
    (0.30, 0.20, "L1=30, L2=20cm"),
    (0.40, 0.40, "L1=L2=40cm (총 80cm)"),
]

colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261', '#E76F51', '#264653']

# 각 패널별 지표
metrics = [
    ('eg', 'ε*_gen', '접기 비율'),
    ('band', '안정 띠 폭 Δε', '안정 마진'),
    ('phi_after_deg', '접기 후 φ₀ [deg]', '소각근사 유효성'),
    ('tau_decay', '감쇠 시간 1/λ [s]', '불안정 모드 감쇠'),
    ('T_osc', '안정 진동 주기 [s]', '진동 시간 스케일'),
    ('xcom_amp_mm', 'x_CoM 진폭 [mm]', 'CoM 진동 범위'),
]

for ax_idx, (key, ylabel, title) in enumerate(metrics):
    ax = axes[ax_idx // 3][ax_idx % 3]
    
    for ci, (l1, l2, label) in enumerate(combos):
        mask = (np.abs(data['L1'] - l1) < 0.005) & (np.abs(data['L2'] - l2) < 0.005)
        if not np.any(mask): continue
        Rs = data['R'][mask]
        vals = data[key][mask]
        ax.plot(Rs, vals, color=colors[ci], lw=2, marker='o', ms=3, label=label)
    
    ax.set_xlabel('R [m]', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.2)
    
    # 특별 표시
    if key == 'phi_after_deg':
        ax.axhline(10, color='red', ls=':', lw=1, alpha=0.5)
        ax.text(0.55, 10.5, '소각근사 한계', fontsize=8, color='red')
    elif key == 'band':
        ax.axhline(0.1, color='red', ls=':', lw=1, alpha=0.5)
        ax.text(0.55, 0.12, '최소 요구 띠폭', fontsize=8, color='red')

fig.suptitle('소형 로봇 파라미터 스캔: R에 따른 주요 지표\n'
             f'(β_i=2°, φ_i=0.5°, φ̇_i=5°/s, β̇_i=3°/s)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
save1 = os.path.join(os.path.dirname(__file__), '..', 'docs', 'small_robot_scan_R.png')
plt.savefig(save1, dpi=150, bbox_inches='tight')
print(f"  저장: {save1}")
plt.close()

# ============================================================
# 그래프 2: L1-L2 히트맵 (R 고정)
# ============================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))

R_fixed_list = [0.15, 0.25, 0.35, 0.45, 0.55, 0.60]

for ri, R_fix in enumerate(R_fixed_list):
    ax = axes2[ri // 3][ri % 3]
    
    mask = np.abs(data['R'] - R_fix) < 0.005
    if not np.any(mask):
        ax.set_title(f'R={R_fix}m — 데이터 없음')
        continue
    
    L1s = np.unique(data['L1'][mask])
    L2s = np.unique(data['L2'][mask])
    
    band_grid = np.full((len(L2s), len(L1s)), np.nan)
    for i, l2 in enumerate(L2s):
        for j, l1 in enumerate(L1s):
            sub = mask & (np.abs(data['L1'] - l1) < 0.005) & (np.abs(data['L2'] - l2) < 0.005)
            if np.any(sub):
                band_grid[i, j] = data['band'][sub][0]
    
    im = ax.imshow(band_grid, origin='lower', aspect='auto',
                   extent=[L1s[0]*100, L1s[-1]*100, L2s[0]*100, L2s[-1]*100],
                   cmap='RdYlGn', vmin=0, vmax=0.5, interpolation='bilinear')
    
    # L1+L2=1.0m 제한선
    ax.plot([10, 90], [90, 10], 'k--', lw=1.5, alpha=0.5, label='L1+L2=1m')
    
    plt.colorbar(im, ax=ax, label='안정 띠 폭 Δε', shrink=0.8)
    ax.set_xlabel('L1 [cm]', fontsize=10)
    ax.set_ylabel('L2 [cm]', fontsize=10)
    ax.set_title(f'R = {R_fix*100:.0f}cm', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)

fig2.suptitle('안정 띠 폭 Δε: L1-L2 히트맵 (R별)\n'
              '녹색: 넓은 안정 띠 (안정화 용이), 빨강: 좁은 띠 (안정화 어려움)',
              fontsize=13, fontweight='bold')
plt.tight_layout()
save2 = os.path.join(os.path.dirname(__file__), '..', 'docs', 'small_robot_scan_L1L2.png')
plt.savefig(save2, dpi=150, bbox_inches='tight')
print(f"  저장: {save2}")
plt.close()

# ============================================================
# 그래프 3: 종합 평가 — 실현 가능 영역
# ============================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))

# (a) ε*_gen vs 안정 띠 폭 (모든 조합)
ax = axes3[0]
sc = ax.scatter(data['eg'][feasible], data['band'][feasible],
                c=data['Mt'][feasible]*1000, cmap='viridis', s=5, alpha=0.5)
ax.scatter(data['eg'][~feasible], data['band'][~feasible],
           c='lightgray', s=2, alpha=0.2, label='실현 불가')
plt.colorbar(sc, ax=ax, label='총 질량 [g]')
ax.set_xlabel('ε*_gen', fontsize=11)
ax.set_ylabel('안정 띠 폭 Δε', fontsize=11)
ax.set_title('ε*_gen vs 안정 띠 폭', fontsize=12, fontweight='bold')
ax.axhline(0.1, color='red', ls=':', lw=1)
ax.set_xlim(0, 8)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.2)
ax.legend(fontsize=8)

# (b) R vs 안정 띠 폭 (L1+L2별 색상)
ax = axes3[1]
Ltotal = data['L1'] + data['L2']
for lt_min, lt_max, label, col in [
    (0.3, 0.45, '30~45cm', '#E63946'),
    (0.45, 0.60, '45~60cm', '#F4A261'),
    (0.60, 0.80, '60~80cm', '#2A9D8F'),
    (0.80, 1.01, '80~100cm', '#264653'),
]:
    mask2 = feasible & (Ltotal >= lt_min) & (Ltotal < lt_max)
    if np.any(mask2):
        ax.scatter(data['R'][mask2], data['band'][mask2],
                   c=col, s=8, alpha=0.4, label=label)

ax.set_xlabel('R [m]', fontsize=11)
ax.set_ylabel('안정 띠 폭 Δε', fontsize=11)
ax.set_title('R vs 안정 띠 폭 (총 길이별)', fontsize=12, fontweight='bold')
ax.axhline(0.1, color='red', ls=':', lw=1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# (c) 상위 20개 최적 조합
ax = axes3[2]
# 점수: band × (band > 0.1) × (φ < 15°) / (1 + eg/5)
score = data['band'] * feasible.astype(float) / (1 + data['eg'] / 5)
top_idx = np.argsort(-score)[:20]

print(f"\n  === 상위 20개 최적 파라미터 조합 ===")
print(f"  {'순위':>3} | {'L1':>5} {'L2':>5} {'R':>5} | {'Mt':>5} {'h':>5} | {'eg':>6} {'Δε':>6} {'φ₀':>6} {'τ':>5} {'xcom':>6}")
print(f"  {'-'*80}")

for rank, i in enumerate(top_idx):
    r = results[i]
    marker = '★' if score[i] > 0 else ' '
    print(f"  {rank+1:3d} | {r['L1']*100:5.0f} {r['L2']*100:5.0f} {r['R']*100:5.0f} | "
          f"{r['Mt']*1000:5.0f} {r['h']*100:5.1f} | "
          f"{r['eg']:6.2f} {r['band']:6.3f} {r['phi_after_deg']:6.1f} "
          f"{r['tau_decay']:5.3f} {r['xcom_amp_mm']:6.1f} {marker}")

# 상위 20개 시각화
ax.barh(range(20), [data['band'][i] for i in top_idx], color='#457B9D', alpha=0.7)
labels = [f"L1={data['L1'][i]*100:.0f} L2={data['L2'][i]*100:.0f} R={data['R'][i]*100:.0f}" 
          for i in top_idx]
ax.set_yticks(range(20))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('안정 띠 폭 Δε', fontsize=11)
ax.set_title('상위 20개 파라미터 (점수순)', fontsize=12, fontweight='bold')
ax.axvline(0.1, color='red', ls=':', lw=1)
ax.invert_yaxis()
ax.grid(True, alpha=0.2, axis='x')

fig3.suptitle('종합 평가: 실현 가능 영역과 최적 파라미터',
              fontsize=13, fontweight='bold')
plt.tight_layout()
save3 = os.path.join(os.path.dirname(__file__), '..', 'docs', 'small_robot_scan_summary.png')
plt.savefig(save3, dpi=150, bbox_inches='tight')
print(f"  저장: {save3}")
plt.close()

print(f"\n  완료!")

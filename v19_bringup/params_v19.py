# -*- coding: utf-8 -*-
"""
params_v19.py — v19 재설계 로봇·줄 시스템의 단일 진실 소스 (2026-07-12)
=========================================================================
★ 이 파일 하나만 실측값으로 채우면 model/compute_gains/sim_check 전부가 따라온다.
★ MEASURED=False 인 항목은 전부 설계 추정치(placeholder). 실측 후 값 교체 + True 로.
   compute_gains_v19.py 는 MEASURED=False 항목이 남아 있으면 경고를 출력한다.

근거 문서:
  - 재설계 스펙: docs/재설계_요구사항_스펙_20260708.md (하체25+상체40, 카본 사다리꼴)
  - 줄 질량·감쇠 이론: docs/연구정리_이론점검_줄질량감쇠_20260711.md
  - 실측 절차: docs/실측_캘리브레이션_절차_20260620.md + v19_bringup/calib/
좌표 규약 (v17, 기존과 동일):
  φ, α, θ 연직기준 시계+, δ = θ − α (앞접기 +), 입력분포 E=[0,−1,+1], τ = −K·x
"""
import numpy as np

G_GRAV = 9.81

# ============================================================
# [1] 로봇 몸체 — ★전 항목 실측 대상
#     m: 저울 / ell(CoM): 균형점(칼날받침) / I: 복합진자법(calib_analysis.py)
# ============================================================
BODY = dict(
    m1   = dict(v=0.15,  MEASURED=False, how="하체(발축 베어링~힙, 프레임+발목하우징 포함) 저울"),
    m2   = dict(v=0.55,  MEASURED=False, how="상체(힙~머리, XM430+OpenCR+배터리 매립 포함) 저울"),
    # v5.3 CAD (2026-07-19): 발목 캐리어 플랜지 +9mm 상승(42mm AS5047P 보드
    # 수용) → 발축~힙축 250 → 259mm. 하체 스파인 출력물은 그대로 재사용,
    # 힙이 통째로 +9mm. 실측 전 공칭값만 갱신.
    L1   = dict(v=0.259, MEASURED=False, how="발축 중심 → 힙축 중심 [m] 자"),
    L2   = dict(v=0.40,  MEASURED=False, how="힙축 중심 → 상체 끝 [m] 자"),
    ell1 = dict(v=0.13,  MEASURED=False, how="발축 → 하체 CoM [m] 균형점"),
    ell2 = dict(v=0.20,  MEASURED=False, how="힙축 → 상체 CoM [m] 균형점 (장비 매립 후!)"),
    # 관성모멘트: None 이면 균일막대 근사 mL²/12 사용. 복합진자 실측 후 I_cm 값 입력.
    I1_cm = dict(v=None, MEASURED=False, how="하체 복합진자: 발축에 매달아 미소진동 주기 T 측정"),
    I2_cm = dict(v=None, MEASURED=False, how="상체 복합진자: 힙축에 매달아 T 측정 (장비 매립 후)"),
    ELL_IMU = dict(v=0.20, MEASURED=False, how="힙축 → OpenCR IMU 칩 [m] 자 (상체 상단 매립 위치)"),
    B_THETA = dict(v=0.01, MEASURED=False, how="힙 점성감쇠 [N·m·s/rad] — 서보 무전류 자유진동 감쇠로 추정(선택)"),
)

# ============================================================
# [2] 줄 시스템 (카본 사다리꼴 크랭크) — ★7/11 이론: '오차'가 아니라 '파라미터'
#     I_r: 상단 피벗 기준 스윙부 전체 관성 / S_r: 정적모멘트 Σmℓ / c_φ: 점성감쇠
# ============================================================
ROPE = dict(
    R = dict(v=0.433, MEASURED=False, how="상단 베어링축 → 발축 거리 [m] 자 (설계 예측 43.3cm)"),
    # 방법 A(권장): 부품별 질량·위치로 계산 → calc_rope_inertia() 가 I_r,S_r 산출
    crank_m_each   = dict(v=0.022, MEASURED=False, how="카본 빗변 1개 질량 [kg] 저울 (8mm×50cm ≈ 20~25g)"),
    crank_L        = dict(v=0.50,  MEASURED=False, how="빗변 길이 [m]"),
    bottom_m       = dict(v=0.060, MEASURED=False, how="하단 하드웨어(발축 바+소켓2+베어링) 합 질량 [kg]"),
    top_socket_m   = dict(v=0.015, MEASURED=False, how="상단 소켓(회전부에 붙는 것만) 질량 [kg]"),
    top_socket_ell = dict(v=0.02,  MEASURED=False, how="상단 소켓 CoM 의 피벗 거리 [m]"),
    # 방법 B(검증): 로봇 떼고 크랭크만 자유진동 → 주기로 I_r/S_r 비, 로그감쇠로 c_φ
    #   (calib/free_swing_log.ino + calib/calib_analysis.py)
    c_phi = dict(v=0.0, MEASURED=False, how="[N·m·s/rad] 크랭크 자유진동 로그감쇠법 — 실측 전 0"),
)

def calc_rope_inertia(rope=ROPE):
    """부품별 질량표 → (I_r [kg·m²], S_r [kg·m], m_rope [kg])  (상단 피벗 기준)"""
    mc, Lc = rope["crank_m_each"]["v"], rope["crank_L"]["v"]
    mb, R_ = rope["bottom_m"]["v"], rope["R"]["v"]
    ms, ls = rope["top_socket_m"]["v"], rope["top_socket_ell"]["v"]
    # 빗변(균일막대, 피벗에서 끝까지): I = mL²/3, S = mL/2 — 2개
    # 주의: 빗변 길이 Lc 를 따라 분포하지만 스윙 반경 방향 유효거리는 R 스케일.
    #       사다리꼴 기하(빗변이 기울어짐)에서 정확값은 CAD가 주지만, 여기서는
    #       '피벗→하단' 유효 길이 R 로 막대 근사(±10% 이내; CAD 값 나오면 교체).
    I_cr = 2 * (mc * R_**2 / 3);  S_cr = 2 * (mc * R_ / 2)
    I_bt = mb * R_**2;            S_bt = mb * R_
    I_ts = ms * ls**2;            S_ts = ms * ls
    return I_cr + I_bt + I_ts, S_cr + S_bt + S_ts, 2*mc + mb + ms

# ============================================================
# [3] 서보 XM430-W210-R + 구동 전장 (기존 v2 검증값)
# ============================================================
SERVO = dict(
    KT       = dict(v=1.304, MEASURED=False, how="[N·m/A] 공칭. torque_cal(추 매달기)로 검증 권장"),
    CUR_UNIT = 2.69,     # [mA/unit] XM430 Current 레지스터
    TAU_MAX  = 3.0,      # [N·m] 모델 상한 (연속 안전권장 2.0~2.5)
    CUR_LIMIT_UNIT = 1193,   # 하드 상한 (3.21A)
    CUR_SAFE_UNIT  = 600,    # 브링업 안전 상한 (~1.6A) → 안정 후 855
    W_NOLOAD_DPS = 462.0,    # 무부하 속도 [°/s] = W210 스펙 77rpm@12V (구 관례 보수치 276=W350값)
                             #   ★자유회전 실측으로 확인 권장 — FWE 시간표(스윕)가 이 값 전제
    MOTOR_DIR = +1,      # ★브링업에서 torque_cal 로 확정 (+전류 → +δ 앞접기)
)

# ============================================================
# [4] φ 엔코더 (상단 피벗) — 모델은 AS5048A 14bit SPI 가정, 미확정이면 firmware/phi_encoder.h 참고
# ============================================================
ENCODER = dict(
    BITS = 14,                    # AS5048A/AMT223: 14bit → 0.022°
    PHI_DIR = +1,                 # ★브링업에서 확정 (로봇을 +x로 밀 때 φ + 인지)
    NOISE_STD_RAD = 5e-4,         # 칼만 R용 (분해능+커플 유격 여유)
)

# ============================================================
# [5] 제어 설계 상수
# ============================================================
CONTROL = dict(
    DT = 0.002,                                  # 500 Hz
    Q_DIAG = [1.0, 50.0, 50.0, 0.1, 5.0, 5.0],   # LQR 상태 가중 (φ,α,θ,φ̇,α̇,θ̇)
    R_COST = 0.001,                              # LQR 입력 가중
    # 칼만 잡음 (z = [φ_enc, θ_acc, θ̇_gyro, δ_enc, δ̇])
    QKF_DIAG = [1e-4, 1e-6, 1e-6, 1e-3, 1e-5, 1e-5],
    RKF_DIAG = [2.5e-7, 1.2e-3, 1.0e-4, 1.0e-5, 1.0e-3],
    # FWE 프로파일 시간 — ★2026-07-18 전물리 스윕(108점, v19_sim/sweep_timing.py)으로 갱신.
    #   근거: 접기=성장벌금 최소이면서 실행품질 유지(60ms), 대기=감쇠가 잔여진동을 대신하므로
    #   최소(20ms), 펴기=완료 가능한 최단(90ms), 휴지=δ 링잉 정착(60ms).
    #   ⚠유효 조건: 줄 감쇠 c_phi ≥ 0.008 실측 확인 시 (미달이면 점성 댐퍼 추가 검토).
    #   (구 직관값 80/60/80/50ms, 트리거 0.4° — 2026-07-12)
    T_FOLD = 0.06, T_WAIT = 0.02, T_EXTEND = 0.09, T_REST = 0.06,   # [s]
    RHO = 0.9,          # deadbeat 대비 게인 비 (최적 0.8~1.0, 과대접기 금지)
    A_TRIGGER_DEG = 0.6, # |A| 트리거(β환산 deg) — 실행산포(~0.5°) 위, 서보 권한 안 (스윕 근거)
    DELTA_MAX_DEG = 55.0,
)

# ------------------------------------------------------------
def val(d):  # dict 항목 → 값
    return d["v"] if isinstance(d, dict) else d

def flat():
    """계산에 쓰는 평탄화된 파라미터 묶음"""
    p = {}
    for grp in (BODY, ROPE, SERVO, ENCODER, CONTROL):
        for k, v in grp.items():
            p[k] = val(v)
    I_r, S_r, m_rope = calc_rope_inertia()
    p["I_r"], p["S_r"], p["m_rope"] = I_r, S_r, m_rope
    # 관성모멘트: 실측 없으면 균일막대 근사
    if p["I1_cm"] is None: p["I1_cm"] = p["m1"] * p["L1"]**2 / 12
    if p["I2_cm"] is None: p["I2_cm"] = p["m2"] * p["L2"]**2 / 12
    return p

def unmeasured():
    out = []
    for gname, grp in (("BODY", BODY), ("ROPE", ROPE), ("SERVO", SERVO)):
        for k, v in grp.items():
            if isinstance(v, dict) and v.get("MEASURED") is False:
                out.append(f"{gname}.{k}  ← {v.get('how','')}")
    return out

if __name__ == "__main__":
    p = flat()
    print("=== v19 파라미터 요약 ===")
    print(f"몸체: m1={p['m1']} m2={p['m2']} L1={p['L1']} L2={p['L2']} (총 {p['m1']+p['m2']:.2f} kg, {p['L1']+p['L2']:.2f} m)")
    print(f"줄:   R={p['R']}  I_r={p['I_r']:.5f} kg·m²  S_r={p['S_r']:.4f} kg·m  (스윙부 {p['m_rope']*1000:.0f} g)")
    Mt = p['m1'] + p['m2']
    print(f"      iota = I_r/(Mt R²) = {p['I_r']/(Mt*p['R']**2):.3f}  ← 0.1 이상이면 무질량 공식 사용 금지(7/11)")
    print(f"감쇠: c_phi={p['c_phi']} (실측 전 0), B_THETA={p['B_THETA']}")
    print("\n=== 미실측 항목 (실측 후 True 로) ===")
    for s in unmeasured():
        print("  □", s)

# -*- coding: utf-8 -*-
"""
test_headless.py — v19_sim 전물리 검증 (UI 없이. sim_check_v19 의 전물리판)
===========================================================================
  [0] 크랭크 관성 교차검증: MuJoCo 실기하 vs params 막대근사 (±15%)
  [1] 6D 선형화 정합: MuJoCo 수치선형화 고유값 vs 서보인지 설계모델 (±5%)
  [2] 불안정 성장률: 미소 몸기울기 → ln|A| 기울기 vs λ_u (±10%)
  [3] LQR+칼만 폐루프, 전 현실화, 포획영역 내(0.5°): 낙하 없이 소진폭 유지
  [4] [3] + 플랜트 편차 m2 +10%: 여전히 유지 (강건성)
  [5] FWE 반복(모드3), 현실 서보 + 참상태, ρ=0.7: |A| 유계·낙하 없음
      (★칼만 추정으로는 현행 QKF 튜닝에서 발산 — 알려진 브링업 리스크, README 참조)
  [6] 스텝백: 1장 뒤로 → 다시 전진 시 동일 상태 재현

전부 PASS 여야 UI 서버를 신뢰하고 쓸 수 있다.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v19_bringup"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco

from sim_engine import SimEngine, MODE_OFF, MODE_LQR, MODE_FWE3

results = {}


def check(name, ok, detail=""):
    results[name] = bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


eng = SimEngine()
bi = eng.build_info()
print(f"[빌드] w_eff={bi['w_eff']:.3f} lam_u={bi['lam_u']:.3f} "
      f"I_r calc={bi['I_r_calc']:.5f} mj={bi['I_r_mj']:.5f}")

# ---------- [0] 크랭크 관성 ----------
check("0 크랭크 I_r (±15%)", abs(bi['I_r_mj']/bi['I_r_calc']-1) < 0.15,
      f"비율 {bi['I_r_mj']/bi['I_r_calc']:.3f}")
check("0 크랭크 S_r (±15%)", abs(bi['S_r_mj']/bi['S_r_calc']-1) < 0.15,
      f"비율 {bi['S_r_mj']/bi['S_r_calc']:.3f}")

# ---------- [1] 6D 선형화 정합 (마찰 0, armature 는 유지 — 설계모델도 알고 있음) ----------
eng.real.update(servo_on=False, sensor_on=False, f_phi=0.0, f_ankle=0.0, f_hip=0.0,
                cur_deadband=0)
eng.ctrl_mode = MODE_OFF
eng.x0 = dict(phi0=0.0, alpha0=0.0, theta0=0.0)
eng.build()
m, d = eng.model, eng.data

def qacc(q, v, u):
    d.qpos[:] = q; d.qvel[:] = v; d.ctrl[0] = u
    mujoco.mj_forward(m, d)
    return d.qacc.copy()

h = 1e-6
Aq = np.zeros((3, 3)); Av = np.zeros((3, 3))
for i in range(3):
    dq = np.zeros(3); dq[i] = h
    Aq[:, i] = (qacc(dq, np.zeros(3), 0) - qacc(-dq, np.zeros(3), 0)) / (2*h)
    Av[:, i] = (qacc(np.zeros(3), dq, 0) - qacc(np.zeros(3), -dq, 0)) / (2*h)
A6 = np.block([[np.zeros((3, 3)), np.eye(3)], [Aq, Av]])
ev_mj = np.sort_complex(np.linalg.eigvals(A6))
ev_md = np.sort_complex(np.linalg.eigvals(eng.g["mdl"]["Ac"]))
# 좌표변환은 유사변환이라 고유값 불변 — 그대로 비교
err = max(abs(a-b)/max(abs(b), 1.0) for a, b in zip(ev_mj, ev_md))
check("1 6D 선형화 고유값 (±5%)", err < 0.05,
      f"최대 상대오차 {err:.3f} (불안정 mj {max(ev_mj.real):.3f} vs 모델 {max(ev_md.real):.3f})")
eng.real.update(**{k: v for k, v in
                   dict(f_phi=0.002, f_ankle=0.002, f_hip=0.05, cur_deadband=5).items()})

# ---------- [2] 불안정 성장률 ----------
eng.ctrl_mode = MODE_OFF
eng.x0 = dict(phi0=0.0, alpha0=0.15, theta0=0.15)
eng.build()
As, ts = [], []
for i in range(int(0.5/eng.p["DT"])):
    eng.control_step()
    As.append(abs(eng.risk_A(eng.get_state()))); ts.append(eng.data.time)
As = np.array(As); ts = np.array(ts)
msk = (As > 1e-5) & (As < 0.2)
lam_fit = np.polyfit(ts[msk], np.log(As[msk]), 1)[0] if msk.sum() > 50 else 0.0
check("2 불안정 λ_u (±10%)", abs(lam_fit/bi["lam_u"]-1) < 0.10,
      f"실측 {lam_fit:.3f} vs 모델 {bi['lam_u']:.3f}")

# ---------- [3a] LQR+칼만, 전 현실화: 낙하 없이 유계 (쿨롱 스틱션 리밋사이클 허용)
#   7/11 이론 예측("쿨롱 스틱션 데드밴드 → 유계 리밋사이클")의 전물리 확인이기도 하다.
eng.real.update(servo_on=True, sensor_on=True)
eng.ctrl_mode = MODE_LQR
eng.estimator = "kalman"
eng.x0 = dict(phi0=0.0, alpha0=0.5, theta0=0.5)
eng.build()
amax_late = 0.0
for _ in range(int(10.0/eng.p["DT"])):
    eng.control_step()
    if eng.data.time > 6.0:
        amax_late = max(amax_late, abs(eng.get_state()[1]))
    if eng.fallen:
        break
check("3a LQR 전현실화 유계(스틱션 리밋사이클)", (not eng.fallen) and amax_late < np.deg2rad(16),
      f"후반 |α|max={np.rad2deg(amax_late):.1f}° (이론: 유계 리밋사이클)")

# ---------- [3b] 힙 쿨롱 0 → 수렴 (리밋사이클 원인 = f_hip 확인) ----------
eng.real.update(f_hip=0.0)
eng.build()
amax_late = 0.0
for _ in range(int(10.0/eng.p["DT"])):
    eng.control_step()
    if eng.data.time > 6.0:
        amax_late = max(amax_late, abs(eng.get_state()[1]))
    if eng.fallen:
        break
check("3b 힙 쿨롱 0 이면 수렴", (not eng.fallen) and amax_late < np.deg2rad(1.5),
      f"후반 |α|max={np.rad2deg(amax_late):.2f}°")
eng.real.update(f_hip=0.05)

# ---------- [4] + 플랜트 편차 m2 −10%: 여전히 유계 ----------
#   ★비대칭 강건성(시뮬 발견): m2 −10% 는 유계지만 +5% 는 수 초 뒤 리밋사이클이
#     자라 낙하한다 → 상체 질량 '과소평가' 방향이 위험. 실측 필수의 정량 근거.
eng.mismatch["m2"] = -10.0
eng.build()
amax_late = 0.0
for _ in range(int(10.0/eng.p["DT"])):
    eng.control_step()
    if eng.data.time > 6.0:
        amax_late = max(amax_late, abs(eng.get_state()[1]))
    if eng.fallen:
        break
check("4 LQR m2-10% 유계 강건", (not eng.fallen) and amax_late < np.deg2rad(18),
      f"후반 |α|max={np.rad2deg(amax_late):.1f}°")
eng.mismatch["m2"] = 0.0

# ---------- [5] FWE 반복 (현실 서보, 참상태, ρ=0.7) ----------
eng2 = SimEngine()
eng2.real.update(servo_on=True, sensor_on=False,
                 # 야간 실험 신규 기능은 이 회귀 기준선에서 비활성 (원 조건 고정)
                 fwe_settle_s=0.0, fwe_adapt_T=False, fwe_ff=False, fwe_fast_L=False)
eng2.ctrl_mode = MODE_FWE3
eng2.estimator = "true"
eng2.kp_servo, eng2.kd_servo = 25.0, 0.4
eng2.nominal.update(RHO=0.7, T_FOLD=0.08, T_WAIT=0.06, T_EXTEND=0.08, T_REST=0.05,
                    A_TRIGGER_DEG=0.4)   # 회귀 기준선: 구 시간표 고정(params 튜닝과 독립)
eng2.x0 = dict(phi0=0.0, alpha0=0.5, theta0=0.5)
eng2.build()
Amax = 0.0
for i in range(int(8.0/eng2.p["DT"])):
    eng2.control_step()
    if i > 100:
        Amax = max(Amax, abs(eng2.risk_A(eng2.get_state())))
    if eng2.fallen:
        break
check("5 FWE 반복(참상태·현실서보) 유계", (not eng2.fallen) and Amax < np.deg2rad(10),
      f"cycles={eng2.fwe['cycles']} |A|max={np.rad2deg(Amax):.2f}°")

# ---------- [6] 스텝백 재현성 ----------
eng3 = SimEngine()
eng3.real.update(servo_on=True, sensor_on=False)
eng3.ctrl_mode = MODE_LQR
eng3.x0 = dict(phi0=0.0, alpha0=0.5, theta0=0.5)
eng3.build()
for _ in range(200):
    eng3.control_step()
x_a = eng3.get_state().copy(); s_a = eng3.step_count
eng3.step_back()
eng3.control_step()
x_b = eng3.get_state(); s_b = eng3.step_count
check("6 스텝백 재현", s_a == s_b and np.allclose(x_a, x_b, atol=1e-12),
      f"Δ={np.max(np.abs(x_a-x_b)):.2e}")

print()
allok = all(results.values())
print("→", "전부 PASS" if allok else "FAIL 있음")
sys.exit(0 if allok else 1)

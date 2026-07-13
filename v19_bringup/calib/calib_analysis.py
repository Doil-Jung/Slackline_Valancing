# -*- coding: utf-8 -*-
"""
calib_analysis.py — 실측 데이터 분석 도구 모음 (v19)
=====================================================
[1] 복합진자법 → 몸체 관성 I_cm
      python calib_analysis.py pendulum --m 0.55 --d 0.20 --T 1.05
      (m: 질량[kg], d: 피벗→CoM[m], T: 미소진동 주기[s] — 10회 왕복 시간/10 권장)
      → 출력 I_cm 을 params_v19.py 의 I1_cm/I2_cm 에 입력

[2] 자유진동 CSV(free_swing_log) → 진동수 + 감쇠 (로그감쇠법)
      python calib_analysis.py freeswing crank_free.csv [--I_pivot 0.014]
      · 크랭크 단독: --I_pivot 에 계산된 I_r 을 주면 c_φ [N·m·s/rad] 산출
        → params_v19.py ROPE.c_phi 에 입력.  ω² ≈ S_r·g/I_r 검증도 출력.
      · 로봇 장착(δ=0, 서보 OFF): ω_measured 를 모델 ω_eff 예측과 비교
        → 조립 후 첫 실물-이론 검증:  python calib_analysis.py verify crank_robot.csv

[3] 모델 검증 (2)의 로봇 장착 진동수를 params 기반 예측과 비교
      python calib_analysis.py verify robot_free.csv
"""
import argparse
import numpy as np


def I_cm_from_pendulum(m, d, T, g=9.81):
    """복합진자: 주기 T → (I_cm, I_pivot)"""
    I_pivot = m * g * d * (T / (2 * np.pi)) ** 2
    return I_pivot - m * d ** 2, I_pivot


def load_csv(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["t_ms"] / 1000.0
    phi = np.deg2rad(d["phi_deg"])
    return t, phi


def find_peaks_simple(t, x):
    """양의 극대점 (이웃보다 큰 점, 잡음 대비 최소 진폭 필터)"""
    thr = 0.1 * np.max(np.abs(x))
    idx = [i for i in range(1, len(x) - 1)
           if x[i] > x[i-1] and x[i] >= x[i+1] and x[i] > thr]
    # 너무 촘촘한(잡음) 피크 제거: 최소 간격 = 전체길이/펄스수 휴리스틱
    out = []
    for i in idx:
        if not out or t[i] - t[out[-1]] > 0.15:
            out.append(i)
    return np.array(out, dtype=int)


def analyze_freeswing(t, phi):
    pk = find_peaks_simple(t, phi)
    if len(pk) < 4:
        raise SystemExit(f"피크 {len(pk)}개 — 데이터 확인 (진동 10회 이상, 's' 타이밍)")
    Tds = np.diff(t[pk])
    Td = float(np.mean(Tds))
    wd = 2 * np.pi / Td
    amps = phi[pk]
    # 로그감쇠: 연속 피크비 평균 (앞쪽 절반만 — 후반은 스틱션 데드밴드 왜곡 가능)
    m = max(3, len(amps) // 2)
    lam = float(np.mean(np.log(amps[:m-1] / amps[1:m])))
    zeta = lam / np.sqrt(4 * np.pi**2 + lam**2)
    wn = wd / np.sqrt(1 - zeta**2)
    return dict(Td=Td, wd=wd, wn=wn, zeta=zeta, n_peaks=len(pk),
                Td_std=float(np.std(Tds)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("pendulum")
    p1.add_argument("--m", type=float, required=True)
    p1.add_argument("--d", type=float, required=True)
    p1.add_argument("--T", type=float, required=True)

    p2 = sub.add_parser("freeswing")
    p2.add_argument("csv")
    p2.add_argument("--I_pivot", type=float, default=None,
                    help="피벗 기준 관성 [kg·m²] (크랭크 단독이면 params 의 I_r)")

    p3 = sub.add_parser("verify")
    p3.add_argument("csv")

    a = ap.parse_args()

    if a.cmd == "pendulum":
        I_cm, I_piv = I_cm_from_pendulum(a.m, a.d, a.T)
        print(f"I_cm = {I_cm:.6f} kg·m²   (I_pivot = {I_piv:.6f})")
        print("→ params_v19.py BODY.I1_cm / I2_cm 에 입력하고 MEASURED=True")

    elif a.cmd == "freeswing":
        t, phi = load_csv(a.csv)
        r = analyze_freeswing(t, phi)
        print(f"피크 {r['n_peaks']}개  주기 Td={r['Td']:.4f}±{r['Td_std']:.4f} s"
              f"  ω_d={r['wd']:.3f} rad/s")
        print(f"감쇠비 ζ={r['zeta']:.4f}  (ω_n={r['wn']:.3f})")
        if a.I_pivot is not None:
            c_phi = 2 * r["zeta"] * r["wn"] * a.I_pivot
            Sr_g = r["wn"]**2 * a.I_pivot
            print(f"→ c_phi = 2ζω_n·I_pivot = {c_phi:.5f} N·m·s/rad"
                  f"   (params_v19 ROPE.c_phi 에 입력)")
            print(f"→ 검증: ω_n²·I_r = {Sr_g:.4f} ≟ S_r·g = params 값과 비교"
                  f" (10% 이상 다르면 질량표/기하 재점검)")
        else:
            print("(--I_pivot 을 주면 c_phi 까지 산출)")

    elif a.cmd == "verify":
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import params_v19 as P
        import model_v19 as MD
        t, phi = load_csv(a.csv)
        r = analyze_freeswing(t, phi)
        mdl = MD.analyze(P.flat(), verbose=False)
        w_pred = mdl["modes"]["w_eff"]
        err = 100 * (r["wn"] - w_pred) / w_pred
        print(f"실측 ω = {r['wn']:.3f} rad/s  vs  모델 ω_eff = {w_pred:.3f} rad/s"
              f"  → 오차 {err:+.1f}%")
        print("±5% 안: 모델·질량표 신뢰 → 게인 이식 진행 /"
              " 밖: R·질량·CoM 실측치 재점검 (첫 실물-이론 검증 항목)")


if __name__ == "__main__":
    main()

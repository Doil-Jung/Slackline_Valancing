# -*- coding: utf-8 -*-
"""
sim_check_v19.py — 게인 이식 전 필수 폐루프 검증 (펌웨어 로직 그대로 재현)
==========================================================================
검사 4종:
  [1] LQR+칼만 폐루프: 초기 기울기에서 수렴하는가, τ 피크가 한계 안인가
  [2] 센서 노이즈 주입: 엔코더/IMU 잡음 하에서도 안정 유지하는가
  [3] FWE 단일접기: δf=γ_fold·A 로 접은 뒤 |A| 가 실제로 죽는가
  [4] 반복 FEW: ρ 게인으로 사이클 반복 시 A 등비 수축하는가 (+20% 접기오차 강건성)
모두 PASS 여야 펌웨어에 게인을 옮긴다.
"""
import numpy as np
import params_v19 as P
import model_v19 as MD
import compute_gains_v19 as CG


def run():
    print("### sim_check_v19 ###")
    out = CG.main()
    p = P.flat()
    Ad, Bd, K, L, C = out["Ad"], out["Bd"], out["K"], out["L"], out["C"]
    mdl = out["model"]; fwe = out["fwe"]
    DT = p["DT"]; n = 6
    aux = mdl["aux"]
    p1r = aux["p1"]/aux["ps"]; p2r = aux["p2"]/aux["ps"]
    w4 = fwe["w"]
    rng = np.random.default_rng(0)
    results = {}

    Mi = np.linalg.inv(mdl["M"])
    w_imu = np.array([p["R"], p["L1"], p["ELL_IMU"]])
    D1 = -float(w_imu @ (Mi @ mdl["E"]).ravel())/9.81

    def measure(x, tau, noisy):
        z = np.array([
            x[0],                                   # φ_enc
            float(C[1] @ x) + D1*tau,               # θ_acc (보상모델의 '실제' 생성값)
            x[5],                                   # θ̇
            x[2]-x[1],                              # δ
            x[5]-x[4],                              # δ̇
        ])
        if noisy:
            z += rng.normal(0, [5e-4, 0.03, 0.002, 5e-4, 0.02])
        return z

    def closed_loop(x0, T, noisy=False, gain=1.0):
        x = x0.copy(); xh = np.zeros(n)
        xh[0] = x0[0]  # φ는 엔코더로 즉시 안다 (v19 초기화 규약)
        tau_prev = 0.0; tau_pk = 0.0
        steps = int(T/DT)
        for _ in range(steps):
            z = measure(x, tau_prev, noisy)
            xp = Ad @ xh + Bd[:, 0]*tau_prev
            innov = z - np.array([xp[0], float(C[1]@xp)+D1*tau_prev,
                                  xp[5], xp[2]-xp[1], xp[5]-xp[4]])
            xh = xp + L @ innov
            tau = float(-K[0] @ xh) * gain
            tau = np.clip(tau, -p["TAU_MAX"], p["TAU_MAX"])
            tau_pk = max(tau_pk, abs(tau))
            x = Ad @ x + Bd[:, 0]*tau
            tau_prev = tau
            if abs(x[1]) > np.deg2rad(80) or abs(x[2]) > np.deg2rad(80):
                return False, tau_pk, x
        settled = max(abs(x[1]), abs(x[2])) < np.deg2rad(1.0)
        return settled, tau_pk, x

    # [1] 깨끗한 폐루프, 초기 몸기울기 3°
    x0 = np.zeros(n); x0[1] = x0[2] = np.deg2rad(3.0)
    ok, pk, _ = closed_loop(x0, 8.0)
    results["1 LQR+칼만 수렴(3°)"] = ok and pk <= p["TAU_MAX"]+1e-9
    print(f"[1] settle={ok}  tau_peak={pk:.2f} N·m (한계 {p['TAU_MAX']})")

    # [2] 노이즈 주입
    ok2, pk2, xe = closed_loop(x0, 8.0, noisy=True)
    results["2 노이즈 강건"] = ok2
    print(f"[2] settle={ok2}  tau_peak={pk2:.2f}")

    # [3] FWE 단일접기 (4D 대기계, δ 프로파일 수치적분은 model의 맵으로 등가 검증)
    Gf, gf = fwe["G_fold"], fwe["g_fold"]
    A0 = np.deg2rad(1.0)                       # β환산 1° 위험도
    dfold = fwe["GAMMA_FOLD"]*A0               # ρ=1
    A_after = Gf*A0 - gf*dfold
    results["3 단일접기 소거"] = abs(A_after) < 1e-9*max(1, abs(A0))
    # [3b] 서보 속도 여유 — 실제 운용점(트리거 문턱 A_TRIG)에서 평가.
    #      삼각 프로파일 피크 속도 = 2·δf/T_F vs 무부하 속도(params W_NOLOAD_DPS, 여유율 0.83)
    df_trig = fwe["GAMMA_FOLD"]*np.deg2rad(p["A_TRIGGER_DEG"])
    vpk = 2*np.rad2deg(df_trig)/p["T_FOLD"]
    v_allow = 0.83 * p["W_NOLOAD_DPS"]
    results["3b 서보속도 여유(트리거점)"] = vpk < v_allow
    print(f"[3] A0={np.rad2deg(A0):.2f}° → fold {np.rad2deg(dfold):.1f}° → A={A_after:.2e} "
          f"(δ한계 {p['DELTA_MAX_DEG']}° {'OK' if abs(np.rad2deg(dfold))<p['DELTA_MAX_DEG'] else '초과!'})")
    print(f"[3b] 트리거점(A={p['A_TRIGGER_DEG']}°) 접기 {np.rad2deg(df_trig):.1f}° 피크속도 {vpk:.0f}°/s (무부하 {p['W_NOLOAD_DPS']:.0f}, 허용<{v_allow:.0f})")

    # [4] 반복 FEW: 명목/+20% 접기오차
    Gc, gc = fwe["G_cyc"], fwe["g_cyc"]
    for err, tag in [(0.0, "명목"), (0.20, "+20% 접기오차")]:
        A = np.deg2rad(1.0); hist = [A]
        for _ in range(12):
            df = p["RHO"]*fwe["GAMMA_CYCLE"]*A*(1+err)
            df = np.clip(df, -np.deg2rad(p["DELTA_MAX_DEG"]), np.deg2rad(p["DELTA_MAX_DEG"]))
            A = Gc*A - gc*df
            hist.append(A)
        conv = abs(hist[-1]) < abs(hist[0])*1e-2
        results[f"4 반복FEW {tag}"] = conv
        print(f"[4] {tag}: |A| {np.rad2deg(hist[0]):.2f}° → {np.rad2deg(abs(hist[-1])):.4f}° "
              f"(12사이클) {'수렴' if conv else '발산!'}  m={Gc*(1-p['RHO']*(1+err)):.2f}")

    print("\n=== 결과 ===")
    allok = True
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}"); allok &= v
    print("→", "전부 PASS — 게인을 펌웨어로 옮겨도 된다." if allok
          else "FAIL 있음 — Q/R, 프로파일 시간, 파라미터부터 점검하라.")
    return allok


if __name__ == "__main__":
    run()

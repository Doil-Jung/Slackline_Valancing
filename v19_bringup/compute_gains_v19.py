# -*- coding: utf-8 -*-
"""
compute_gains_v19.py — v19 게인 일괄 계산 + gains_v19.h 자동 생성
=================================================================
compute_K_measured_v2.py 대비 변경 (★v19 전용):
  1. ★φ 직접 측정 — 상단 엔코더가 z0=φ 를 준다.
     칼만 측정 z = [φ_enc, θ_acc(선형가속 보상), θ̇_gyro, δ_enc, δ̇] (4→5행).
     구버전의 φ 간접 초기추정 핵(C1_PHI/C2_PHI)은 폐기 — 불필요해짐.
  2. ★플랜트에 줄 관성 I_r·S_r, 베어링 감쇠 c_φ 반영 (model_v19).
  3. ★FWE 상수(w4, λ_u, γ_fold, γ_cycle, ε*_damped)까지 함께 출력.
  4. 출력을 stdout 복붙이 아니라 gains_v19.h 파일로 생성해
     firmware/lqr_balance_v19/ 와 firmware/fwe_v19/ 에 자동 복사.
실행: python compute_gains_v19.py   (repo/v19_bringup 에서)
"""
import os
import numpy as np
from scipy.linalg import solve_discrete_are

import params_v19 as P
import model_v19 as MD


def main():
    p = P.flat()
    print("=" * 64)
    print("  v19 게인 계산 (줄 관성·감쇠 포함, φ 직접측정 칼만)")
    print("=" * 64)
    miss = P.unmeasured()
    if miss:
        print(f"⚠ 미실측 파라미터 {len(miss)}건 — 아래는 설계 추정치 기준 게인이다.")
        print("  브링업 예행연습용으로만 쓰고, 실측 후 반드시 재계산할 것!\n")

    mdl = MD.analyze(p)
    Ac, Bc = mdl["Ac"], mdl["Bc"]
    aux = mdl["aux"]
    DT = p["DT"]
    Ad, Bd = MD.discretize(Ac, Bc, DT)
    n = 6

    # ---------------- LQR ----------------
    Ctrb = np.hstack([np.linalg.matrix_power(Ad, k) @ Bd for k in range(n)])
    assert np.linalg.matrix_rank(Ctrb) == n, "비가제어!"
    Pric = solve_discrete_are(Ad, Bd, np.diag(p["Q_DIAG"]), np.array([[p["R_COST"]]]))
    K = (np.linalg.inv(np.array([[p["R_COST"]]]) + Bd.T @ Pric @ Bd) @ (Bd.T @ Pric @ Ad))
    cl = max(abs(e) for e in np.linalg.eigvals(Ad - Bd @ K))
    print(f"\n[LQR] K = [{', '.join(f'{k:.3f}' for k in K[0])}]  폐루프 max|eig|={cl:.6f} "
          f"{'✅' if cl < 1 else '❌'}")

    # ---------------- 칼만 (z 5행: φ, θ_acc보상, θ̇, δ, δ̇) ----------------
    Mi = np.linalg.inv(mdl["M"])
    w_imu = np.array([p["R"], p["L1"], p["ELL_IMU"]])
    wMiG = w_imu @ (Mi @ mdl["G"]); wMiD = w_imu @ (Mi @ mdl["D"])
    wMiE = float(w_imu @ (Mi @ mdl["E"]).ravel())
    m_obs = 5
    C = np.zeros((m_obs, n))
    C[0, 0] = 1.0                                   # ★φ_enc (신규)
    C[1, 2] = 1.0; C[1, 0:3] -= wMiG / 9.81; C[1, 3:6] -= wMiD / 9.81   # θ_acc 보상행
    D1_feed = -wMiE / 9.81                          # innovation의 τ 직결항
    C[2, 5] = 1.0                                   # θ̇
    C[3, 2] = 1.0; C[3, 1] = -1.0                   # δ
    C[4, 5] = 1.0; C[4, 4] = -1.0                   # δ̇
    rank_O = np.linalg.matrix_rank(
        np.vstack([C @ np.linalg.matrix_power(Ad, k) for k in range(n)]))
    Pk = solve_discrete_are(Ad.T, C.T, np.diag(p["QKF_DIAG"]), np.diag(p["RKF_DIAG"]))
    L = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + np.diag(p["RKF_DIAG"]))
    ob = max(abs(e) for e in np.linalg.eigvals(Ad - L @ C))
    print(f"[칼만] z=[φ_enc, θ_acc(보상), θ̇, δ, δ̇]  rank={rank_O}/6 "
          f"{'✅' if rank_O == n else '❌'}  관측기 max|eig|={ob:.6f} {'✅' if ob < 1 else '❌'}")

    # ---------------- FWE 상수 ----------------
    fwe = mdl["fwe"]; md = mdl["modes"]
    tau2unit = 1000.0 / (p["KT"] * p["CUR_UNIT"])
    p1r = aux["p1"] / aux["ps"]; p2r = aux["p2"] / aux["ps"]   # β = p1r·α + p2r·θ

    # ---------------- gains_v19.h 생성 ----------------
    def fmt_mat(name, A, r, c, per=None):
        s = f"const float {name}[{r}][{c}] = {{\n"
        for i in range(r):
            s += "  {" + ", ".join(f"{A[i, j]:.8f}f" for j in range(c)) + "}" + ("," if i < r-1 else "") + "\n"
        return s + "};\n"
    def fmt_vec(name, v):
        return f"const float {name}[{len(v)}] = {{{', '.join(f'{x:.8f}f' for x in v)}}};\n"

    h = []
    h.append("// gains_v19.h — compute_gains_v19.py 가 생성 (직접 수정 금지)")
    h.append(f"// 플랜트: R={p['R']}, I_r={p['I_r']:.5f}, S_r={p['S_r']:.4f}, c_phi={p['c_phi']}, DT={DT}")
    h.append(f"// 미실측 {len(miss)}건 — 실측 후 재생성할 것" if miss else "// 실측 파라미터 기준")
    h.append("#pragma once\n")
    h.append(f"#define GAINS_MEASURED {0 if miss else 1}   // 0이면 펌웨어가 부팅 시 경고")
    h.append(f"const float DT_MODEL   = {DT}f;")
    h.append(f"const float TAU_MAX_NM = {p['TAU_MAX']}f;")
    h.append(f"const float TAU2UNIT   = {tau2unit:.4f}f;")
    h.append(f"const int   CUR_LIMIT_U = {p['CUR_LIMIT_UNIT']};")
    h.append(f"const int   CUR_SAFE_U  = {p['CUR_SAFE_UNIT']};\n")
    h.append(fmt_vec("K_lqr", K[0]))
    h.append(fmt_mat("Ad_mat", Ad, 6, 6))
    h.append(fmt_vec("Bd_vec", Bd[:, 0]))
    h.append(fmt_mat("L_kf", L, 6, 5))
    h.append(fmt_vec("C1_acc", C[1]))           # θ_acc 보상 측정행
    h.append(f"const float D1_FEED = {D1_feed:.8f}f;\n")
    h.append("// --- FWE (A = wᵀ[φ, β, φ̇, β̇], β성분=1 정규화 → A는 β환산 rad) ---")
    h.append(fmt_vec("W4_risk", fwe["w"]))
    h.append(f"const float LAM_U      = {fwe['lu']:.6f}f;   // 감쇠 포함 불안정 성장률")
    h.append(f"const float OMEGA_EFF  = {md['w_eff']:.6f}f;")
    h.append(f"const float R_MODE     = {md['r']:.6f}f;     // 안정모드선 기울기 (참고)")
    h.append(f"const float EPS_STAR_D = {mdl['eps_star']:.6f}f; // 감쇠 포함 ε* (참고)")
    h.append(f"const float GAMMA_FOLD  = {fwe['GAMMA_FOLD']:.6f}f; // 단일접기: δf=ρ·γ·A")
    h.append(f"const float GAMMA_CYCLE = {fwe['GAMMA_CYCLE']:.6f}f; // 완전사이클 deadbeat")
    h.append(f"const float T_FOLD_S={p['T_FOLD']}f, T_WAIT_S={p['T_WAIT']}f, "
             f"T_EXTEND_S={p['T_EXTEND']}f, T_REST_S={p['T_REST']}f;")
    h.append(f"const float RHO_GAIN   = {p['RHO']}f;        // 0.8~1.0 (과대접기 금지)")
    h.append(f"const float A_TRIG_RAD = {np.deg2rad(p['A_TRIGGER_DEG']):.6f}f;")
    h.append(f"const float DELTA_MAX_RAD = {np.deg2rad(p['DELTA_MAX_DEG']):.6f}f;\n")
    h.append("// β 합성 상수: β = P1R·α + P2R·θ  (β̇ 동일)")
    h.append(f"const float P1R = {p1r:.8f}f;")
    h.append(f"const float P2R = {p2r:.8f}f;")
    txt = "\n".join(h) + "\n"

    here = os.path.dirname(os.path.abspath(__file__))
    targets = [os.path.join(here, "firmware", d, "gains_v19.h")
               for d in ("lqr_balance_v19", "fwe_v19")]
    for t in targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        with open(t, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"[생성] {os.path.relpath(t, here)}")

    print("\n다음 단계: python sim_check_v19.py  (게인 이식 전 폐루프 검증 — 필수)")
    return dict(K=K, L=L, Ad=Ad, Bd=Bd, C=C, fwe=fwe, model=mdl)


if __name__ == "__main__":
    main()

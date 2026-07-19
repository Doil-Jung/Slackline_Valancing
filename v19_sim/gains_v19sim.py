# -*- coding: utf-8 -*-
"""
gains_v19sim.py — v19_sim 용 게인 계산 (compute_gains_v19 와 동일 수식 + 서보 인지 옵션)
==========================================================================================
compute_gains_v19.main() 은 gains_v19.h 를 펌웨어 폴더에 써 버리므로,
시뮬레이터는 같은 수식을 부작용 없이 호출하는 이 모듈을 쓴다.

★servo_aware (v19_sim 의 핵심 발견, 2026-07-17):
  XM430 감속기어 반영 관성 I_a = N²·J_rotor ≈ 0.0045 kg·m² 는 이 로봇의
  힙 유효 관성 1/(Eᵀ M⁻¹ E) ≈ 0.0018 kg·m² 의 ~2.5배다. 즉 실물 힙의
  토크→가속 응답은 이론 모델의 ~27% 뿐이다. 게인·칼만을 이론 모델로 만들면
  추정기가 4배 빠른 응답을 예측해 폐루프가 발산한다 (전물리 시뮬로 확인).
  servo_aware=True 면 설계 모델에 다음을 반영한다:
    M += I_a·S_dd,   S_dd = [[0,0,0],[0,1,-1],[0,-1,1]]   (δ̇=θ̇−α̇ 의 에너지 ½I_a δ̇²)
    D 의 힙 감쇠를 '절대 θ̇'(model_v19 근사) 대신 '상대 δ̇'(물리) + 기어 점성 b_gear 로
  → 실물 브링업 전에 compute_gains_v19.py 에도 같은 반영이 필요하다는 근거.
"""
import numpy as np
from scipy.linalg import solve_discrete_are

import model_v19 as MD

S_DD = np.array([[0.0, 0, 0], [0, 1, -1], [0, -1, 1]])


def build_plant_servo(p, armature=0.0, b_gear=0.0, servo_aware=True):
    """model_v19.build_plant + (옵션) 서보 관성/감쇠 증강"""
    M, G, D, E, aux = MD.build_plant(p)
    if servo_aware:
        M = M + armature * S_DD
        # model_v19 의 절대 θ̇ 감쇠(-B_THETA)를 물리적 상대 δ̇ 감쇠로 교체
        D = D.copy()
        D[2, 2] += p["B_THETA"]                    # 절대항 제거
        D = D - (p["B_THETA"] + b_gear) * S_DD     # 상대항 (기어 점성 포함)
    return M, G, D, E, aux


def compute(p, armature=0.0, b_gear=0.0, servo_aware=True, verbose=False):
    """p = params_v19.flat() (+UI 오버라이드).  반환 dict:
       K(6,), L(6×5), C(5×6), D1_feed, Ad, Bd, mdl(모드/FWE 상수 포함)"""
    M, G, D, E, aux = build_plant_servo(p, armature, b_gear, servo_aware)
    Ac, Bc, Mi = MD.build_ss(M, G, D, E)
    Ad, Bd = MD.discretize(Ac, Bc, p["DT"])
    n = 6

    # ---- (φ,β) 축약 → 모드·ε*·FWE 상수 (model_v19.analyze 와 동일 흐름) ----
    M2, G2, D2, mu = MD.reduce_phibeta(M, G, D, aux)
    md = MD.modes_undamped(M2, G2)
    eps, s, lu, w = MD.eps_star_damped(M2, G2, D2)
    fwe = MD.fwe_map_constants(M2, G2, D2, mu,
                               p["T_FOLD"], p["T_WAIT"], p["T_EXTEND"], p["T_REST"])

    # ---------------- LQR ----------------
    Pric = solve_discrete_are(Ad, Bd, np.diag(p["Q_DIAG"]), np.array([[p["R_COST"]]]))
    K = (np.linalg.inv(np.array([[p["R_COST"]]]) + Bd.T @ Pric @ Bd) @ (Bd.T @ Pric @ Ad))

    # ---------------- 칼만 (z 5행: φ_enc, θ_acc보상, θ̇, δ, δ̇) ----------------
    w_imu = np.array([p["R"], p["L1"], p["ELL_IMU"]])
    wMiG = w_imu @ (Mi @ G); wMiD = w_imu @ (Mi @ D)
    wMiE = float(w_imu @ (Mi @ E).ravel())
    C = np.zeros((5, n))
    C[0, 0] = 1.0
    C[1, 2] = 1.0; C[1, 0:3] -= wMiG / 9.81; C[1, 3:6] -= wMiD / 9.81
    D1_feed = -wMiE / 9.81
    C[2, 5] = 1.0
    C[3, 2] = 1.0; C[3, 1] = -1.0
    C[4, 5] = 1.0; C[4, 4] = -1.0
    Pk = solve_discrete_are(Ad.T, C.T, np.diag(p["QKF_DIAG"]), np.diag(p["RKF_DIAG"]))
    L = Pk @ C.T @ np.linalg.inv(C @ Pk @ C.T + np.diag(p["RKF_DIAG"]))

    cl = max(abs(e) for e in np.linalg.eigvals(Ad - Bd @ K))
    ob = max(abs(e) for e in np.linalg.eigvals(Ad - L @ C))
    assert cl < 1.0, f"LQR 폐루프 불안정 max|eig|={cl}"
    assert ob < 1.0, f"관측기 불안정 max|eig|={ob}"

    mdl = dict(M=M, G=G, D=D, E=E, Ac=Ac, Bc=Bc, aux=aux,
               M2=M2, G2=G2, D2=D2, mu=mu, modes=md,
               eps_star=eps, s=s, lam_u=lu, w4=w, fwe=fwe)
    if verbose:
        print(f"[gains] servo_aware={servo_aware} I_a={armature} b_gear={b_gear} | "
              f"w_eff={md['w_eff']:.3f} lam_u={fwe['lu']:.3f} eps*={eps:.3f} "
              f"γf={fwe['GAMMA_FOLD']:.2f} γc={fwe['GAMMA_CYCLE']:.2f} | "
              f"cl={cl:.4f} ob={ob:.4f}")
    # ---- 2-엔코더 칼만 (발목 엔코더·IMU 없이 φ+δ 만: z 행 [0,3,4]) ----
    #   α 는 직접 측정이 없지만 몸 기울기가 크랭크를 미는 결합항을 통해 관측 가능.
    #   발목 엔코더 배송 전 브링업용 (2026-07-19 사용자 질문).
    idx3 = [0, 3, 4]
    C3 = C[idx3]
    R3 = np.diag(np.array(p["RKF_DIAG"])[idx3])
    # ★대역 배율 ENC2_QX: α 는 φ̈ 결합(이중적분 경로)으로만 보이므로 기본 튜닝은
    #   λ_u(6.45/s)보다 느려 발산 추적 실패 — 프로세스 잡음을 키워 관측기를 조인다.
    q3 = np.diag(np.array(p["QKF_DIAG"]) * p.get("ENC2_QX", 100.0))
    Pk3 = solve_discrete_are(Ad.T, C3.T, q3, R3)
    L3 = Pk3 @ C3.T @ np.linalg.inv(C3 @ Pk3 @ C3.T + R3)
    ob3 = max(abs(e) for e in np.linalg.eigvals(Ad - L3 @ C3))
    assert ob3 < 1.0, f"2-엔코더 관측기 불안정 max|eig|={ob3}"

    return dict(K=K[0], L=L, C=C, D1_feed=D1_feed, Ad=Ad, Bd=Bd, mdl=mdl,
                L3=L3, C3=C3, ob3_eig=ob3,
                p1r=aux["p1"] / aux["ps"], p2r=aux["p2"] / aux["ps"],
                cl_eig=cl, ob_eig=ob)

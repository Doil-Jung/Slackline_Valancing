# -*- coding: utf-8 -*-
"""
model_v19.py — 줄 관성(I_r,S_r)·감쇠(c_φ) 포함 v19 플랜트/모드/FWE 상수
=========================================================================
compute_K_measured_v2.py 의 플랜트를 다음과 같이 확장/수정:
  [수정1] M[φ,φ] = Mt R² + I_r,  G[φ,φ] = −(Mt R + S_r) g   ← 줄 관성 (7/11 이론)
  [수정2] D[φ,φ] = −c_φ                                        ← 베어링 점성감쇠
  [수정3] (φ,β,δ) 좌표의 2×2 대기(Wait) 부분계·모드·ε*·위험도 w 를
          3DOF 행렬에서 '유도'한다 (사람스케일 상수 하드코딩 제거)
  [검산]  φ행·δ열 = 0, G의 (φ,β)행·δ열 = 0  (이론: 접기는 μδ̈로 β행에만 들어옴)
  [검산]  I_r=S_r=c_φ=0 극한에서 ε* = −h/(rR+h) 복원

FWE 상수는 대수 대신 '수치 맵 캘리브레이션'으로 구한다(부호/계수 실수 원천 차단):
  fold-hold 맵:   A⁺ = Gf·A⁻ − gf·δf   → GAMMA_FOLD  = Gf/gf
  full-cycle 맵:  A⁺ = Gc·A⁻ − gc·δf   → GAMMA_CYCLE = Gc/gc
  (A = wᵀx, w 는 β성분=1 로 정규화 → A는 'β 환산 각도[rad]' 단위)
"""
import numpy as np


# ============================================================
# 3DOF 플랜트 (φ, α, θ)
# ============================================================
def build_plant(p):
    Mt = p["m1"] + p["m2"]
    p1 = p["m1"] * p["ell1"] + p["m2"] * p["L1"]
    p2 = p["m2"] * p["ell2"]
    p3 = p["m2"] * p["L1"] * p["ell2"]
    g = 9.81

    M = np.array([
        [Mt * p["R"]**2 + p["I_r"], p["R"] * p1,                              p["R"] * p2],
        [p["R"] * p1,               p["m1"]*p["ell1"]**2 + p["I1_cm"] + p["m2"]*p["L1"]**2, p3],
        [p["R"] * p2,               p3,                                       p["m2"]*p["ell2"]**2 + p["I2_cm"]],
    ])
    G = np.array([
        [-(Mt * p["R"] + p["S_r"]) * g, 0,      0],
        [0,                             p1 * g, 0],
        [0,                             0,      p2 * g],
    ])
    D = np.diag([-p["c_phi"], 0.0, -p["B_THETA"]])
    E = np.array([[0.0], [-1.0], [1.0]])

    assert np.all(np.linalg.eigvalsh(M) > 0), "M 비양정치 — 파라미터 점검!"
    aux = dict(Mt=Mt, p1=p1, p2=p2, ps=p1 + p2, h=(p1 + p2) / Mt, g=g)
    return M, G, D, E, aux


def build_ss(M, G, D, E):
    """6D 연속 상태공간 x=[q, q̇]"""
    Mi = np.linalg.inv(M)
    n = 6
    Ac = np.zeros((n, n)); Bc = np.zeros((n, 1))
    Ac[:3, 3:] = np.eye(3)
    Ac[3:, :3] = Mi @ G
    Ac[3:, 3:] = Mi @ D
    Bc[3:, :] = Mi @ E
    return Ac, Bc, Mi


def discretize(Ac, Bc, DT):
    from scipy.linalg import expm
    n = Ac.shape[0]
    Z = np.zeros((n + 1, n + 1))
    Z[:n, :n] = Ac * DT
    Z[:n, n:] = Bc * DT
    eZ = expm(Z)
    return eZ[:n, :n], eZ[:n, n:n + 1]


# ============================================================
# (φ, β, δ) 좌표로 축약 — δ 프로파일 구동 형태
#   q3 = J · (φ, β, δ)ᵀ,  J 유도: p1α+p2θ = ps·β (h=ps/Mt),  θ−α = δ
# ============================================================
def reduce_phibeta(M, G, D, aux):
    ps, p1, p2 = aux["ps"], aux["p1"], aux["p2"]
    J = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, -p2 / ps],
        [0.0, 1.0,  p1 / ps],
    ])
    Mp = J.T @ M @ J   # (φ,β,δ)
    Gp = J.T @ G @ J
    Dp = J.T @ D @ J
    # [검산] 이론 구조: 접기(δ̈)는 φ행에 안 들어오고(무게중심/공액운동량 보존),
    #        접힌 자세의 중력은 (φ,β)행에 안 들어온다 (β의 가중 정의 덕)
    assert abs(Mp[0, 2]) < 1e-9 * abs(Mp[0, 0]), "φ행 δ열 ≠ 0 — J/모델 점검"
    assert abs(Gp[0, 2]) < 1e-9 and abs(Gp[1, 2]) < 1e-9, "(φ,β)행 G δ열 ≠ 0"
    M2 = Mp[:2, :2]; G2 = Gp[:2, :2]; D2 = Dp[:2, :2]
    mu = Mp[1, 2]          # β행의 δ̈ 관성 결합 계수
    return M2, G2, D2, mu


def F4_matrix(M2, G2, D2):
    """4D 대기 동역학 ẋ = F x,  x = [φ, β, φ̇, β̇]"""
    A2 = np.linalg.solve(M2, G2)
    Dd = np.linalg.solve(M2, D2)
    F = np.zeros((4, 4))
    F[:2, 2:] = np.eye(2)
    F[2:, :2] = A2
    F[2:, 2:] = Dd
    return F


def modes_undamped(M2, G2):
    A2 = np.linalg.solve(M2, G2)
    lam, V = np.linalg.eig(A2)
    lam = lam.real; V = V.real
    iu = int(np.argmax(lam)); ist = 1 - iu
    return dict(
        w_eff=float(np.sqrt(-lam[ist])), l_eff=float(np.sqrt(lam[iu])),
        r=float(V[0, ist] / V[1, ist]),
        u2=np.linalg.inv(V)[iu, :],
    )


def unstable_left4(F):
    """감쇠 포함 4D: 유일한 양의 실고유값 λ_u 와 좌고유벡터 w (β성분=1 정규화)"""
    lam, V = np.linalg.eig(F)
    pos = [i for i in range(4) if lam[i].real > 1e-9 and abs(lam[i].imag) < 1e-9]
    assert len(pos) == 1, f"불안정 고유값 {len(pos)}개 ≠ 1 — 모델 점검"
    iu = pos[0]
    w = np.real(np.linalg.inv(V)[iu, :])
    w = w / w[1]                     # A = wᵀx 가 'β 환산 rad' 단위가 되도록
    return float(lam[iu].real), w, lam


def eps_star_damped(M2, G2, D2):
    """정지출발·순간접기·감쇠 포함 ε* (일반형; 무감쇠·무질량 극한서 −h/(rR+h))"""
    lu, w, _ = unstable_left4(F4_matrix(M2, G2, D2))
    s = M2[0, 1] / M2[0, 0]          # 점프 기하: Δφ = s·(1+ε)β_i (p_φ 보존)
    k = w[0] * s - w[1]
    one_plus = -w[1] / k
    return one_plus - 1.0, s, lu, w


# ============================================================
# FWE 수치 맵 캘리브레이션 — 삼각 가감속 프로파일로 δ(t) 구동
# ============================================================
def _tri_profile(t, T, dmax):
    """삼각 가감속: δ(0)=0 → δ(T)=dmax.  반환 (δ, δ̇, δ̈)"""
    if t <= 0:      return 0.0, 0.0, 0.0
    if t >= T:      return dmax, 0.0, 0.0
    a = 4.0 * dmax / T**2
    if t < T / 2:   return 0.5*a*t*t, a*t, a
    tr = T - t
    return dmax - 0.5*a*tr*tr, a*tr, -a


def _simulate_A(F, b, w, lu, schedule, x0, dt=2e-5):
    """schedule = [(dur, profile_fn(t)->(δ,δ̇,δ̈)), ...] 구간별 δ̈ 강제 RK4.
       반환: 종료 시 A = wᵀx"""
    x = x0.copy()
    for dur, fn in schedule:
        n = int(round(dur / dt))
        for i in range(n):
            t = i * dt
            def f(xx, tt):
                dd = fn(tt)[2]
                return F @ xx + b * dd
            k1 = f(x, t); k2 = f(x + dt/2*k1, t + dt/2)
            k3 = f(x + dt/2*k2, t + dt/2); k4 = f(x + dt*k3, t + dt)
            x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return float(w @ x), x


def fwe_map_constants(M2, G2, D2, mu, Tf, Tw, Te, Trest):
    """fold-hold 맵과 full-cycle 맵의 (G, g) — 수치 캘리브레이션.
       A⁺ = G·A⁻ − g·δf  (δf: 접기각 [rad], A: β환산 [rad])"""
    F = F4_matrix(M2, G2, D2)
    lu, w, _ = unstable_left4(F)
    Mi2 = np.linalg.inv(M2)
    b = np.array([0.0, 0.0, Mi2[0, 1] * (-mu), Mi2[1, 1] * (-mu)])
    zero = np.zeros(4)

    # --- fold only (접고 유지): A⁻=0, δf=1rad 로 g 측정 ---
    hold = lambda dmax: (lambda t: (dmax, 0.0, 0.0))
    A_end, _ = _simulate_A(F, b, w, lu, [(Tf, lambda t: _tri_profile(t, Tf, 1.0))], zero)
    gf = -A_end
    Gf = float(np.exp(lu * Tf))
    # --- full cycle (접기 Tf → 대기 Tw → 펴기 Te): ---
    sched = [
        (Tf, lambda t: _tri_profile(t, Tf, 1.0)),
        (Tw, hold(1.0)),
        (Te, lambda t: tuple(np.subtract((1.0, 0.0, 0.0), _tri_profile(t, Te, 1.0)))),
    ]
    A_end2, _ = _simulate_A(F, b, w, lu, sched, zero)
    gc = -A_end2
    Gc = float(np.exp(lu * (Tf + Tw + Te + Trest)))   # 사이클+휴지 전체 성장
    return dict(
        lu=lu, w=w,
        G_fold=Gf, g_fold=gf, GAMMA_FOLD=Gf / gf,
        G_cyc=Gc, g_cyc=gc, GAMMA_CYCLE=Gc / gc,
    )


# ============================================================
# 종합 리포트
# ============================================================
def analyze(p, verbose=True):
    M, G, D, E, aux = build_plant(p)
    Ac, Bc, Mi = build_ss(M, G, D, E)
    M2, G2, D2, mu = reduce_phibeta(M, G, D, aux)
    md = modes_undamped(M2, G2)
    eps, s, lu, w = eps_star_damped(M2, G2, D2)
    fwe = fwe_map_constants(M2, G2, D2, mu,
                            p["T_FOLD"], p["T_WAIT"], p["T_EXTEND"], p["T_REST"])

    # [검산] 무질량·무감쇠 극한 → 기존 ε* 공식
    p0 = dict(p); p0.update(I_r=0.0, S_r=0.0, c_phi=0.0, B_THETA=0.0)
    M0, G0, D0, _, aux0 = build_plant(p0)
    M20, G20, D20, _ = reduce_phibeta(M0, G0, D0, aux0)
    md0 = modes_undamped(M20, G20)
    e0, _, _, _ = eps_star_damped(M20, G20, D20)
    e_formula = -aux0["h"] / (md0["r"] * p["R"] + aux0["h"])
    assert abs(e0 - e_formula) < 1e-9, "무질량 극한 ε* 불일치 — 이론 검산 실패"

    out = dict(M=M, G=G, D=D, E=E, Ac=Ac, Bc=Bc, aux=aux,
               M2=M2, G2=G2, D2=D2, mu=mu, modes=md,
               eps_star=eps, s=s, lam_u=lu, w4=w, fwe=fwe,
               eps_star_massless=e0)
    if verbose:
        Mt = aux["Mt"]; iota = p["I_r"] / (Mt * p["R"]**2)
        print(f"[플랜트] Mt={Mt:.3f} kg  h={aux['h']:.3f} m  R={p['R']} m  "
              f"I_r={p['I_r']:.5f} (iota={iota:.3f})  S_r={p['S_r']:.4f}  c_phi={p['c_phi']}")
        print(f"[모드]   ω_eff={md['w_eff']:.3f}  λ_eff={md['l_eff']:.3f} rad/s "
              f"(배가 {np.log(2)/md['l_eff']*1000:.0f} ms)  r={md['r']:.3f}  s={s:.3f}")
        print(f"[FWE]    λ_u(감쇠)={lu:.3f}  ε*_damped={eps:.3f}  "
              f"(무질량·무감쇠였다면 {e0:.3f} → 차이 {100*(eps-e0)/e0:+.1f}% ★줄관성 반영 필수 근거)")
        print(f"[FWE맵]  fold: G={fwe['G_fold']:.3f} g={fwe['g_fold']:.4f} γ_fold={fwe['GAMMA_FOLD']:.2f}")
        print(f"         cycle:G={fwe['G_cyc']:.3f} g={fwe['g_cyc']:.4f} γ_cyc={fwe['GAMMA_CYCLE']:.2f} "
              f"| 수축조건 ρ∈({1-1/fwe['G_cyc']:.2f}, {1+1/fwe['G_cyc']:.2f})")
        print(f"[검산]   무질량 극한 ε*={e0:.4f} == −h/(rR+h)={e_formula:.4f} ✓ / "
              f"불안정 부분공간 1차원 ✓ / φ행 δ̈ 결합 0 ✓")
    return out


if __name__ == "__main__":
    import params_v19 as P
    analyze(P.flat())

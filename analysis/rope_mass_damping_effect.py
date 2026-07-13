# -*- coding: utf-8 -*-
"""
줄(스윙 링크) 질량·감쇠가 FWE 이론에 미치는 영향 — 2026-07-11
================================================================
질문(정도일): 재설계(카본 크랭크 + 베어링)로 줄의 질량과 감쇠가 무시할 수
없게 됐다. FWE 메커니즘이 깨지는가? 이론적으로 어떻게 되는가?

답의 구조 (수치로 확인):
  [A] 줄 관성 I_r, 줄 정적모멘트 S_r=m_r·ℓ_r 은 φ행에만 들어간다:
        M[φ,φ] = M_t R² + I_r,   G[φ,φ] = −(M_t R + S_r)·g
      → 2×2 결합 LTI 구조 보존. 모든 닫힌형이 "새 행렬로 재계산"만으로 유지.
      → 등고선 기울기: −h/R → −Rp_s/(M_tR²+I_r)/R 로 변경 (보존량이
        x_CoM이 아니라 φ-공액운동량 p_φ이기 때문)
      → ε* = −s/(r+s),  s ≡ M[φ,β]/M[φ,φ]  (무질량 극한서 s=h/R, 기존식 복원)
  [B] 점성감쇠 c_φ (베어링): 4D 상태행렬 F의 고유분해로 일반화.
      불안정 부분공간은 여전히 1차원 (양의 실고유값 1개) → A ≡ wᵀx
      (w = λ_u의 좌고유벡터)로 FWE 형식이 그대로 이월.
      무감쇠 극한서 wᵀx ∝ u₂ᵀ(q + q̇/λ) (예측변위) 복원 확인.
      λ_u는 감쇠로 '감소'(유리), 안정모드는 감쇠진동(잔여진동 자동소멸, 유리).
  [C] 건마찰(쿨롱)은 선형이론 밖 — 데드밴드/리밋사이클 논의는 문서 참조.

실행: repo 루트에서 python analysis/rope_mass_damping_effect.py
출력: 표 3장 + docs/줄질량감쇠_영향_20260711.png
"""
import numpy as np

g = 9.81

# ---------------------------------------------------------------
# 사람스케일 정본 파라미터 (docs 정합: R=0.7 → ω=6.30, λ=3.54, r=−2.12)
# ---------------------------------------------------------------
Mt, h, ps, Jb = 75.0, 0.96, 72.0, 89.03
R = 0.7

def build(R, I_r=0.0, S_r=0.0):
    """(φ,β) 2×2 M, G — 줄 관성/정적모멘트 포함"""
    M = np.array([[Mt * R**2 + I_r, R * ps],
                  [R * ps,          Jb]])
    G = np.array([[-(Mt * R + S_r) * g, 0.0],
                  [0.0,                 ps * g]])
    return M, G

def modes(M, G):
    """무감쇠 2×2 → (ω_eff, λ_eff, r, u2) ; u2: 불안정 좌고유벡터"""
    A2 = np.linalg.solve(M, G)
    lam, V = np.linalg.eig(A2)
    iu = int(np.argmax(lam.real))          # λ² > 0 : 불안정
    ist = 1 - iu
    w_eff = np.sqrt(-lam[ist].real)
    l_eff = np.sqrt(lam[iu].real)
    r = V[0, ist] / V[1, ist]
    U = np.linalg.inv(V)
    return w_eff, l_eff, r, U[iu, :]

def eps_star(M, G):
    """정지 ε* — 일반형 ε* = −s/(r+s), s = M01/M00 (관성 포함 점프 기하)"""
    _, _, r, _ = modes(M, G)
    s = M[0, 1] / M[0, 0]
    return -s / (r + s), s

# ---------------------------------------------------------------
# 표 1 : 줄 관성 효과 (하단집중 가정: S_r = I_r/R)
# ---------------------------------------------------------------
print("=" * 76)
print("표 1. 줄(크랭크) 관성의 효과 — 사람스케일 R=0.7 (하단집중: S_r=I_r/R)")
print("      iota = I_r/(M_t R²)")
print("-" * 76)
print(f"{'iota':>6} {'ω_eff':>8} {'λ_eff':>8} {'r':>8} {'ε*':>8} "
      f"{'등고선기울기':>12} {'Δε*':>8}")
M0, G0 = build(R)
e0, _ = eps_star(M0, G0)
for iota in [0.0, 0.05, 0.10, 0.20, 0.30]:
    I_r = iota * Mt * R**2
    M, G = build(R, I_r=I_r, S_r=I_r / R)
    w, l, r, _ = modes(M, G)
    e, s = eps_star(M, G)
    print(f"{iota:6.2f} {w:8.3f} {l:8.3f} {r:8.3f} {e:8.3f} "
          f"{-s:12.3f} {100*(e-e0)/e0:7.1f}%")

# 검산: iota=0 → 기존 ε* = −h/(rR+h)
w, l, r, _ = modes(M0, G0)
assert abs(eps_star(M0, G0)[0] - (-h / (r * R + h))) < 1e-12
print(f"[검산] iota=0: ω={w:.3f}(6.302), λ={l:.3f}(3.538), r={r:.3f}(−2.119), "
      f"ε*={e0:.3f}(1.832~1.834) — docs 정합 OK")

# ---------------------------------------------------------------
# 표 2 : 점성감쇠 — 4D 고유구조와 일반화 ε*_damped
# ---------------------------------------------------------------
def F4(M, G, c_phi):
    """4D 상태행렬 (x = [φ, β, φ̇, β̇])"""
    A2 = np.linalg.solve(M, G)
    Dm = np.linalg.solve(M, np.diag([c_phi, 0.0]))
    F = np.zeros((4, 4))
    F[:2, 2:] = np.eye(2)
    F[2:, :2] = A2
    F[2:, 2:] = -Dm
    return F

def unstable_left(F):
    """양의 실고유값 λ_u 와 그 좌고유벡터 w (개수=1 확인)"""
    lam, V = np.linalg.eig(F)
    pos = [i for i in range(4) if lam[i].real > 1e-9 and abs(lam[i].imag) < 1e-9]
    assert len(pos) == 1, f"불안정 고유값 개수 {len(pos)} ≠ 1"
    iu = pos[0]
    W = np.linalg.inv(V)          # 행 = 좌고유벡터
    return lam[iu].real, np.real(W[iu, :]), lam

def eps_star_damped(M, G, c_phi, beta_i=np.deg2rad(2)):
    """정지출발, 순간접기, 감쇠 포함: wᵀ(x0 + Δx(ε)) = 0 → ε 1차식"""
    lu, w4, lam = unstable_left(F4(M, G, c_phi))
    s = M[0, 1] / M[0, 0]
    # x0 = (0, β_i, 0, 0);  Δx = (s(1+ε)β_i, −(1+ε)β_i, 0, 0)
    # wᵀx0 + (1+ε)β_i(w0·s − w1) = 0
    k = w4[0] * s - w4[1]
    one_plus_eps = -(w4[1] * 1.0) / k          # β_i 약분
    return one_plus_eps - 1.0, lu, lam

print()
print("=" * 76)
print("표 2. 점성감쇠 c_φ 의 효과 — 사람스케일 R=0.7, iota=0.20 고정")
print("      d ≡ c_φ/(2·M[φ,φ]·ω_eff)  (무차원 감쇠 수준)")
print("-" * 76)
M, G = build(R, I_r=0.20 * Mt * R**2, S_r=0.20 * Mt * R**2 / R)
w_eff0, l_eff0, _, u2 = modes(M, G)
print(f"{'d':>6} {'λ_u':>8} {'안정모드 (Re±Im)':>22} {'감쇠시정수':>10} "
      f"{'ε*_damped':>10} {'Δε* vs 무감쇠':>13}")
e_ud, _, _ = eps_star_damped(M, G, 0.0)
for d in [0.0, 0.01, 0.03, 0.05, 0.10]:
    c = d * 2 * M[0, 0] * w_eff0
    e, lu, lam = eps_star_damped(M, G, c)
    osc = lam[np.argmax(np.abs(lam.imag))]
    tdec = np.inf if abs(osc.real) < 1e-12 else -1 / osc.real
    print(f"{d:6.2f} {lu:8.4f} {osc.real:10.4f}±{abs(osc.imag):7.3f}j "
          f"{tdec:10.2f} {e:10.4f} {100*(e-e_ud)/e_ud:12.2f}%")

# 무감쇠 극한 검산: w4-사영 == u2ᵀ(q + q̇/λ)
lu, w4, _ = unstable_left(F4(M, G, 1e-12))
w4 = w4 / w4[1] * u2[1]
pred = np.hstack([u2, u2 / l_eff0])
assert np.allclose(w4 / np.linalg.norm(w4), pred / np.linalg.norm(pred), atol=1e-5)
print("[검산] c_φ→0: 4D 좌고유벡터 사영 == u₂ᵀ(q+q̇/λ) (예측변위) — 형식 이월 OK")

# ---------------------------------------------------------------
# 표 3 : "감쇠 무시하고 접으면?" — 잔여 A와 발산 시간 (시간적분)
# ---------------------------------------------------------------
def simulate(M, G, c_phi, eps, beta_i=np.deg2rad(2), T=6.0, dt=1e-4):
    s = M[0, 1] / M[0, 0]
    x = np.array([s * (1 + eps) * beta_i, -eps * beta_i, 0.0, 0.0])
    F = F4(M, G, c_phi)
    lu, w4, _ = unstable_left(F)
    n = int(T / dt)
    t_fall = None
    for i in range(n):
        # RK4
        k1 = F @ x; k2 = F @ (x + dt/2*k1); k3 = F @ (x + dt/2*k2); k4 = F @ (x + dt*k3)
        x = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        if t_fall is None and abs(np.rad2deg(x[1])) > 15:
            t_fall = (i+1) * dt
    return t_fall, x

print()
print("=" * 76)
print("표 3. 감쇠계에서 '무감쇠 ε*'로 접었을 때 (iota=0.20, β_i=2°, 6 s 적분)")
print("-" * 76)
print(f"{'d':>6} {'ε 사용':>12} {'β>15° 도달':>12} {'6s 후 |β|':>10}")
for d in [0.03, 0.10]:
    c = d * 2 * M[0, 0] * w_eff0
    e_d, _, _ = eps_star_damped(M, G, c)
    for tag, e in [("무감쇠 ε*", e_ud), ("감쇠보정 ε*", e_d)]:
        tf, x = simulate(M, G, c, e)
        print(f"{d:6.2f} {tag:>12} {str(tf) if tf else '없음(유계)':>12} "
              f"{np.rad2deg(abs(x[1])):9.2f}°")

# ---------------------------------------------------------------
# 그림
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
iotas = np.linspace(0, 0.35, 60)
es, ls_, ws_ = [], [], []
for io in iotas:
    I_r = io * Mt * R**2
    Mx, Gx = build(R, I_r=I_r, S_r=I_r / R)
    wx, lx, _, _ = modes(Mx, Gx)
    ex, _ = eps_star(Mx, Gx)
    es.append(ex); ls_.append(lx); ws_.append(wx)
ax[0].plot(iotas, es, 'r-', label=r'$\epsilon^*$')
ax[0].set_xlabel(r'rope inertia ratio  $\iota=I_r/(M_tR^2)$')
ax[0].set_ylabel(r'$\epsilon^*$', color='r')
ax0b = ax[0].twinx()
ax0b.plot(iotas, ls_, 'b--', label=r'$\lambda_{eff}$')
ax0b.plot(iotas, ws_, 'g--', label=r'$\omega_{eff}$')
ax0b.set_ylabel('rad/s')
ax[0].set_title('Rope inertia: structure preserved,\nnumbers shift (R=0.7, bottom-heavy)')
ax0b.legend(loc='center right'); ax[0].legend(loc='upper left')

ds = np.linspace(0, 0.12, 40)
lus, eds = [], []
for d in ds:
    c = d * 2 * M[0, 0] * w_eff0
    e, lu, _ = eps_star_damped(M, G, c)
    lus.append(lu); eds.append(e)
ax[1].plot(ds, eds, 'r-')
ax[1].set_xlabel(r'damping level  $d=c_\phi/(2M_{\phi\phi}\omega)$')
ax[1].set_ylabel(r'$\epsilon^*_{damped}$', color='r')
ax1b = ax[1].twinx()
ax1b.plot(ds, lus, 'b--')
ax1b.set_ylabel(r'$\lambda_u$  (rad/s)', color='b')
ax[1].set_title('Viscous damping: unstable subspace stays 1-D,\n'
                r'$\lambda_u$ decreases (helps), $\epsilon^*$ shifts mildly')
plt.tight_layout()
out = "docs/줄질량감쇠_영향_20260711.png"
plt.savefig(out, dpi=130)
print(f"\n그림 저장: {out}")

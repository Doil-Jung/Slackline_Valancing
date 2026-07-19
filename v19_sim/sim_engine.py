# -*- coding: utf-8 -*-
"""
sim_engine.py — v19 MuJoCo 전물리 시뮬 엔진 (서보·마찰·센서 현실화 포함)
=========================================================================
구성 (500Hz 제어 루프, 물리는 0.5ms × 4 서브스텝):
  [센서]   φ엔코더 14bit 양자화 / IMU 가속도계(atan2 thAcc, 1g 게이팅)·자이로(노이즈+바이어스)
           / XM430 엔코더 0.088° / PRESENT_VELOCITY 0.229rpm 양자화
  [추정]   펌웨어 lqr_balance_v19 와 동일한 칼만5 (Ad,Bd,L,C,D1_feed) 또는 참값
  [제어]   OFF / LQR / FWE(단일접기·완전사이클·반복FEW — fwe_v19 로직 이식)
  [서보]   τ→전류unit 양자화(2.69mA)·CUR_SAFE 클립·데드밴드 → Kt(플랜트값)·전류
           → 토크-속도 곡선(역기전력 한계) → MuJoCo 토크. armature·기어마찰은 MJCF 관절에.
  [부호]   모델 φ = −q_phi(MuJoCo).  α,θ 는 월드피치 그대로 (AGENTS §9).

스텝백: 매 제어스텝 전 스냅샷(qpos,qvel,추정기,제어기 상태) 링버퍼 → 1장 뒤로 복원.
"""
import numpy as np
import mujoco

import params_v19 as P
import mjcf_v19 as MJ
import gains_v19sim as GS

# ------------------------------------------------------------
# 현실화 기본값 (실측 전 설계 추정치 — UI 로 조정)
# ------------------------------------------------------------
REAL_DEFAULT = dict(
    servo_on   = True,    # 서보 전체 모델 (양자화·Kt·토크-속도 한계·데드밴드)
    sensor_on  = True,    # 센서 양자화·노이즈
    mj_dt      = 5e-4,    # 물리 서브스텝
    # --- 서보 XM430-W210 ---
    tau_stall  = 3.0,     # [N·m] 스톨
    w_noload_dps = 276.0, # [°/s] 무부하 속도 (프로젝트 관례값, sim_check 3b 와 동일)
    cur_deadband = 5,     # [unit] 이하 전류명령은 0 (정지마찰 미달 대략치)
    kt_scale   = 1.0,     # 플랜트측 Kt 편차 (제어기는 공칭 KT 사용)
    armature   = 0.0045,  # [kg·m²] 기어 반영 관성 N²·J_rotor (212.6² × ~1e-7) ★실측 대상
    b_gear     = 0.01,    # [N·m·s/rad] 기어 점성 (B_THETA 에 합산) ★실측 대상
    f_hip      = 0.05,    # [N·m] 힙 기어 쿨롱 마찰 ★실측 대상
    # --- 베어링 ---
    f_phi      = 0.002,   # [N·m] 상단 베어링 쿨롱 (608 베어링 ×2)
    c_phi_extra= 0.0,     # c_phi(params) 에 더할 추가 점성
    f_ankle    = 0.002,   # [N·m] 발목 베어링 쿨롱
    b_ankle    = 0.0,
    # --- 센서 ---
    enc_bits   = 14,      # φ 엔코더 (AS5048A)
    hip_enc_quant = 2*np.pi/4096,          # XM430 0.088°
    hip_vel_quant = 0.229*2*np.pi/60.0,    # PRESENT_VELOCITY 0.229 rpm
    acc_noise  = 0.02,    # thAcc 노이즈 [rad] σ
    gyro_noise = 0.002,   # [rad/s] σ
    gyro_bias_rw = 0.0,   # [rad/s per √s] 바이어스 랜덤워크
    acc_gate_g = 0.30,    # |‖a‖−g| > 0.30g 이면 thAcc 게이팅(그 행 혁신 0)
    # --- 상보필터(CF) 추정기 (v19: φ·δ·δ̇·θ̇ 전부 직접측정 → 미지수는 θ뿐) ---
    cf_k_acc   = 0.002,   # θ 보정 게인 (시정수 ≈ DT/k ≈ 1 s; 게이팅 시 보정 중단)
    cf_tau_phid= 0.02,    # φ̇ = 엔코더 차분 LPF 시정수 [s]
    cf_tau_deld= 0.015,   # δ̇(PRESENT_VELOCITY 양자화 0.024rad/s) LPF 시정수 [s]
                          #   ★없으면 양자화 계단이 α̇로 직결 → LQR 토크 채터로 낙하
    # --- FWE 실센서화 (야간 실험 2026-07-17) ---
    fwe_settle_s  = 0.25, # 트리거 무장 전 최소 정착시간 [s] (리셋/펴기 종료 후. 0=끔)
    fwe_fast_L    = True, # 접기~펴기(+직후) 구간 고대역 관측기 L_fast 로 전환
    fwe_fastL_qx  = 300.0,# L_fast 용 QKF 배율 (프로세스 잡음↑ = 측정 신뢰↑)
    fwe_fastL_hold= 0.10, # 펴기 종료 후 L_fast 유지시간 [s]
    # --- 발목 엔코더 추정기 (사용자 제안 2026-07-17 밤: IMU 가속도계 대체) ---
    ankle_enc_bits = 14,  # 발목 엔코더 분해능 (AS5048A 가정)
    ankle_enc_mass = 0.015,# [kg] 발축 추가 질량(칩·자석·마운트) — bottom_m 에 가산해 플랜트 반영
    enc_use_gyro = True,  # True: θ̇ 는 자이로 사용(권장) / False: 순수 엔코더 차분
    enc_tau_v   = 0.010,  # 엔코더 차분 속도 LPF 시정수 [s]
    # --- 하이브리드 (평상시 LQR + |A| 초과 시 FWE 사이클) ---
    hyb_trig_deg = 2.0,   # [β환산 °] 이 이상이면 접기 사이클 발동 (LQR 포획영역 언저리)
    fwe_adapt_T = True,   # ★접기/펴기 시간을 서보 속도한계에 맞춰 자동 연장 (δf 클 때)
    fwe_v_margin = 1.25,  # 피크속도 여유율 (vpk = 2δf/T ≤ vmax/margin)
    fwe_ff = True,        # ★접기 프로파일 관성 피드포워드 τ_ff = I_δ·δ̈_ref (PD 강성 의존 제거)
    fwe_gamma_cal = 0.0,  # ★엔진 실행 데이터로 적합한 γ* (0=공칭 γ 사용). 야간 적합값 18.6
    fwe_online = False,   # ★사이클마다 (A⁻,δf,A⁺) RLS 로 Gc·gc 온라인 갱신 (자가 노브)
    fwe_online_ff = 0.90, # RLS 망각계수
    # --- 방어적 RLS (2026-07-18: 위치모드에서 무방비 RLS 는 오염돼 3초 내 사망) ---
    fwe_online_gate_deg = 0.0, # >0: |A⁻|·|A⁺| 가 이 값[°] 초과이거나 δf 포화면 갱신 스킵
                               #    (선형 영역 밖 사이클은 지도 모델이 안 맞음 — 배우면 독)
    fwe_online_rej_deg = 0.0,  # >0: |혁신 y−xᵀθ| 가 이 값[°] 초과면 갱신 스킵 (이상치 거부)
    # --- 루프 지연 (2026-07-18: 센서→연산→구동 왕복 지연. 실물 예상 10~30ms) ---
    latency_ms = 0.0,     # 측정이 이 시간만큼 옛것 (제어스텝 2ms 단위 반올림)
    latency_comp = False, # ★트리거 시 A ← A·e^{λ·지연}: 지연은 아니까 예측으로 보정
    lqr_fric_ff = 0.0,    # [N·m] LQR 스틱션 보상 피드포워드 τ += f̂·tanh(δ̇/0.05) (0=끔)
    # --- 위치모드 FWE (사용자 제안 2026-07-18: 서보 내장 제어기에 '최대속도로 가라') ---
    fwe_pos_mode = False, # ★True: 외부 전류 PD 대신 내장 위치제어 흉내 (fwe3 전용)
    pos_kp = 400.0,       # [N·m/rad] 내장 PD 강성 (0.43° 오차면 토크 포화 → 사실상 뱅뱅)
    pos_kd = 3.0,         # [N·m·s/rad] 내장 PD 감쇠 (ζ≈0.95 임계감쇠: kd=2√(kp·I_δ))
    pos_tol_deg = 0.5,    # 접기/펴기 '도달' 판정 허용오차 — 시간이 아니라 도달로 상전이
    pos_vel_dps = 420.0,  # [°/s] 프로파일 순항속도 (Profile Velocity). 무부하 462 미만 필수
                          #   (무부하 속도에선 토크 0 → 추종 불능). 접기 ~13° 는 a=8000 에서
                          #   순항 도달 전에 감속 시작 → 실질 삼각 프로파일, v 는 거의 안 묶임
    pos_acc_dps2 = 8000.0,# [°/s²] 프로파일 가속도 (Profile Acceleration) ★프로파일 스윕 최적
                          #   (2026-07-18: 3000~12000 스윕, 지도적합 rms 1.84° 최소·생존 최대.
                          #   물리: I_δ·a ≈ 중속 가용토크 0.9N·m — '가진 토크만큼만' 이 최적.
                          #   더 빠르면 추종 붕괴, 더 느리면 사이클 길어져 Gc(오차증폭) 폭증)
    armature_plant_scale = 1.0,  # 플랜트측 armature 배율 (제어기 비공개 — 미지 모델오차 실험용)
    gear_eta = 1.0,       # ★기어 전달효율 η (부하 비례 마찰의 방향 비대칭).
                          #   모터가 일할 때(τ·δ̇>0) τ→η·τ, 역구동/제동 시 τ→τ/η (η_back=2−1/η 물리).
                          #   1.0=끔(기존 기준선 보존). XM430 다단 평기어 추정 0.7~0.85.
)

MODE_OFF, MODE_LQR, MODE_FWE1, MODE_FWE2, MODE_FWE3, MODE_HYB = "off", "lqr", "fwe1", "fwe2", "fwe3", "hyb"
PH_IDLE, PH_FOLD, PH_WAIT, PH_EXTEND, PH_REST, PH_HOLD, PH_DONE = \
    "idle", "fold", "wait", "extend", "rest", "hold", "done"


def _tri(t, T, d0, d1):
    """삼각 가감속 δ0→δ1. 반환 (δ_ref, δ̇_ref, δ̈_ref)"""
    if T <= 0 or t >= T:
        return d1, 0.0, 0.0
    if t <= 0:
        return d0, 0.0, 0.0
    D = d1 - d0
    a = 4.0 * D / T**2
    if t < T/2:
        return d0 + 0.5*a*t*t, a*t, a
    tr = T - t
    return d1 - 0.5*a*tr*tr, a*tr, -a


class SimEngine:
    def __init__(self):
        self.nominal = {}      # params_v19 오버라이드 (공칭 = 제어기가 아는 값)
        self.mismatch = dict(m1=0.0, m2=0.0, L1=0.0, L2=0.0, R=0.0, kt=0.0)  # [%]
        self.real = dict(REAL_DEFAULT)
        self.ctrl_mode = MODE_LQR
        self.estimator = "kalman"   # "kalman" | "true"
        self.servo_aware = True     # ★게인 설계 모델에 armature·기어감쇠 반영 (OFF=현행 펌웨어 파이프라인)
        self.gain_scale = 1.0
        self.kp_servo = 25.0        # FWE 프로파일 추종 PD [N·m/rad] (전물리 튜닝값)
        self.kd_servo = 0.4
        # 기본 초기조건은 전현실화 LQR 포획영역(±0.5~1°) 안쪽으로
        self.x0 = dict(phi0=0.0, alpha0=0.3, theta0=0.3)   # [deg]
        self.rng = np.random.default_rng(0)
        self._phi4_cache = {}    # 지연 예측기 expm 캐시 (build 시 초기화)
        self.build()

    # ---------------- 파라미터/빌드 ----------------
    def params(self):
        p = P.flat()
        p.update(self.nominal)
        # 파생량 재계산 (crank 질량 등 오버라이드 반영)
        rope = {k: dict(v=p[k]) for k in
                ("crank_m_each", "crank_L", "bottom_m", "top_socket_m", "top_socket_ell", "R")}
        p["I_r"], p["S_r"], p["m_rope"] = P.calc_rope_inertia(rope)
        if self.nominal.get("I1_cm") is None or "I1_cm" not in self.nominal:
            p["I1_cm"] = p["m1"]*p["L1"]**2/12
            p["I2_cm"] = p["m2"]*p["L2"]**2/12
        return p

    def build(self):
        self._phi4_cache = {}    # 모델 바뀌면 지연 예측기 재계산
        p = self.params()
        self.p = p
        self.g = GS.compute(p, armature=self.real["armature"],
                            b_gear=self.real["b_gear"],
                            servo_aware=self.servo_aware)   # 제어기는 공칭 파라미터만 안다
        # 접기 구간용 고대역 관측기 (같은 모델, QKF 확대 → 직접측정 신뢰↑)
        pf = dict(p); pf["QKF_DIAG"] = [q*self.real["fwe_fastL_qx"] for q in p["QKF_DIAG"]]
        self.g_fast = GS.compute(pf, armature=self.real["armature"],
                                 b_gear=self.real["b_gear"], servo_aware=self.servo_aware)
        scale = {k: 1.0 + self.mismatch.get(k, 0.0)/100.0 for k in ("m1", "m2", "L1", "L2", "R")}
        real_plant = dict(self.real)
        real_plant["armature"] = self.real["armature"] * self.real.get("armature_plant_scale", 1.0)
        xml = MJ.generate(p, real_plant, plant_scale=scale)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.jid = {n: self.model.joint(f"j_{n}").id for n in ("phi", "ankle", "hip")}
        self.qadr = {n: self.model.jnt_qposadr[j] for n, j in self.jid.items()}
        self.vadr = {n: self.model.jnt_dofadr[j] for n, j in self.jid.items()}
        self.sid_acc = self.model.sensor("imu_acc").adr[0]
        self.sid_gyr = self.model.sensor("imu_gyro").adr[0]
        self.uid_lower = self.model.body("lower").id
        self.uid_upper = self.model.body("upper").id
        self.n_sub = max(1, int(round(p["DT"] / self.real["mj_dt"])))
        # 크랭크 관성 교차검증 (막대근사 params vs MuJoCo 실기하)
        cid = self.model.body("crank").id
        Iy = self.model.body_inertia[cid][1]
        m = self.model.body_mass[cid]
        d = float(np.linalg.norm(self.model.body_ipos[cid]))
        self.rope_check = dict(I_r_mj=float(Iy + m*d*d), S_r_mj=float(m*d),
                               I_r_calc=p["I_r"], S_r_calc=p["S_r"])
        self.reset()

    # ---------------- 리셋/스냅샷 ----------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        a0 = np.deg2rad(self.x0["alpha0"]); t0 = np.deg2rad(self.x0["theta0"])
        f0 = np.deg2rad(self.x0["phi0"])
        self.data.qpos[self.qadr["phi"]] = -f0          # 모델 φ → mj 부호 반전
        self.data.qpos[self.qadr["ankle"]] = a0 - (-f0)  # 월드피치 α = q_phi+q_ankle
        self.data.qpos[self.qadr["hip"]] = t0 - a0
        mujoco.mj_forward(self.model, self.data)
        x = self.get_state()
        self.xh = np.zeros(6); self.xh[0] = x[0]        # v19 초기화 규약: φ는 엔코더로 즉시
        # CF 추정기 상태: [θ_cf, φ_prev(엔코더), φ̇_f(LPF)] — 정지 캘리브 가정으로 θ는 thAcc 초기화
        self.cf = dict(th=None, phi_prev=x[0], phid=0.0, deld=0.0)
        self.xcf = np.zeros(6); self.xcf[0] = x[0]
        # 발목 엔코더 추정기: 각도 3개는 전부 엔코더, 속도는 차분 LPF
        self.enc = dict(prev=None, v=np.zeros(3))   # prev=[φ,α,θ]_enc, v=필터된 [φ̇,α̇,θ̇]
        self.xenc = x.copy()
        # 2-엔코더 칼만 상태: φ·δ 는 즉시 측정, α 는 미지(0 가정) → 동역학 결합으로 수렴
        self.xk2 = np.zeros(6); self.xk2[0] = x[0]; self.xk2[2] = x[2] - x[1]
        self.tau_prev = 0.0
        self.gyro_bias = 0.0
        self.zbuf = []          # 루프 지연 FIFO
        self.fwe = dict(phase=PH_IDLE, t=0.0, d0=0.0, df=0.0, armed=True, cycles=0,
                        settle=0.0, fastL=0.0,   # settle: idle 정착시간, fastL: 고대역 L 잔여시간
                        A_pre=None)              # 온라인 RLS 용 직전 트리거 기록
        # RLS 상태: 파라미터 [Gc, gc], 공분산 P (사전값: 캘리브 or 공칭)
        g0 = self.real.get("fwe_gamma_cal", 0.0) or self.g["mdl"]["fwe"]["GAMMA_CYCLE"]
        Gc0 = self.real.get("fwe_Gc_cal", self.g["mdl"]["fwe"]["G_cyc"])
        self.rls = dict(th=np.array([Gc0, Gc0/g0]), P=np.diag([4.0, 0.05]))
        self.fallen = False
        self.last = dict(tau_cmd=0.0, tau_act=0.0, cur_unit=0, z=None, gated=False)
        self.hist = []          # 스냅샷 링버퍼
        self.step_count = 0

    def snapshot(self):
        return dict(qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(),
                    warm=self.data.qacc_warmstart.copy(), ctrl=self.data.ctrl.copy(),
                    sens=self.data.sensordata.copy(),
                    time=self.data.time, xh=self.xh.copy(), tau_prev=self.tau_prev,
                    gyro_bias=self.gyro_bias, fwe=dict(self.fwe), fallen=self.fallen,
                    cf=dict(self.cf), xcf=self.xcf.copy(),
                    enc=dict(prev=None if self.enc["prev"] is None else self.enc["prev"].copy(),
                             v=self.enc["v"].copy()),
                    xenc=self.xenc.copy(),
                    rls=dict(th=self.rls["th"].copy(), P=self.rls["P"].copy()),
                    zbuf=[(z.copy(), g) for z, g in self.zbuf], xk2=self.xk2.copy(),
                    step=self.step_count)

    def restore(self, s):
        self.data.qpos[:] = s["qpos"]; self.data.qvel[:] = s["qvel"]
        self.data.ctrl[:] = s["ctrl"]             # 직전 인가토크
        self.data.time = s["time"]
        self.xh = s["xh"].copy(); self.tau_prev = s["tau_prev"]
        self.gyro_bias = s["gyro_bias"]; self.fwe = dict(s["fwe"])
        self.cf = dict(s["cf"]); self.xcf = s["xcf"].copy()
        self.enc = dict(prev=None if s["enc"]["prev"] is None else s["enc"]["prev"].copy(),
                        v=s["enc"]["v"].copy())
        self.xenc = s["xenc"].copy()
        self.rls = dict(th=s["rls"]["th"].copy(), P=s["rls"]["P"].copy())
        self.zbuf = [(z.copy(), g) for z, g in s.get("zbuf", [])]
        self.xk2 = s["xk2"].copy() if "xk2" in s else self.xk2
        self.fallen = s["fallen"]; self.step_count = s["step"]
        mujoco.mj_forward(self.model, self.data)
        # ★forward 뒤에 복원해야 함: mj_forward 가 warmstart 를 덮어쓴다 (결정론 핵심)
        self.data.qacc_warmstart[:] = s["warm"]   # frictionloss 제약 솔버 워밍스타트
        self.data.sensordata[:] = s["sens"]       # IMU 1서브스텝 지연 타이밍까지 원 타임라인과 동일하게

    def step_back(self):
        """1 제어스텝 뒤로: 직전 스냅샷(이번 스텝 직전 상태) 복원.
           앞으로 다시 전진하면 같은 스텝이 재실행된다."""
        if self.hist:
            self.restore(self.hist.pop())
            return True
        return False

    # ---------------- 상태 (모델 규약) ----------------
    def get_state(self):
        d, m = self.data, self.model
        phi = -d.qpos[self.qadr["phi"]]
        phid = -d.qvel[self.vadr["phi"]]
        xm_l = d.xmat[self.uid_lower].reshape(3, 3)
        xm_u = d.xmat[self.uid_upper].reshape(3, 3)
        alpha = np.arctan2(-xm_l[2, 0], xm_l[2, 2])
        theta = np.arctan2(-xm_u[2, 0], xm_u[2, 2])
        alphad = d.cvel[self.uid_lower][1]
        thetad = d.cvel[self.uid_upper][1]
        return np.array([phi, alpha, theta, phid, alphad, thetad])

    def risk_A(self, x):
        """A = w4ᵀ[φ, β, φ̇, β̇] (β 환산 rad)"""
        g = self.g
        b = g["p1r"]*x[1] + g["p2r"]*x[2]
        bd = g["p1r"]*x[4] + g["p2r"]*x[5]
        return float(g["mdl"]["fwe"]["w"] @ np.array([x[0], b, x[3], bd]))

    # ---------------- 센서 + 칼만 ----------------
    def _measure(self, x, tau_prev):
        r = self.real
        d = self.data
        # sensordata 는 마지막 서브스텝 이전 상태 기준(=0.5ms IMU 지연, 현실적) — 그대로 사용.
        acc = d.sensordata[self.sid_acc:self.sid_acc+3].copy()
        gyr = d.sensordata[self.sid_gyr:self.sid_gyr+3].copy()
        # 1g 게이팅은 센서 이상화 여부와 무관하게 펌웨어 로직 그대로 상시 적용
        # (격렬 과도구간에서 thAcc 는 물리적으로 θ가 아니다 — 게이팅이 관측 안정성의 핵심)
        anorm = float(np.linalg.norm(acc))
        gated = abs(anorm - 9.81) > r["acc_gate_g"]*9.81
        if r["sensor_on"]:
            # φ 엔코더 양자화
            qph = 2*np.pi / (1 << int(r["enc_bits"]))
            z_phi = np.round(x[0]/qph) * qph
            th_acc = float(np.arctan2(-acc[0], acc[2])) + self.rng.normal(0, r["acc_noise"])
            # 자이로 (+바이어스 랜덤워크)
            self.gyro_bias += self.rng.normal(0, r["gyro_bias_rw"]) * np.sqrt(self.p["DT"])
            th_gyr = float(gyr[1]) + self.gyro_bias + self.rng.normal(0, r["gyro_noise"])
            # 힙 엔코더/속도 양자화
            z_del = np.round(d.qpos[self.qadr["hip"]]/r["hip_enc_quant"]) * r["hip_enc_quant"]
            z_deld = np.round(d.qvel[self.vadr["hip"]]/r["hip_vel_quant"]) * r["hip_vel_quant"]
        else:
            z_phi = x[0]
            th_acc = float(np.arctan2(-acc[0], acc[2]))
            th_gyr = float(gyr[1])
            z_del = x[2] - x[1]
            z_deld = x[5] - x[4]
        return np.array([z_phi, th_acc, th_gyr, z_del, z_deld]), gated

    def cf_update(self, z, gated):
        """상보필터 추정 (모델 무관): θ = 자이로 적분 + thAcc 저속 보정(게이팅 존중),
           α = θ − δ_enc, φ = 엔코더 직접, φ̇ = 엔코더 차분 LPF, α̇ = θ̇ − δ̇."""
        r = self.real; DT = self.p["DT"]
        z_phi, th_acc, th_gyr, z_del, z_deld = z
        c = self.cf
        if c["th"] is None:                      # 최초: 정지 캘리브 (thAcc 로 초기화)
            c["th"] = th_acc
        c["th"] += th_gyr * DT                   # 자이로 적분
        if not gated:
            c["th"] += r["cf_k_acc"] * (th_acc - c["th"])   # 저속 기울기 보정
        a_lpf = DT / (r["cf_tau_phid"] + DT)
        c["phid"] += a_lpf * ((z_phi - c["phi_prev"]) / DT - c["phid"])
        c["phi_prev"] = z_phi
        d_lpf = DT / (r["cf_tau_deld"] + DT)     # δ̇ 양자화 평활
        c["deld"] += d_lpf * (z_deld - c["deld"])
        th = c["th"]
        self.xcf = np.array([z_phi, th - z_del, th,
                             c["phid"], th_gyr - c["deld"], th_gyr])

    def enc_update(self, z, th_gyr):
        """발목 엔코더 추정기: α = q_ankle_enc + q_phi_enc (전부 절대각, 드리프트·가속도 오염 없음).
           속도는 차분+LPF (enc_use_gyro=True 면 θ̇·α̇ 에 자이로 활용)."""
        r = self.real; DT = self.p["DT"]
        d = self.data
        qa = 2*np.pi / (1 << int(r["ankle_enc_bits"]))
        if r["sensor_on"]:
            q_ank = np.round(d.qpos[self.qadr["ankle"]]/qa) * qa
        else:
            q_ank = d.qpos[self.qadr["ankle"]]
        phi = z[0]                       # φ 엔코더
        alpha = q_ank - phi              # α = q_phi+q_ankle, q_phi = −φ
        theta = alpha + z[3]             # θ = α + δ_enc
        ang = np.array([phi, alpha, theta])
        e = self.enc
        if e["prev"] is None:
            e["prev"] = ang.copy()
        a_lpf = DT / (r["enc_tau_v"] + DT)
        e["v"] = e["v"] + a_lpf * ((ang - e["prev"]) / DT - e["v"])
        e["prev"] = ang.copy()
        v = e["v"].copy()
        if r["enc_use_gyro"]:
            v[2] = th_gyr                # θ̇ 자이로 직접
            v[1] = th_gyr - self.cf["deld"]   # α̇ = θ̇ − δ̇(LPF)
        self.xenc = np.array([ang[0], ang[1], ang[2], v[0], v[1], v[2]])

    def kalman(self, z, gated, tau_prev):
        g = self.g
        use_fast = (self.real["fwe_fast_L"]
                    and self.ctrl_mode in (MODE_FWE1, MODE_FWE2, MODE_FWE3)
                    and (self.fwe["phase"] in (PH_FOLD, PH_WAIT, PH_EXTEND, PH_HOLD)
                         or self.fwe["fastL"] > 0.0))
        L = self.g_fast["L"] if use_fast else g["L"]
        Ad, Bd, C, D1 = g["Ad"], g["Bd"], g["C"], g["D1_feed"]
        xp = Ad @ self.xh + Bd[:, 0]*tau_prev
        zp = np.array([xp[0], float(C[1] @ xp) + D1*tau_prev, xp[5],
                       xp[2]-xp[1], xp[5]-xp[4]])
        innov = z - zp
        if gated:
            innov[1] = 0.0          # 1g 게이팅: thAcc 행 혁신 차단 (펌웨어 동일)
        self.xh = xp + L @ innov

    def kalman2(self, z, tau_prev):
        """★2-엔코더 칼만 (발목·IMU 없음): z3 = [φ_enc, δ, δ̇] 만으로 6상태 추정.
        α 직접 측정이 없어 관측성은 '기운 몸이 크랭크를 미는' 동역학 결합에 의존
        — 모델 의존 추정이므로 모델 오차가 α̂ 바이어스로 직결된다 (브링업 한정 권장)."""
        g = self.g
        Ad, Bd = g["Ad"], g["Bd"]
        xp = Ad @ self.xk2 + Bd[:, 0]*tau_prev
        zp = np.array([xp[0], xp[2]-xp[1], xp[5]-xp[4]])
        innov = np.array([z[0], z[3], z[4]]) - zp
        self.xk2 = xp + g["L3"] @ innov

    # ---------------- 제어기 ----------------
    def _Idd(self):
        """힙 δ̈ 유효 관성 = 1/(Eᵀ M⁻¹ E) + armature (피드포워드용, 공칭 모델 기준)"""
        M = self.g["mdl"]["M"]; E = self.g["mdl"]["E"]
        return 1.0/float((E.T @ np.linalg.inv(M) @ E).item())

    def _fold_T(self, dfrom, dto, T_nom):
        """삼각 프로파일 피크속도 2|Δδ|/T 가 서보 무부하속도를 안 넘도록 T 연장"""
        if not self.real.get("fwe_adapt_T", False):
            return T_nom
        vmax = np.deg2rad(self.real["w_noload_dps"]) / self.real["fwe_v_margin"]
        return max(T_nom, 2.0*abs(dto-dfrom)/max(vmax, 1e-6))

    def controller(self, xe):
        p = self.p
        mode = self.ctrl_mode
        if mode == MODE_OFF or self.fallen:
            return 0.0
        if mode == MODE_LQR:
            tau = float(-self.g["K"] @ xe) * self.gain_scale
            fh = self.real.get("lqr_fric_ff", 0.0)
            if fh > 0:
                tau += fh * np.tanh(self.z_now[4] / 0.05)   # δ̇ 방향 쿨롱 보상
            return tau

        if mode == MODE_HYB:
            fwe = self.fwe
            A = self.risk_A(xe)
            delta = self.z_now[3]; deltad = self.z_now[4]
            dmax = np.deg2rad(self.p["DELTA_MAX_DEG"])
            dda = 0.0
            if fwe["phase"] in (PH_IDLE, PH_REST, PH_DONE):
                if abs(A) > np.deg2rad(self.real["hyb_trig_deg"]):
                    df = self._size_fold(A)
                    fwe.update(phase=PH_FOLD, t=0.0, d0=delta, df=df,
                               Tf=self._fold_T(delta, df, self.p["T_FOLD"]),
                               Te=self._fold_T(df, 0.0, self.p["T_EXTEND"]))
                else:
                    return float(-self.g["K"] @ xe) * self.gain_scale
            if fwe["phase"] == PH_FOLD:
                Tf = fwe.get("Tf", self.p["T_FOLD"])
                d_ref, dd_ref, dda = _tri(fwe["t"], Tf, fwe["d0"], fwe["df"])
                if fwe["t"] >= Tf:
                    fwe.update(phase=PH_WAIT, t=0.0)
            elif fwe["phase"] == PH_WAIT:
                d_ref, dd_ref, dda = fwe["df"], 0.0, 0.0
                if fwe["t"] >= self.p["T_WAIT"]:
                    fwe.update(phase=PH_EXTEND, t=0.0)
            else:  # EXTEND
                Te = fwe.get("Te", self.p["T_EXTEND"])
                d_ref, dd_ref, dda = _tri(fwe["t"], Te, fwe["df"], 0.0)
                if fwe["t"] >= Te:
                    fwe["cycles"] += 1
                    fwe.update(phase=PH_IDLE, t=0.0)
            fwe["t"] += self.p["DT"]
            tau = self.kp_servo*(d_ref - delta) + self.kd_servo*(dd_ref - deltad)
            if self.real.get("fwe_ff", False):
                tau += self._Idd() * dda
            return tau

        # ---- FWE (fwe_v19 로직) ----
        fwe = self.fwe
        A = self.risk_A(xe)
        # ★지연 보정 (전상태 예측기): 스칼라 e^{λd} 배율은 실패 — 지연은 진폭만이 아니라
        #   줄 진동 위상도 밀기 때문(30ms ≈ 12°). 축약 4상태 [φ, β, φ̇, β̇] 를 강체(δ 고정)
        #   모델 Φ(d)=expm(A₄d) 로 d 만큼 앞으로 밀어 보낸 뒤 A 를 계산한다.
        if self.real.get("latency_comp") and self.real.get("latency_ms", 0.0) > 0:
            dlat = self.real["latency_ms"] / 1000.0
            Phi = self._phi4_cache.get(dlat)
            if Phi is None:
                from scipy.linalg import expm
                import model_v19 as MD
                md = self.g["mdl"]
                # 모드 분석과 동일한 조립(F4_matrix) 재사용 — 부호 관례 통일
                Phi = expm(MD.F4_matrix(md["M2"], md["G2"], md["D2"]) * dlat)
                self._phi4_cache[dlat] = Phi
            gg = self.g
            b = gg["p1r"]*xe[1] + gg["p2r"]*xe[2]
            bd = gg["p1r"]*xe[4] + gg["p2r"]*xe[5]
            x4p = Phi @ np.array([xe[0], b, xe[3], bd])
            A = float(gg["mdl"]["fwe"]["w"] @ x4p)
        # ★프로파일 추종용 δ·δ̇ 는 직접측정(힙 엔코더 + PRESENT_VELOCITY 원신호).
        #   추정 속도의 LPF 지연(~10ms)이 접기 실행 킥을 ±수십% 요동시켜 발산 원인이 됨.
        delta = self.z_now[3]
        deltad = self.z_now[4]
        gamma = self.g["mdl"]["fwe"]["GAMMA_FOLD"] if mode == MODE_FWE1 \
            else self.g["mdl"]["fwe"]["GAMMA_CYCLE"]
        if mode != MODE_FWE1 and self.real.get("fwe_gamma_cal", 0.0) > 0:
            gamma = self.real["fwe_gamma_cal"]      # 실행 캘리브레이션 γ*
        dmax = np.deg2rad(p["DELTA_MAX_DEG"])
        if mode == MODE_FWE3 and self.real.get("fwe_pos_mode"):
            # ★위치모드의 프로파일·PD·도달판정은 서보 내부(로컬 엔코더, 지연 0).
            #   외부 루프 지연은 트리거용 A 에만 걸린다 — 위치모드의 지연 내성 근원.
            d_loc, dd_loc = getattr(self, "hip_local", (delta, deltad))
            return self._fwe_pos(fwe, A, d_loc, dd_loc)

        if fwe["phase"] == PH_IDLE:
            d_ref, dd_ref, dda = 0.0, 0.0, 0.0
            fwe["settle"] += p["DT"]
            fwe["fastL"] = max(0.0, fwe["fastL"] - p["DT"])
            if (fwe["armed"] and abs(A) > np.deg2rad(p["A_TRIGGER_DEG"])
                    and fwe["settle"] >= self.real["fwe_settle_s"]):
                df = self._size_fold(A)
                fwe.update(phase=PH_FOLD, t=0.0, d0=delta, df=df, A_pre=A,
                           Tf=self._fold_T(delta, df, p["T_FOLD"]),
                           Te=self._fold_T(df, 0.0, p["T_EXTEND"]))
        elif fwe["phase"] == PH_FOLD:
            Tf = fwe.get("Tf", p["T_FOLD"])
            d_ref, dd_ref, dda = _tri(fwe["t"], Tf, fwe["d0"], fwe["df"])
            if fwe["t"] >= Tf:
                nxt = PH_HOLD if mode == MODE_FWE1 else PH_WAIT
                fwe.update(phase=nxt, t=0.0)
        elif fwe["phase"] == PH_HOLD:      # 모드1: 접고 유지 (유계 진동 관찰)
            d_ref, dd_ref, dda = fwe["df"], 0.0, 0.0
        elif fwe["phase"] == PH_WAIT:
            d_ref, dd_ref, dda = fwe["df"], 0.0, 0.0
            if fwe["t"] >= p["T_WAIT"]:
                fwe.update(phase=PH_EXTEND, t=0.0)
        elif fwe["phase"] == PH_EXTEND:
            Te = fwe.get("Te", p["T_EXTEND"])
            d_ref, dd_ref, dda = _tri(fwe["t"], Te, fwe["df"], 0.0)
            if fwe["t"] >= Te:
                fwe["cycles"] += 1
                if mode == MODE_FWE2:
                    fwe.update(phase=PH_DONE, t=0.0, armed=False)   # 1회 후 직립 유지
                else:
                    fwe.update(phase=PH_REST, t=0.0)
        elif fwe["phase"] == PH_REST:
            d_ref, dd_ref, dda = 0.0, 0.0, 0.0
            if fwe["t"] >= p["T_REST"]:
                if self.real.get("fwe_online") and fwe.get("A_pre") is not None:
                    self._rls_update(fwe["A_pre"], fwe["df"], A)
                    fwe["A_pre"] = None
                fwe.update(phase=PH_IDLE, t=0.0, settle=0.0,
                           fastL=self.real["fwe_fastL_hold"])
        else:                               # DONE
            d_ref, dd_ref, dda = 0.0, 0.0, 0.0
        fwe["t"] += p["DT"]
        # 전류모드 PD + 관성 피드포워드 프로파일 추종
        tau = self.kp_servo*(d_ref - delta) + self.kd_servo*(dd_ref - deltad)
        if self.real.get("fwe_ff", False):
            tau += self._Idd() * dda
        return tau

    def _fwe_pos(self, fwe, A, delta, deltad):
        """★위치모드 FWE (2026-07-18 사용자 제안): 서보 내장 제어기에 '목표로 최대속도'.
        - 레퍼런스 = 속도제한 램프(내장 프로파일 발생기 흉내), 추종 = 뻣뻣한 내장 PD.
          pos_kp=400 이면 0.43° 오차부터 토크 포화 → 사실상 '전력 질주'(순간접기에 최근접).
        - 접기/펴기 종료를 시간이 아니라 '도달'로 판정 → 가변 타이밍이 구조에 내장,
          죽음의 나선 1고리(펴기 미완료)가 원리적으로 제거.
        - 대기/휴지/유휴 중 δ=0(또는 δf)을 강하게 물어 경계 잔여(δ₀,δ̇₀) 기계 제거
          (사이클 오차 분해에서 계통분 81% 의 범인이 δ₀·δ̇₀ 였음).
        - IMU 미사용(발목 엔코더) 전제라 저크 충격은 무해. 토크-속도 한계는 servo() 가
          동일하게 적용하므로 물리적으로 공짜는 없음 — 명령 방식만 다름."""
        p, r = self.p, self.real
        vmax = np.deg2rad(min(r["pos_vel_dps"], r["w_noload_dps"]))
        amax = np.deg2rad(r["pos_acc_dps2"])
        tol = np.deg2rad(r["pos_tol_deg"])
        ref = fwe.get("ref", 0.0)
        vref = fwe.get("vref", 0.0)
        tgt = 0.0
        if fwe["phase"] in (PH_IDLE, PH_DONE):
            fwe["settle"] += p["DT"]
            if (fwe["armed"] and abs(A) > np.deg2rad(p["A_TRIGGER_DEG"])
                    and fwe["settle"] >= r["fwe_settle_s"]):
                df = self._size_fold(A)
                fwe.update(phase=PH_FOLD, t=0.0, d0=delta, df=df, A_pre=A)
        arrived = lambda t_: (abs(ref - t_) < 1e-9 and abs(vref) < 1e-9
                              and abs(delta - t_) < tol)
        if fwe["phase"] == PH_FOLD:
            tgt = fwe["df"]
            if arrived(tgt) or fwe["t"] > 0.5:
                fwe.update(phase=PH_WAIT, t=0.0)
        if fwe["phase"] == PH_WAIT:
            tgt = fwe["df"]
            if fwe["t"] >= p["T_WAIT"]:
                fwe.update(phase=PH_EXTEND, t=0.0)
        if fwe["phase"] == PH_EXTEND:
            tgt = 0.0
            if (arrived(0.0) and abs(deltad) < 1.0) or fwe["t"] > 0.5:
                fwe["cycles"] += 1
                fwe.update(phase=PH_REST, t=0.0)
        if fwe["phase"] == PH_REST:
            tgt = 0.0
            if fwe["t"] >= p["T_REST"]:
                if r.get("fwe_online") and fwe.get("A_pre") is not None:
                    self._rls_update(fwe["A_pre"], fwe["df"], A)
                    fwe["A_pre"] = None
                fwe.update(phase=PH_IDLE, t=0.0, settle=0.0, fastL=0.0)
        fwe["t"] += p["DT"]
        # 사다리꼴 프로파일 발생기 (Profile Velocity/Acceleration):
        #   남은 거리 안에서 멈출 수 있으면 가속/순항, 아니면 감속 → 도착 시 v=0
        err = tgt - ref
        if abs(err) < 1e-9 and abs(vref) < amax * p["DT"]:
            ref, vref = tgt, 0.0                     # 도착 스냅
        else:
            stop_d = vref * vref / (2.0 * amax)
            if (vref * np.sign(err) > 0) and (abs(err) <= stop_d):
                vref -= np.sign(vref) * amax * p["DT"]           # 감속 구간
            else:
                vref += np.sign(err) * amax * p["DT"]            # 가속 (방향 전환 포함)
                vref = float(np.clip(vref, -vmax, vmax))
            ref += vref * p["DT"]
            if (tgt - ref) * err < 0:                # 목표 통과 방지
                ref, vref = tgt, 0.0
        fwe["ref"], fwe["vref"] = ref, vref
        return r["pos_kp"]*(ref - delta) + r["pos_kd"]*(vref - deltad)

    # ---------------- 서보 모델 ----------------
    def servo(self, tau_cmd):
        p, r = self.p, self.real
        if not r["servo_on"]:
            tau = float(np.clip(tau_cmd, -p["TAU_MAX"], p["TAU_MAX"]))
            return tau, int(round(tau*1000.0/(p["KT"]*p["CUR_UNIT"])))
        # 명령 경로 (펌웨어와 동일): τ → unit 양자화 → 상한
        unit = int(round(tau_cmd * 1000.0 / (p["KT"] * p["CUR_UNIT"])))
        unit = int(np.clip(unit, -p["CUR_SAFE_UNIT"], p["CUR_SAFE_UNIT"]))
        if abs(unit) < r["cur_deadband"]:
            unit = 0
        # 물리 경로: 실제 Kt(편차 반영) × 전류
        kt_true = p["KT"] * (1.0 + self.mismatch.get("kt", 0.0)/100.0) * r["kt_scale"]
        tau = kt_true * unit * p["CUR_UNIT"] / 1000.0
        # 토크-속도 곡선 (역기전력): 구동 방향 가용토크 감소
        w = float(self.data.qvel[self.vadr["hip"]])
        w_nl = np.deg2rad(r["w_noload_dps"])
        if tau * w > 0:
            avail = r["tau_stall"] * max(0.0, 1.0 - abs(w)/w_nl)
        else:
            avail = r["tau_stall"]
        tau = float(np.clip(tau, -avail, avail))
        # 기어 전달효율(방향 비대칭): 부하 비례 이빨 마찰 — 구동은 깎이고 제동은 거들어진다
        eta = r.get("gear_eta", 1.0)
        if eta < 1.0 and abs(w) > 1e-3:
            tau = tau * eta if tau * w > 0 else tau / eta
        return tau, unit

    # ---------------- 1 제어스텝 ----------------
    def control_step(self):
        self.hist.append(self.snapshot())
        if len(self.hist) > 30000:
            del self.hist[:5000]
        x = self.get_state()
        z, gated = self._measure(x, self.tau_prev)
        # ★루프 지연: 제어기·추정기는 latency_ms 만큼 옛 측정을 본다 (FIFO)
        nlat = int(round(self.real.get("latency_ms", 0.0) / 1000.0 / self.p["DT"]))
        if nlat > 0:
            self.zbuf.append((np.array(z, dtype=float), bool(gated)))
            if len(self.zbuf) > nlat + 1:
                self.zbuf.pop(0)
            z, gated = self.zbuf[0]
        self.z_now = z
        # 서보 내장 루프용 힙 각도·속도 (지연 없음 — 서보 자신의 엔코더는 로컬)
        self.hip_local = (float(self.data.qpos[self.qadr["hip"]]),
                          float(self.data.qvel[self.vadr["hip"]]))
        self.kalman(z, gated, self.tau_prev)
        self.cf_update(z, gated)
        self.enc_update(z, z[2])
        self.kalman2(z, self.tau_prev)
        xe = dict(kalman=self.xh, cf=self.xcf, enc=self.xenc,
                  enc2=self.xk2).get(self.estimator, x)
        if abs(x[1]) > np.deg2rad(80) or abs(x[2]) > np.deg2rad(80):
            self.fallen = True
        tau_cmd = self.controller(xe)
        tau_act, unit = self.servo(tau_cmd)
        self.data.ctrl[0] = tau_act
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        # ★스텝 종료 후 forward: mj_step 이 남긴 xmat/cvel/sensordata 는 마지막 적분 '이전'
        #   상태 기준(1 서브스텝 stale) — 여기서 갱신해야 상태추출·IMU·스텝백 재현이 전부 정합.
        mujoco.mj_forward(self.model, self.data)
        self.tau_prev = tau_act        # 칼만 예측은 '실제 인가 토크' 기준 (φ직접측정이라 안전)
        self.step_count += 1
        self.last = dict(tau_cmd=float(tau_cmd), tau_act=float(tau_act),
                         cur_unit=int(unit), z=z, gated=bool(gated))

    # ---------------- γ* 자동 캘리브레이션 (실물 브링업 절차의 시뮬판) ----------------
    def fwe_autocal(self, n_cycles=12, tilt_deg=0.5, seed=1):
        """아마추어 절차: 로봇을 살짝 기울여 올려놓고 한 사이클 접게 두기를 n회 반복,
           '추정기가 본' (A⁻, δf, A⁺)만으로 A⁺=Gc·A⁻−gc·δf 최소제곱 → γ*=Gc/gc 적용.
           원인(armature? Kt? 질량?)을 몰라도 됨 — 전부 두 숫자에 흡수된다."""
        rng = np.random.default_rng(seed)
        saved = dict(mode=self.ctrl_mode, est=self.estimator, x0=dict(self.x0),
                     rho=self.p["RHO"])
        rows = []
        RHO_STEPS = (0.6, 0.9, 1.2)   # ★접기 세기를 단계별로 바꿔야 Gc·gc 가 분리된다(공선성 방지)
        for k in range(n_cycles):
            a0 = tilt_deg * (0.6 + 0.8*rng.random()) * (1 if k % 2 == 0 else -1)
            self.x0 = dict(phi0=0.0, alpha0=a0, theta0=a0)
            self.ctrl_mode = MODE_FWE3
            self.p["RHO"] = RHO_STEPS[k % len(RHO_STEPS)]
            self.reset()
            Ap = dfp = None
            for _ in range(int(3.0/self.p["DT"])):
                ph0 = self.fwe["phase"]
                self.control_step()
                xe = dict(kalman=self.xh, cf=self.xcf, enc=self.xenc).get(self.estimator)
                if ph0 == PH_IDLE and self.fwe["phase"] == PH_FOLD and Ap is None:
                    Ap, dfp = self.risk_A(xe), self.fwe["df"]
                if Ap is not None and ph0 == PH_REST and self.fwe["phase"] == PH_IDLE:
                    rows.append((Ap, dfp, self.risk_A(xe)))
                    break
                if self.fallen:
                    break
        self.ctrl_mode = saved["mode"]; self.estimator = saved["est"]; self.x0 = saved["x0"]
        self.p["RHO"] = saved["rho"]
        self.reset()
        if len(rows) < 4:
            return None
        a = np.array(rows)
        X = np.column_stack([a[:, 0], -a[:, 1]])
        coef = np.linalg.lstsq(X, a[:, 2], rcond=None)[0]
        Gc, gc = float(coef[0]), float(coef[1])
        if gc <= 0:
            return None
        pred = X @ coef
        rms = float(np.sqrt(np.mean((a[:, 2] - pred)**2)))   # 사이클 맵 잔차 [rad]
        self.real["fwe_gamma_cal"] = Gc / gc
        return dict(Gc=Gc, gc=gc, gamma=Gc/gc, n=len(rows), rms_deg=np.degrees(rms))

    def fwe_autocal_curve(self, n_per=3, tilt_deg=0.5, seed=1,
                          df_grid_deg=(5, 10, 18, 28, 40, 55)):
        """킥 곡선 K(δf) 캘리브레이션 (γ 스칼라의 진폭의존 일반화).
           각 δf 격자에서 n_per회: 작은 기울기로 준비 → 고정 δf 사이클 → K = Gc·A⁻ − A⁺.
           Gc 는 δf 최소 두 단계의 회귀로 추정. 결과는 real['fwe_kick_curve'] 에
           [(δf[rad], K[rad]), ...] 오름차순 저장 — 사이징은 역보간."""
        rng = np.random.default_rng(seed)
        saved = dict(mode=self.ctrl_mode, est=self.estimator, x0=dict(self.x0),
                     rho=self.p["RHO"], cal=self.real.get("fwe_gamma_cal", 0.0))
        rows = []   # (A-, df, A+)
        for df_deg in df_grid_deg:
            for k in range(n_per):
                a0 = tilt_deg*(0.6+0.8*rng.random())*(1 if k % 2 == 0 else -1)
                self.x0 = dict(phi0=0.0, alpha0=a0, theta0=a0)
                self.ctrl_mode = MODE_FWE3
                self.real["_autocal_fix_df"] = np.deg2rad(df_deg)   # 사이징 우회 플래그
                self.reset()
                Ap = None
                for _ in range(int(3.0/self.p["DT"])):
                    ph0 = self.fwe["phase"]
                    self.control_step()
                    xe = dict(kalman=self.xh, cf=self.xcf, enc=self.xenc).get(self.estimator)
                    if ph0 == PH_IDLE and self.fwe["phase"] == PH_FOLD and Ap is None:
                        Ap = self.risk_A(xe)
                    if Ap is not None and ph0 == PH_REST and self.fwe["phase"] == PH_IDLE:
                        rows.append((Ap, abs(self.fwe["df"]), self.risk_A(xe)*np.sign(Ap)))
                        break
                    if self.fallen:
                        break
        self.real.pop("_autocal_fix_df", None)
        self.ctrl_mode = saved["mode"]; self.estimator = saved["est"]; self.x0 = saved["x0"]
        self.p["RHO"] = saved["rho"]
        self.reset()
        if len(rows) < 8:
            return None
        a = np.array(rows)         # A- 부호 정규화됨(양수 기준), df 절대값
        # Gc: 작은 δf 두 단계(선형 구간)에서 A+ = Gc·A- − gc·δf 회귀
        small = a[a[:, 1] <= np.deg2rad(df_grid_deg[1]) + 1e-6]
        X = np.column_stack([np.abs(small[:, 0]), -small[:, 1]])
        cGc, cgc = np.linalg.lstsq(X, small[:, 2], rcond=None)[0]
        Gc = float(cGc)
        # K(δf) = Gc·A- − A+ 를 δf 격자별 평균
        curve = []
        for df_deg in df_grid_deg:
            m = np.isclose(a[:, 1], np.deg2rad(df_deg), atol=1e-6)
            if m.sum():
                K = float(np.mean(Gc*np.abs(a[m, 0]) - a[m, 2]))
                curve.append((float(np.deg2rad(df_deg)), max(K, 0.0)))
        curve.sort()
        self.real["fwe_kick_curve"] = curve
        self.real["fwe_Gc_cal"] = Gc
        return dict(Gc=Gc, curve=[(round(np.degrees(d), 1), round(np.degrees(K), 2))
                                  for d, K in curve], n=len(rows))

    def _rls_update(self, A_pre, df, A_post):
        """A⁺ = Gc·A⁻ − gc·δf 회귀의 재귀 최소제곱 (망각계수) — 자가 노브.
        방어 게이트(옵션): 선형 영역 밖·δf 포화·이상치 사이클은 배우지 않는다."""
        lam = self.real.get("fwe_online_ff", 0.9)
        th, Pm = self.rls["th"], self.rls["P"]
        x = np.array([A_pre, -df])
        y = A_post
        gate = self.real.get("fwe_online_gate_deg", 0.0)
        if gate > 0:
            dmax = np.deg2rad(self.p["DELTA_MAX_DEG"])
            if (abs(A_pre) > np.deg2rad(gate) or abs(A_post) > np.deg2rad(gate)
                    or abs(df) >= dmax - 1e-6):
                return
        rej = self.real.get("fwe_online_rej_deg", 0.0)
        if rej > 0 and abs(y - x @ th) > np.deg2rad(rej):
            return
        if self.real.get("fwe_online_1p"):
            # ★1-노브 RLS: 폐루프에선 δf∝A⁻ (공선성) → 2파라미터는 식별 불가 표류.
            #   Gc 는 구조 지식(e^{λT}, 캘리브)으로 고정하고 gc 하나만 학습 — 서사 그 자체.
            Pg = float(self.rls["P"][1, 1])
            yg = th[0]*A_pre - y                     # gc·δf 관측치
            k1 = Pg*df / (lam + df*Pg*df)
            th[1] = float(np.clip(th[1] + k1*(yg - df*th[1]), 0.02, 1.5))
            self.rls["P"][1, 1] = (Pg - k1*df*Pg) / lam
            self.rls["th"] = th
            return
        Px = Pm @ x
        k = Px / (lam + x @ Px)
        th = th + k * (y - x @ th)
        Pm = (Pm - np.outer(k, Px)) / lam
        th[0] = np.clip(th[0], 1.5, 12.0)           # 물리 범위 가드
        th[1] = np.clip(th[1], 0.02, 1.5)
        self.rls["th"], self.rls["P"] = th, Pm

    def _size_fold(self, A):
        """접기량 결정: 킥곡선 있으면 K(δf)=ρ·Gc·A 역보간, 없으면 δf=ρ·γ·A."""
        p = self.p
        dmax = np.deg2rad(p["DELTA_MAX_DEG"])
        if self.real.get("_autocal_fix_df"):                      # 캘리브 모드: 고정 δf
            return float(np.sign(A) * min(self.real["_autocal_fix_df"], dmax))
        if self.real.get("fwe_online"):
            Gc, gc = self.rls["th"]
            return float(np.clip(p["RHO"]*(Gc/gc)*A, -dmax, dmax))
        curve = self.real.get("fwe_kick_curve")
        if curve:
            Gc = self.real.get("fwe_Gc_cal", 5.3)
            target = p["RHO"] * Gc * abs(A)
            ds = [c[0] for c in curve]; Ks = [c[1] for c in curve]
            Kmax_i = int(np.argmax(Ks))
            if target >= Ks[Kmax_i]:
                df = ds[Kmax_i]                                    # 천장: 최효율점까지만
            else:
                df = float(np.interp(target, Ks[:Kmax_i+1], ds[:Kmax_i+1]))
            return float(np.sign(A) * min(df, dmax))
        gamma = self.real.get("fwe_gamma_cal", 0.0) or self.g["mdl"]["fwe"]["GAMMA_CYCLE"]
        return float(np.clip(p["RHO"]*gamma*A, -dmax, dmax))

    # ---------------- 교란 ----------------
    def perturb(self, dphi_dot):
        """모델 규약 +φ̇ 킥 [rad/s]"""
        self.data.qvel[self.vadr["phi"]] += -dphi_dot

    # ---------------- 프레임 (클라이언트 전송용) ----------------
    def frame(self):
        x = self.get_state()
        d = self.data
        return dict(
            t=round(d.time, 4),
            q=[float(d.qpos[self.qadr["phi"]]), float(d.qpos[self.qadr["ankle"]]),
               float(d.qpos[self.qadr["hip"]])],
            x=[float(v) for v in x],
            xh=[float(v) for v in dict(cf=self.xcf, enc=self.xenc).get(self.estimator, self.xh)],
            A=self.risk_A(x),
            Ae=self.risk_A(dict(cf=self.xcf, enc=self.xenc).get(self.estimator, self.xh)),
            tau_cmd=self.last["tau_cmd"], tau_act=self.last["tau_act"],
            cur=self.last["cur_unit"], fwe=self.fwe["phase"], cyc=self.fwe["cycles"],
            fallen=self.fallen, step=self.step_count,
        )

    def build_info(self):
        md = self.g["mdl"]["modes"]
        fwe = self.g["mdl"]["fwe"]
        rc = self.rope_check
        return dict(
            w_eff=md["w_eff"], l_eff=md["l_eff"], lam_u=fwe["lu"],
            eps_star=self.g["mdl"]["eps_star"],
            gamma_fold=fwe["GAMMA_FOLD"], gamma_cycle=fwe["GAMMA_CYCLE"],
            K=[float(k) for k in self.g["K"]],
            I_r_calc=rc["I_r_calc"], I_r_mj=rc["I_r_mj"],
            S_r_calc=rc["S_r_calc"], S_r_mj=rc["S_r_mj"],
            R=self.p["R"], DT=self.p["DT"],
        )

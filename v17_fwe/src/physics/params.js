/**
 * Slackline Balance Simulator — Physical Parameters (3-DOF Hip Torque Model)
 * 
 * 좌표계: 원호 최저점이 원점, x = 수평(오른쪽 양), y = 수직(위 양)
 * 일반화 좌표:
 *   φ : 원호 위 발 각도 (발 위치 결정)
 *   α : 하체(다리) 기울기 (수직 기준, 독립 자유도)
 *   θ : 상체(몸통) 기울기 (수직 기준, 독립 자유도)
 * 제어 입력: 힙 토크 τ (유일)
 */
var SL = SL || {};

SL.Params = {
    // === 하체 (다리) ===
    m1: 30,         // 하체 질량 (kg)
    L1: 0.85,       // 하체 길이 — 발에서 힙 (m)

    // === 상체 (몸통) ===
    m2: 40,         // 상체 질량 (kg)
    L2: 0.80,       // 상체 길이 — 힙에서 머리 (m)

    // === 줄 ===
    R: 1.0,         // 원호 반경 = 처짐 깊이 (m)
    g: 9.81,        // 중력 가속도 (m/s²)

    // === 렌더링 치수 ===
    lowerWidth: 0.18,  // 하체 폭 (m)
    upperWidth: 0.30,  // 상체 폭 (m)

    // === 감쇠 ===
    b_phi: 2.0,     // 원호 운동 감쇠 (N·m·s/rad)
    b_alpha: 1.0,   // 하체 회전 감쇠 (N·m·s/rad)
    b_theta: 0.5,   // 상체 회전 감쇠 (N·m·s/rad)

    // === 시뮬레이션 ===
    dt: 0.001,
    stepsPerFrame: 16,
    speedMultiplier: 1.0,

    // === PID 제어기 ===
    Kp: 300,
    Kd: 80,
    Ki: 5,
    controllerOn: true,

    // === 초기 조건 ===
    phi0: 0,
    alpha0: 0.1 * Math.PI / 180,
    theta0: 0.1 * Math.PI / 180,
    phiDot0: 0,
    alphaDot0: 0,
    thetaDot0: 0,

    // === 제한 ===
    phiMax: 80 * Math.PI / 180,
    tauMax: 100000,

    // === 파생 상수 (getter) ===
    get ell1() { return this.L1 / 2; },                     // 하체 CoM까지 거리
    get ell2() { return this.L2 / 2; },                     // 상체 CoM까지 거리
    get I1() { return this.m1 * this.L1 * this.L1 / 12; },  // 하체 관성모멘트
    get I2() { return this.m2 * this.L2 * this.L2 / 12; },  // 상체 관성모멘트

    // EOM 유도용 보조 상수
    get p1() { return this.m1 * this.ell1 + this.m2 * this.L1; },  // m₁ℓ₁ + m₂L₁
    get p2() { return this.m2 * this.ell2; },                       // m₂ℓ₂
    get p3() { return this.m2 * this.L1 * this.ell2; },             // m₂L₁ℓ₂
    get Mtotal() { return this.m1 + this.m2; },

    // 질량행렬 대각 상수 (상태 비의존)
    get M11() { return this.Mtotal * this.R * this.R; },
    get M22() { return this.m1 * this.ell1 * this.ell1 + this.I1 + this.m2 * this.L1 * this.L1; },
    get M33() { return this.m2 * this.ell2 * this.ell2 + this.I2; },

    /** UI 슬라이더 범위 */
    ranges: {
        R:       { min: 0.3, max: 5.0, step: 0.1,  label: '원호 반경 R (m)' },
        m1:      { min: 10,  max: 80,  step: 1,     label: '하체 질량 m₁ (kg)' },
        m2:      { min: 10,  max: 100, step: 1,     label: '상체 질량 m₂ (kg)' },
        L1:      { min: 0.4, max: 1.2, step: 0.05,  label: '하체 길이 L₁ (m)' },
        L2:      { min: 0.3, max: 1.2, step: 0.05,  label: '상체 길이 L₂ (m)' },
        b_phi:   { min: 0,   max: 10,  step: 0.1,   label: '원호 감쇠 b_φ' },
        b_alpha: { min: 0,   max: 10,  step: 0.1,   label: '하체 감쇠 b_α' },
        b_theta: { min: 0,   max: 5,   step: 0.1,   label: '상체 감쇠 b_θ' },
        Kp:      { min: 0,   max: 2000, step: 10,   label: 'Kp (비례)' },
        Kd:      { min: 0,   max: 300,  step: 5,    label: 'Kd (미분)' },
        Ki:      { min: 0,   max: 50,   step: 1,    label: 'Ki (적분)' },
        alpha0:  { min: -30, max: 30, step: 0.01, label: '초기 하체 기울기 α₀ (°)', scale: Math.PI / 180 },
        theta0:  { min: -30, max: 30, step: 0.01, label: '초기 상체 기울기 θ₀ (°)', scale: Math.PI / 180 },
    }
};

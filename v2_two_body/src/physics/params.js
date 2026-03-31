/**
 * Slackline Balance Simulator — Physical Parameters (2-Body Hip Torque Model)
 * 
 * 좌표계: 원호 최저점이 원점, x = 줄 수직 수평, y = 수직(위 양)
 * 일반화 좌표: φ (원호 위 발 각도), θ (상체 기울기)
 * 하체 각도: -φ (원호 법선 방향, 자동 결정)
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
    R: 1.5,         // 원호 반경 = 처짐 깊이 (m)
    g: 9.81,        // 중력 가속도 (m/s²)

    // === 렌더링 치수 ===
    lowerWidth: 0.18,  // 하체 폭 (m)
    upperWidth: 0.30,  // 상체 폭 (m)

    // === 감쇠 ===
    b_phi: 2.0,     // 원호 운동 감쇠 (N·m·s/rad)
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
    theta0: 0.15,
    phiDot0: 0,
    thetaDot0: 0,

    // === 제한 ===
    phiMax: 80 * Math.PI / 180,
    tauMax: 1500,

    // === 파생 상수 (getter) ===
    get r() { return this.R - this.L1; },          // 힙의 유효 반경
    get R1() { return this.R - this.L1 / 2; },      // 하체 CoM 유효 반경
    get ell() { return this.L2 / 2; },                // 상체 CoM까지 거리
    get I1() { return this.m1 * this.L1 * this.L1 / 12; }, // 하체 관성모멘트
    get I2() { return this.m2 * this.L2 * this.L2 / 12; }, // 상체 관성모멘트

    // EOM 상수
    get A() { return this.m1 * this.R1 * this.R1 + this.I1 + this.m2 * this.r * this.r; },
    get B() { return this.m2 * this.ell * this.ell + this.I2; },
    get C_() { return this.m2 * this.r * this.ell; }, // C는 예약어 가능성
    get Mg_phi() { return (this.m1 * this.R1 + this.m2 * this.r) * this.g; },
    get Mg_theta() { return this.m2 * this.g * this.ell; },

    /** UI 슬라이더 범위 */
    ranges: {
        R: { min: 0.3, max: 5.0, step: 0.1, label: '원호 반경 R (m)' },
        m1: { min: 10, max: 80, step: 1, label: '하체 질량 m₁ (kg)' },
        m2: { min: 10, max: 100, step: 1, label: '상체 질량 m₂ (kg)' },
        L1: { min: 0.4, max: 1.2, step: 0.05, label: '하체 길이 L₁ (m)' },
        L2: { min: 0.3, max: 1.2, step: 0.05, label: '상체 길이 L₂ (m)' },
        b_phi: { min: 0, max: 10, step: 0.1, label: '원호 감쇠 b_φ' },
        b_theta: { min: 0, max: 5, step: 0.1, label: '상체 감쇠 b_θ' },
        Kp: { min: 0, max: 2000, step: 10, label: 'Kp (비례)' },
        Kd: { min: 0, max: 300, step: 5, label: 'Kd (미분)' },
        Ki: { min: 0, max: 50, step: 1, label: 'Ki (적분)' },
        theta0: { min: -0.5, max: 0.5, step: 0.01, label: '초기 상체 기울기 θ₀' },
    }
};

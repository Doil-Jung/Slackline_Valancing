/**
 * Slackline Balance Simulator V1 — Physical Parameters (1-Body Ankle Torque Model)
 * 
 * 구형 모델 (보존용):
 * 단일 강체(발목부터 머리까지 일체형)이며, 제어 입력이 발목 토크임.
 * 발목으로 몸을 일으키는 현실성 부족 모델. (발전 과정 설명용)
 */
var SL = SL || {};

SL.Params = {
    m: 70,          // 질량 (kg)
    L: 0.85,        // 무게중심 높이 (m) - 발부터 CoM까지 거리
    R: 1.5,         // 원호 반경 (m)
    g: 9.81,        // 중력 가속도

    bodyWidth: 0.3, // 렌더링용 몸 폭

    // 감쇠
    b_phi: 2.0,     // 횡방향(원호) 감쇠 (N·m·s/rad)
    b_theta: 0.5,   // 회전 감쇠 (N·m·s/rad)

    // 시뮬레이션
    dt: 0.001,
    stepsPerFrame: 16,
    speedMultiplier: 1.0,

    // 초기 상태
    phi0: 0,
    theta0: 0.10,
    phiDot0: 0,
    thetaDot0: 0,

    // 제어기 초기값 추가
    Kp: 800,
    Kd: 150,
    Ki: 10,
    controllerOn: true,

    // 한계
    phiMax: 80 * Math.PI / 180,
    tauMax: 1000,

    // 파생 상수 (getter)
    get I() { return this.m * this.L * this.L / 3; }, // 단순 막대 가정 관성모멘트
    get r() { return this.R - this.L; },              // CoM의 실효 회전 반경

    // EOM 계수들
    get A() { return this.m * this.R * this.R; },
    get B() { return this.I + this.m * this.L * this.L; },
    get C() { return this.m * this.R * this.L; },

    // 슬라이더 바인딩용
    ranges: {
        R: { min: 0.3, max: 3.0, step: 0.1, label: '원호 반경 R (m)' },
        L: { min: 0.5, max: 1.5, step: 0.05, label: 'CoM 높이 L (m)' },
        m: { min: 30, max: 120, step: 1, label: '질량 m (kg)' },
        b_phi: { min: 0, max: 10, step: 0.1, label: '원호 감쇠 b_φ' },
        b_theta: { min: 0, max: 5, step: 0.1, label: '몸체 감쇠 b_θ' },
        Kp: { min: 0, max: 2000, step: 10, label: 'Kp (비례)' },
        Kd: { min: 0, max: 300, step: 5, label: 'Kd (미분)' },
        Ki: { min: 0, max: 50, step: 1, label: 'Ki (적분)' },
        theta0: { min: -0.5, max: 0.5, step: 0.01, label: '초기 기울기 θ₀ (rad)' },
    }
};

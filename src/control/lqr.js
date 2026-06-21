/**
 * Slackline Balance Simulator — LQR Controller (Hip Torque)
 * 
 * 최적 선형 상태 피드백 제어기:
 *   τ = −K · x = −(k₁φ + k₂α + k₃θ + k₄φ̇ + k₅α̇ + k₆θ̇)
 * 
 * K는 연속시간 대수 리카티 방정식(CARE)의 해에서 도출:
 *   Q = diag(1, 50, 50, 0.1, 5, 5),  R = 0.001
 * 
 * 행렬 부호 함수법으로 계산됨 (analysis/controllability.html 참조)
 */
var SL = SL || {};

SL.LQRController = class {
    constructor() {
        // 기본 LQR 게인 (analysis/controllability.html에서 계산된 값)
        this.K = [-6454.691363, -5908.227575, -2148.820176, -1312.073937, -1095.998565, -579.948137];

        // 게인 스케일링 (UI 조절용, 1.0 = 기본)
        this.gainScale = 1.0;

        // 토크 포화 방지용 적분 안티와인드업
        this.integral = 0;
        this.maxIntegral = 50;
        this.Ki = 0; // 적분 게인 (선택적, 기본=0)
    }

    /**
     * @param {Object} state - {phi, alpha, theta, phiDot, alphaDot, thetaDot}
     * @param {Object} model - SL.Model 인스턴스
     * @param {number} dt - 시간 스텝
     * @returns {number} 힙 토크 τ
     */
    compute(state, model, dt) {
        const x = [
            state.phi,      // φ  (발 원호 위치)
            state.alpha,    // α  (하체 기울기)
            state.theta,    // θ  (상체 기울기)
            state.phiDot,   // φ̇ (발 속도)
            state.alphaDot, // α̇ (하체 각속도)
            state.thetaDot  // θ̇ (상체 각속도)
        ];

        // τ = −K · x
        let tau = 0;
        for (let i = 0; i < 6; i++) {
            tau -= this.K[i] * x[i];
        }

        // 게인 스케일링 적용
        tau *= this.gainScale;

        // 선택적 적분항 (정상상태 오차 제거)
        if (this.Ki > 0) {
            const comError = model.getCoMError(state);
            this.integral += comError * dt;
            this.integral = Math.max(-this.maxIntegral, Math.min(this.maxIntegral, this.integral));
            tau -= this.Ki * this.integral;
        }

        return tau;
    }

    setGainScale(scale) { this.gainScale = scale; }
    setKi(Ki) { this.Ki = Ki; }
    reset() { this.integral = 0; }
};

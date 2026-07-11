/**
 * Slackline Balance Simulator — PID Controller (Hip Torque, CoM-based)
 * 
 * 제어 목표: 전체 CoM의 x좌표가 발(지지점)의 x좌표 위에 유지되도록
 * 오차 = x_com − x_foot
 * 힙 토크 τ 출력: τ < 0 → 오차 양수(CoM이 발 오른쪽)일 때 복원
 */
var SL = SL || {};

SL.PIDController = class {
    constructor(Kp, Kd, Ki) {
        this.Kp = Kp || 300;
        this.Kd = Kd || 80;
        this.Ki = Ki || 5;
        this.integral = 0;
        this.maxIntegral = 30;
        this.prevError = 0;
    }

    /**
     * @param {Object} state - 현재 상태 {phi, alpha, theta, phiDot, alphaDot, thetaDot}
     * @param {Object} model - SL.Model 인스턴스 (getCoMError 등 메서드 접근)
     * @param {number} dt - 시간 스텝
     * @returns {number} 힙 토크 τ
     */
    compute(state, model, dt) {
        // 균형 오차: CoM x - 발 x
        const error = model.getCoMError(state);

        // 오차의 미분 (수치 미분)
        const errorDot = (error - this.prevError) / dt;
        this.prevError = error;

        // 적분 (anti-windup)
        this.integral += error * dt;
        this.integral = Math.max(-this.maxIntegral, Math.min(this.maxIntegral, this.integral));

        // PID 출력: 오차가 양수(CoM이 발 오른쪽) → 음의 토크로 복원
        return -(this.Kp * error + this.Kd * errorDot + this.Ki * this.integral);
    }

    setGains(Kp, Kd, Ki) { this.Kp = Kp; this.Kd = Kd; this.Ki = Ki; }
    reset() { this.integral = 0; this.prevError = 0; }
};

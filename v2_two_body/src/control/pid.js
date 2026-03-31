/**
 * Slackline Balance Simulator — PID Controller (Hip Torque)
 * 
 * 힙 토크 τ를 출력하여 θ(상체 기울기) → 0으로 수렴.
 * τ < 0 (θ > 0일 때): 상체 복원 + 반작용으로 발 우측 이동 (사이드 서핑)
 */
var SL = SL || {};

SL.PIDController = class {
    constructor(Kp, Kd, Ki) {
        this.Kp = Kp || 300;
        this.Kd = Kd || 80;
        this.Ki = Ki || 5;
        this.integral = 0;
        this.maxIntegral = 30;
    }

    compute(theta, thetaDot, dt) {
        this.integral += theta * dt;
        this.integral = Math.max(-this.maxIntegral, Math.min(this.maxIntegral, this.integral));
        return -(this.Kp * theta + this.Kd * thetaDot + this.Ki * this.integral);
    }

    setGains(Kp, Kd, Ki) { this.Kp = Kp; this.Kd = Kd; this.Ki = Ki; }
    reset() { this.integral = 0; }
};

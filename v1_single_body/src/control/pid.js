/**
 * Slackline Balance Simulator V1 — PID Controller (Ankle Torque)
 */
var SL = SL || {};

SL.PIDController = class {
    constructor(Kp, Kd, Ki) {
        this.Kp = Kp || 800;
        this.Kd = Kd || 150;
        this.Ki = Ki || 10;
        this.integral = 0;
        this.maxIntegral = 50;
    }

    compute(theta, thetaDot, dt) {
        this.integral += theta * dt;
        this.integral = Math.max(-this.maxIntegral, Math.min(this.maxIntegral, this.integral));

        // 역진자 음의 피드백 (θ > 0 → τ < 0 → 발목이 몸을 되돌림)
        return -(this.Kp * theta + this.Kd * thetaDot + this.Ki * this.integral);
    }

    setGains(Kp, Kd, Ki) {
        this.Kp = Kp;
        this.Kd = Kd;
        this.Ki = Ki;
    }

    reset() {
        this.integral = 0;
    }
};

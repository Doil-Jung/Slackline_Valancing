/**
 * Slackline Balance Simulator V9 — 센서 노이즈 생성기
 * 
 * - 각도 측정 노이즈 (가우시안 백색 노이즈)
 * - 각속도 측정 노이즈 (가우시안 + 자이로 바이어스 드리프트)
 * 
 * Box-Muller 변환으로 가우시안 난수 생성
 */
var SL = SL || {};

SL.SensorNoise = class {
    constructor() {
        // === 설정 ===
        this.angleNoiseSigma = 0;      // 각도 노이즈 σ (rad), 예: 0.005 ≈ 0.3°
        this.gyroNoiseSigma = 0;       // 각속도 노이즈 σ (rad/s), 예: 0.02
        this.gyroBiasDrift = 0;        // 자이로 바이어스 드리프트 강도 (rad/s²)

        // === 내부 상태 ===
        this._biasAlphaDot = 0;        // α̇ 자이로 바이어스 (누적)
        this._biasThetaDot = 0;        // θ̇ 자이로 바이어스 (누적)
        this._dt = 0.001;              // 시뮬레이션 dt

        // Box-Muller 캐시
        this._hasSpare = false;
        this._spare = 0;
    }

    /** 초기화 */
    init(dt) {
        this._dt = dt || 0.001;
        this.reset();
    }

    /** 바이어스 리셋 */
    reset() {
        this._biasAlphaDot = 0;
        this._biasThetaDot = 0;
        this._hasSpare = false;
    }

    /**
     * Box-Muller 변환으로 표준 정규분포 난수 생성
     * @returns {number} N(0, 1)
     */
    _randn() {
        if (this._hasSpare) {
            this._hasSpare = false;
            return this._spare;
        }

        let u, v, s;
        do {
            u = Math.random() * 2 - 1;
            v = Math.random() * 2 - 1;
            s = u * u + v * v;
        } while (s >= 1 || s === 0);

        const mul = Math.sqrt(-2 * Math.log(s) / s);
        this._spare = v * mul;
        this._hasSpare = true;
        return u * mul;
    }

    /**
     * 실제 상태에 센서 노이즈를 추가하여 "측정값" 반환
     * @param {Object} trueState - 실제 물리 상태 {alpha, theta, alphaDot, thetaDot, ...}
     * @returns {Object} 노이즈가 추가된 센서 읽기값
     */
    addNoise(trueState) {
        // 각도 노이즈
        const nAlpha = this.angleNoiseSigma > 0 ? this._randn() * this.angleNoiseSigma : 0;
        const nTheta = this.angleNoiseSigma > 0 ? this._randn() * this.angleNoiseSigma : 0;

        // 자이로 바이어스 랜덤 워크 (드리프트)
        if (this.gyroBiasDrift > 0) {
            this._biasAlphaDot += this._randn() * this.gyroBiasDrift * this._dt;
            this._biasThetaDot += this._randn() * this.gyroBiasDrift * this._dt;
        }

        // 각속도 노이즈 (백색 + 바이어스)
        const nAlphaDot = (this.gyroNoiseSigma > 0 ? this._randn() * this.gyroNoiseSigma : 0) + this._biasAlphaDot;
        const nThetaDot = (this.gyroNoiseSigma > 0 ? this._randn() * this.gyroNoiseSigma : 0) + this._biasThetaDot;

        return {
            alpha: trueState.alpha + nAlpha,
            theta: trueState.theta + nTheta,
            alphaDot: trueState.alphaDot + nAlphaDot,
            thetaDot: trueState.thetaDot + nThetaDot,
            // 통으로 복사 (phi 등)
            phi: trueState.phi,
            phiDot: trueState.phiDot,
            time: trueState.time
        };
    }

    /** 현재 노이즈 수준 요약 */
    getNoiseSummary() {
        if (this.angleNoiseSigma === 0 && this.gyroNoiseSigma === 0 && this.gyroBiasDrift === 0) {
            return '노이즈 없음';
        }
        const parts = [];
        if (this.angleNoiseSigma > 0) parts.push(`각도σ=${(this.angleNoiseSigma * 180 / Math.PI).toFixed(2)}°`);
        if (this.gyroNoiseSigma > 0) parts.push(`자이로σ=${this.gyroNoiseSigma.toFixed(3)} rad/s`);
        if (this.gyroBiasDrift > 0)  parts.push(`바이어스=${this.gyroBiasDrift.toFixed(2)} rad/s²`);
        return parts.join(', ');
    }

    /** 현재 자이로 바이어스 값 */
    getBiasInfo() {
        return {
            biasAlphaDot: this._biasAlphaDot,
            biasThetaDot: this._biasThetaDot
        };
    }
};

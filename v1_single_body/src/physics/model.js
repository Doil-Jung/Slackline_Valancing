/**
 * Slackline Balance Simulator V1 — Physics Model (1-Body Ankle Torque)
 * 
 * 구형 모델 (보존용)
 * 좌표계: 원호 기준, 역진자. 
 * 제어 입력: 단일 발목 토크 (tau)
 */
var SL = SL || {};

SL.Model = class {
    constructor(params) {
        this.params = params;
        this.reset();
    }

    reset(params) {
        if (params) this.params = params;
        const p = this.params;
        this.state = {
            phi: p.phi0 || 0,
            theta: p.theta0 || 0.10,
            phiDot: p.phiDot0 || 0,
            thetaDot: p.thetaDot0 || 0,
            time: 0
        };
        this.tau = 0;
    }

    computeDerivatives(state, tau) {
        const p = this.params;
        const { phi, theta, phiDot, thetaDot } = state;
        const A = p.A, B = p.B, C = p.C;

        const sum = phi + theta;
        const sinS = Math.sin(sum);
        const cosS = Math.cos(sum);

        const Cc = C * cosS;
        const Cs = C * sinS;

        // B = [-1, +1]^T (tau 작용)
        const f1 = Cs * thetaDot * thetaDot - p.m * p.g * p.R * Math.sin(phi) - p.b_phi * phiDot - tau;
        const f2 = Cs * phiDot * phiDot + p.m * p.g * p.L * Math.sin(theta) - p.b_theta * thetaDot + tau;

        const det = A * B - Cc * Cc;
        if (Math.abs(det) < 1e-12) return [phiDot, thetaDot, 0, 0];

        const phiDDot = (B * f1 - Cc * f2) / det;
        const thetaDDot = (A * f2 - Cc * f1) / det;

        return [phiDot, thetaDot, phiDDot, thetaDDot];
    }

    rk4Step(state, tau, dt) {
        const s = [state.phi, state.theta, state.phiDot, state.thetaDot];
        const toObj = (a) => ({ phi: a[0], theta: a[1], phiDot: a[2], thetaDot: a[3] });

        const k1 = this.computeDerivatives(toObj(s), tau);
        const s2 = s.map((v, i) => v + 0.5 * dt * k1[i]);
        const k2 = this.computeDerivatives(toObj(s2), tau);
        const s3 = s.map((v, i) => v + 0.5 * dt * k2[i]);
        const k3 = this.computeDerivatives(toObj(s3), tau);
        const s4 = s.map((v, i) => v + dt * k3[i]);
        const k4 = this.computeDerivatives(toObj(s4), tau);

        const ns = s.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
        return { phi: ns[0], theta: ns[1], phiDot: ns[2], thetaDot: ns[3], time: state.time + dt };
    }

    step(controller) {
        const p = this.params;
        const steps = Math.round(p.stepsPerFrame * p.speedMultiplier);

        for (let i = 0; i < steps; i++) {
            if (controller && p.controllerOn) {
                // 음성 피드백: θ가 양수이면 τ를 양의 방향으로 하여 발목을 폄
                this.tau = controller.compute(this.state.theta, this.state.thetaDot, p.dt);
                this.tau = Math.max(-p.tauMax, Math.min(p.tauMax, this.tau));
            } else {
                this.tau = 0;
            }
            this.state = this.rk4Step(this.state, this.tau, p.dt);

            if (Math.abs(this.state.phi) > p.phiMax) {
                this.state.phi = Math.sign(this.state.phi) * p.phiMax;
                this.state.phiDot *= -0.3;
            }
        }
        return this.state;
    }

    getFootPos(st) {
        st = st || this.state;
        const R = this.params.R;
        return { x: R * Math.sin(st.phi), y: R * (1 - Math.cos(st.phi)) };
    }

    getCoMPos(st) {
        st = st || this.state;
        const R = this.params.R, L = this.params.L;
        const foot = this.getFootPos(st);
        return {
            x: foot.x + L * Math.sin(st.theta),
            y: foot.y + L * Math.cos(st.theta)
        };
    }

    getEnergy(st) {
        st = st || this.state;
        const p = this.params;
        const sum = st.phi + st.theta;
        const vx = p.R * Math.cos(st.phi) * st.phiDot + p.L * Math.cos(st.theta) * st.thetaDot;
        const vy = p.R * Math.sin(st.phi) * st.phiDot - p.L * Math.sin(st.theta) * st.thetaDot;
        const T = 0.5 * p.m * (vx * vx + vy * vy) + 0.5 * p.I * st.thetaDot * st.thetaDot;
        const com = this.getCoMPos(st);
        const V = p.m * p.g * com.y;
        return { kinetic: T, potential: V, total: T + V };
    }

    applyPerturbation(impulse) {
        this.state.thetaDot += impulse;
    }
};

/**
 * Slackline Balance Simulator — 2-Body Physics Model
 * 
 * 하체(다리): 원호 법선에 고정, 각도 = -φ (발목 관절 없음)
 * 상체(몸통): 힙 관절에서 자유 회전, 각도 = θ
 * 제어 입력: 힙 토크 τ (유일)
 * 
 * 일반화 좌표: q = [φ, θ]
 * 일반화 힘:   Q = [τ, τ]  (힙 상대각 = θ+φ에 작용)
 * 
 * EOM:
 *   [A,  Cc]  [φ̈]   [Cs θ̇² − Mg_φ sinφ − b_φ φ̇ + τ]
 *   [Cc, B ]  [θ̈] = [Cs φ̇² + Mg_θ sinθ − b_θ θ̇ + τ]
 * 
 *   Cc = C·cos(φ+θ),  Cs = C·sin(φ+θ)
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
            theta: p.theta0 || 0.15,
            phiDot: p.phiDot0 || 0,
            thetaDot: p.thetaDot0 || 0,
            time: 0
        };
        this.tau = 0;
    }

    /** 운동방정식 우변: 반환 [φ̇, θ̇, φ̈, θ̈] */
    computeDerivatives(state, tau) {
        const p = this.params;
        const { phi, theta, phiDot, thetaDot } = state;
        const A = p.A, B = p.B, C = p.C_;
        const Mg_phi = p.Mg_phi, Mg_theta = p.Mg_theta;

        const sum = phi + theta;
        const sinS = Math.sin(sum);
        const cosS = Math.cos(sum);

        const Cc = C * cosS;   // 질량행렬 비대각 성분
        const Cs = C * sinS;   // 코리올리 계수

        // 우변 벡터
        const f1 = Cs * thetaDot * thetaDot
            - Mg_phi * Math.sin(phi)
            - p.b_phi * phiDot
            + tau;

        const f2 = Cs * phiDot * phiDot
            + Mg_theta * Math.sin(theta)
            - p.b_theta * thetaDot
            + tau;

        // 2×2 역행렬로 가속도 계산
        const det = A * B - Cc * Cc;
        if (Math.abs(det) < 1e-12) {
            return [phiDot, thetaDot, 0, 0];
        }

        const phiDDot = (B * f1 - Cc * f2) / det;
        const thetaDDot = (A * f2 - Cc * f1) / det;

        return [phiDot, thetaDot, phiDDot, thetaDDot];
    }

    /** RK4 1스텝 */
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

    /** 프레임 스텝 */
    step(controller) {
        const p = this.params;
        const steps = Math.round(p.stepsPerFrame * p.speedMultiplier);

        for (let i = 0; i < steps; i++) {
            if (controller && p.controllerOn) {
                this.tau = controller.compute(this.state.theta, this.state.thetaDot, p.dt);
                this.tau = Math.max(-p.tauMax, Math.min(p.tauMax, this.tau));
            } else {
                this.tau = 0;
            }
            this.state = this.rk4Step(this.state, this.tau, p.dt);

            // φ 소프트 리밋
            if (Math.abs(this.state.phi) > p.phiMax) {
                this.state.phi = Math.sign(this.state.phi) * p.phiMax;
                this.state.phiDot *= -0.3;
            }
        }
        return this.state;
    }

    /** 발 위치 */
    getFootPos(st) {
        st = st || this.state;
        const R = this.params.R;
        return { x: R * Math.sin(st.phi), y: R * (1 - Math.cos(st.phi)) };
    }

    /** 힙 위치 */
    getHipPos(st) {
        st = st || this.state;
        const { R, r } = this.params;
        return { x: r * Math.sin(st.phi), y: R - r * Math.cos(st.phi) };
    }

    /** 하체 CoM 위치 */
    getLowerCoM(st) {
        st = st || this.state;
        const { R, R1 } = this.params;
        return { x: R1 * Math.sin(st.phi), y: R - R1 * Math.cos(st.phi) };
    }

    /** 상체 CoM 위치 */
    getUpperCoM(st) {
        st = st || this.state;
        const { R, r, ell } = this.params;
        return {
            x: r * Math.sin(st.phi) + ell * Math.sin(st.theta),
            y: R - r * Math.cos(st.phi) + ell * Math.cos(st.theta)
        };
    }

    /** 전체 CoM */
    getTotalCoM(st) {
        st = st || this.state;
        const { m1, m2 } = this.params;
        const lc = this.getLowerCoM(st);
        const uc = this.getUpperCoM(st);
        const M = m1 + m2;
        return {
            x: (m1 * lc.x + m2 * uc.x) / M,
            y: (m1 * lc.y + m2 * uc.y) / M
        };
    }

    /** 상체 꼭대기 위치 (렌더링용) */
    getHeadPos(st) {
        st = st || this.state;
        const hip = this.getHipPos(st);
        const { L2 } = this.params;
        return {
            x: hip.x + L2 * Math.sin(st.theta),
            y: hip.y + L2 * Math.cos(st.theta)
        };
    }

    /** 총 에너지 */
    getEnergy(st) {
        st = st || this.state;
        const { m1, m2, R, R1, r, ell, I1, I2, g } = this.params;

        // 하체 CoM 속도
        const vx1 = R1 * Math.cos(st.phi) * st.phiDot;
        const vy1 = R1 * Math.sin(st.phi) * st.phiDot;
        // 상체 CoM 속도
        const vx2 = r * Math.cos(st.phi) * st.phiDot + ell * Math.cos(st.theta) * st.thetaDot;
        const vy2 = r * Math.sin(st.phi) * st.phiDot - ell * Math.sin(st.theta) * st.thetaDot;

        const T = 0.5 * m1 * (vx1 * vx1 + vy1 * vy1) + 0.5 * I1 * st.phiDot * st.phiDot
            + 0.5 * m2 * (vx2 * vx2 + vy2 * vy2) + 0.5 * I2 * st.thetaDot * st.thetaDot;

        const lc = this.getLowerCoM(st);
        const uc = this.getUpperCoM(st);
        const V = m1 * g * lc.y + m2 * g * uc.y;

        return { kinetic: T, potential: V, total: T + V };
    }

    /** 외부 교란 */
    applyPerturbation(impulse) {
        this.state.thetaDot += impulse;
    }
};

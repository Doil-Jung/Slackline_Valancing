/**
 * Slackline Balance Simulator — 3-DOF Physics Model
 * 
 * 일반화 좌표: q = [φ, α, θ]
 *   φ : 원호 위 발 각도 (발 위치 = R sinφ, R(1−cosφ))
 *   α : 하체(다리) 기울기 (수직 기준, 독립 자유도)
 *   θ : 상체(몸통) 기울기 (수직 기준, 독립 자유도)
 * 
 * 구속 조건: 발은 반경 R 원호 위에서 미끄러짐 (자유 힌지, 발목 토크 없음)
 * 제어 입력: 힙 토크 τ (유일)
 *   → α 방정식에 −τ, θ 방정식에 +τ (힙 상대각 θ−α에 작용)
 * 
 * EOM (라그랑주 역학):
 *   M(q) q̈ = f(q, q̇, τ)
 * 
 * 질량행렬 M (대칭 3×3):
 *   M₁₁ = (m₁+m₂)R²
 *   M₁₂ = R·p₁·cos(φ+α)
 *   M₁₃ = R·p₂·cos(φ+θ)
 *   M₂₂ = m₁ℓ₁² + I₁ + m₂L₁²
 *   M₂₃ = p₃·cos(α−θ)
 *   M₃₃ = m₂ℓ₂² + I₂
 * 
 * RHS f (코리올리/원심력 + 중력 + 토크 + 감쇠):
 *   f₁ = R·p₁·sin(φ+α)·α̇² + R·p₂·sin(φ+θ)·θ̇² − (m₁+m₂)gR·sinφ − b_φ·φ̇
 *   f₂ = R·p₁·sin(φ+α)·φ̇² − p₃·sin(α−θ)·θ̇² + p₁·g·sinα − τ − b_α·α̇
 *   f₃ = R·p₂·sin(φ+θ)·φ̇² + p₃·sin(α−θ)·α̇² + p₂·g·sinθ + τ − b_θ·θ̇
 * 
 * 여기서:
 *   ℓ₁ = L₁/2,  ℓ₂ = L₂/2
 *   p₁ = m₁ℓ₁ + m₂L₁
 *   p₂ = m₂ℓ₂
 *   p₃ = m₂L₁ℓ₂
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
            alpha: p.alpha0 || 0,
            theta: p.theta0 || 0.15,
            phiDot: p.phiDot0 || 0,
            alphaDot: p.alphaDot0 || 0,
            thetaDot: p.thetaDot0 || 0,
            time: 0
        };
        this.tau = 0;
    }

    /**
     * 3×3 대칭 행렬의 역행렬 × 벡터  (M⁻¹ · f)
     * M = [[a,b,c],[b,d,e],[c,e,f]],  v = [v1,v2,v3]
     * 반환: [x1, x2, x3] where M·x = v
     */
    solve3x3(M, v) {
        const [a, b, c, d, e, f] = [M[0][0], M[0][1], M[0][2], M[1][1], M[1][2], M[2][2]];
        // Determinant of symmetric 3x3
        const det = a * (d * f - e * e)
            - b * (b * f - e * c)
            + c * (b * e - d * c);

        if (Math.abs(det) < 1e-12) {
            return [0, 0, 0]; // 특이 행렬 방지
        }

        const invDet = 1.0 / det;

        // Cofactors (symmetric matrix)
        const A11 = d * f - e * e;
        const A12 = -(b * f - c * e);
        const A13 = b * e - c * d;
        const A22 = a * f - c * c;
        const A23 = -(a * e - b * c);
        const A33 = a * d - b * b;

        return [
            (A11 * v[0] + A12 * v[1] + A13 * v[2]) * invDet,
            (A12 * v[0] + A22 * v[1] + A23 * v[2]) * invDet,
            (A13 * v[0] + A23 * v[1] + A33 * v[2]) * invDet
        ];
    }

    /** 운동방정식: 반환 [φ̇, α̇, θ̇, φ̈, α̈, θ̈] */
    computeDerivatives(state, tau) {
        const p = this.params;
        const { phi, alpha, theta, phiDot, alphaDot, thetaDot } = state;

        const R = p.R, g = p.g;
        const p1 = p.p1, p2 = p.p2, p3 = p.p3;
        const Mtotal = p.Mtotal;

        // 삼각함수 캐싱
        const sinPA = Math.sin(phi + alpha);
        const cosPA = Math.cos(phi + alpha);
        const sinPT = Math.sin(phi + theta);
        const cosPT = Math.cos(phi + theta);
        const sinAT = Math.sin(alpha - theta);
        const cosAT = Math.cos(alpha - theta);
        const sinPhi = Math.sin(phi);
        const sinAlpha = Math.sin(alpha);
        const sinTheta = Math.sin(theta);

        // === 질량행렬 M (대칭) ===
        const M11 = p.M11;    // (m₁+m₂)R²
        const M12 = R * p1 * cosPA;
        const M13 = R * p2 * cosPT;
        const M22 = p.M22;    // m₁ℓ₁²+I₁+m₂L₁²
        const M23 = p3 * cosAT;
        const M33 = p.M33;    // m₂ℓ₂²+I₂

        const M = [
            [M11, M12, M13],
            [M12, M22, M23],
            [M13, M23, M33]
        ];

        // === RHS 벡터 f ===
        const f1 = R * p1 * sinPA * alphaDot * alphaDot
            + R * p2 * sinPT * thetaDot * thetaDot
            - Mtotal * g * R * sinPhi
            - p.b_phi * phiDot;

        const f2 = R * p1 * sinPA * phiDot * phiDot
            - p3 * sinAT * thetaDot * thetaDot
            + p1 * g * sinAlpha
            - tau
            - p.b_alpha * alphaDot;

        const f3 = R * p2 * sinPT * phiDot * phiDot
            + p3 * sinAT * alphaDot * alphaDot
            + p2 * g * sinTheta
            + tau
            - p.b_theta * thetaDot;

        // === 가속도 = M⁻¹ · f ===
        const accel = this.solve3x3(M, [f1, f2, f3]);

        return [phiDot, alphaDot, thetaDot, accel[0], accel[1], accel[2]];
    }

    /** RK4 1스텝 */
    rk4Step(state, tau, dt) {
        const s = [state.phi, state.alpha, state.theta,
        state.phiDot, state.alphaDot, state.thetaDot];
        const toObj = (a) => ({
            phi: a[0], alpha: a[1], theta: a[2],
            phiDot: a[3], alphaDot: a[4], thetaDot: a[5]
        });

        const k1 = this.computeDerivatives(toObj(s), tau);
        const s2 = s.map((v, i) => v + 0.5 * dt * k1[i]);
        const k2 = this.computeDerivatives(toObj(s2), tau);
        const s3 = s.map((v, i) => v + 0.5 * dt * k2[i]);
        const k3 = this.computeDerivatives(toObj(s3), tau);
        const s4 = s.map((v, i) => v + dt * k3[i]);
        const k4 = this.computeDerivatives(toObj(s4), tau);

        const ns = s.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
        return {
            phi: ns[0], alpha: ns[1], theta: ns[2],
            phiDot: ns[3], alphaDot: ns[4], thetaDot: ns[5],
            time: state.time + dt
        };
    }

    /** 프레임 스텝 */
    step(controller) {
        const p = this.params;
        const steps = Math.round(p.stepsPerFrame * p.speedMultiplier);

        for (let i = 0; i < steps; i++) {
            if (controller && p.controllerOn) {
                this.tau = controller.compute(this.state, this, p.dt);
                this.tau = Math.max(-p.tauMax, Math.min(p.tauMax, this.tau));
            } else {
                this.tau = 0;
            }
            this.state = this.rk4Step(this.state, this.tau, p.dt);

            // φ 소프트 리밋 (줄 끝에서 반사)
            if (Math.abs(this.state.phi) > p.phiMax) {
                this.state.phi = Math.sign(this.state.phi) * p.phiMax;
                this.state.phiDot *= -0.3;
            }

            // NaN/발산 감지 → 시뮬레이션 중지 (발산 숨기기 방지)
            const s = this.state;
            if (isNaN(s.phi) || isNaN(s.alpha) || isNaN(s.theta) ||
                !isFinite(s.phi) || !isFinite(s.alpha) || !isFinite(s.theta) ||
                Math.abs(s.alpha) > 2*Math.PI || Math.abs(s.theta) > 2*Math.PI) {
                console.error('⚠ State DIVERGED!', 
                    {phi: s.phi?.toFixed(4), alpha: s.alpha?.toFixed(4), 
                     theta: s.theta?.toFixed(4), tau: this.tau?.toFixed(4), 
                     time: s.time?.toFixed(4)});
                // 마지막 유효 상태로 되돌리기 (NaN 방지)
                this.state = {phi: 0, alpha: 0, theta: 0, phiDot: 0, alphaDot: 0, thetaDot: 0, time: s.time || 0};
                this.tau = 0;
                this._diverged = true;  // 외부에서 감지 가능
                break;
            }
        }
        return this.state;
    }

    // ================================================================
    //   기하학 (위치 계산)
    // ================================================================

    /** 발 위치 (원호 구속) */
    getFootPos(st) {
        st = st || this.state;
        const R = this.params.R;
        return {
            x: R * Math.sin(st.phi),
            y: R * (1 - Math.cos(st.phi))
        };
    }

    /** 힙 위치 (발 + 하체 방향으로 L₁) */
    getHipPos(st) {
        st = st || this.state;
        const foot = this.getFootPos(st);
        const L1 = this.params.L1;
        return {
            x: foot.x + L1 * Math.sin(st.alpha),
            y: foot.y + L1 * Math.cos(st.alpha)
        };
    }

    /** 하체 CoM 위치 (발 + ℓ₁ 방향) */
    getLowerCoM(st) {
        st = st || this.state;
        const foot = this.getFootPos(st);
        const ell1 = this.params.ell1;
        return {
            x: foot.x + ell1 * Math.sin(st.alpha),
            y: foot.y + ell1 * Math.cos(st.alpha)
        };
    }

    /** 상체 CoM 위치 (힙 + ℓ₂ 방향) */
    getUpperCoM(st) {
        st = st || this.state;
        const hip = this.getHipPos(st);
        const ell2 = this.params.ell2;
        return {
            x: hip.x + ell2 * Math.sin(st.theta),
            y: hip.y + ell2 * Math.cos(st.theta)
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

    /** CoM 수평 오차 = x_com − x_foot (균형 기준) */
    getCoMError(st) {
        st = st || this.state;
        const com = this.getTotalCoM(st);
        const foot = this.getFootPos(st);
        return com.x - foot.x;
    }

    /** 총 에너지 */
    getEnergy(st) {
        st = st || this.state;
        const p = this.params;
        const { m1, m2, R, g } = p;
        const ell1 = p.ell1, ell2 = p.ell2, L1 = p.L1;

        // 하체 CoM 속도
        const vx1 = R * Math.cos(st.phi) * st.phiDot + ell1 * Math.cos(st.alpha) * st.alphaDot;
        const vy1 = R * Math.sin(st.phi) * st.phiDot - ell1 * Math.sin(st.alpha) * st.alphaDot;
        // 상체 CoM 속도
        const vx2 = R * Math.cos(st.phi) * st.phiDot + L1 * Math.cos(st.alpha) * st.alphaDot + ell2 * Math.cos(st.theta) * st.thetaDot;
        const vy2 = R * Math.sin(st.phi) * st.phiDot - L1 * Math.sin(st.alpha) * st.alphaDot - ell2 * Math.sin(st.theta) * st.thetaDot;

        const T = 0.5 * m1 * (vx1 * vx1 + vy1 * vy1) + 0.5 * p.I1 * st.alphaDot * st.alphaDot
            + 0.5 * m2 * (vx2 * vx2 + vy2 * vy2) + 0.5 * p.I2 * st.thetaDot * st.thetaDot;

        const lc = this.getLowerCoM(st);
        const uc = this.getUpperCoM(st);
        const V = m1 * g * lc.y + m2 * g * uc.y;

        return { kinetic: T, potential: V, total: T + V };
    }

    /** 외부 교란 (상체에 각속도 충격) */
    applyPerturbation(impulse) {
        this.state.thetaDot += impulse;
    }
};

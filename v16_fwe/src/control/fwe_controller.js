/**
 * Slackline Balance Simulator V16 — FWE 4-Phase Controller (True Equation 4)
 * 
 * 제4 메인 식 원본 구현:
 *   k₁ (접기 비례 게인)을 설정하면, 매 사이클 트리거 시
 *   Newton-Raphson 2×2 연립 솔버가 최적의 (T_f*, T_w*)를 실시간 계산.
 * 
 * 상태머신: IDLE → FOLD(T_f*) → WAIT1(T_w*) → EXTEND(T_f*) → IDLE
 * 
 * 설계 변수: k₁, β_threshold, max_δ, Kp_servo, Kd_servo
 */
var SL = SL || {};

SL.FWEController = class {
    constructor() {
        // ===== 식 4 설계 파라미터 =====
        this.k1 = 100.0;                     // 접기 비례 게인 (δ_fold = k₁ · β₀)
        this.beta_threshold = 0.001 * Math.PI / 180;  // FOLD 트리거 임계값 [rad] (0.001°)
        this.max_delta = 45 * Math.PI / 180;  // 최대 접기 각도 [rad] (45°)

        // ===== 힙 PD 서보 게인 =====
        this.Kp_servo = 35000.0;
        this.Kd_servo = 1000.0;

        // ===== 상태머신 변수 =====
        this.phase = 'IDLE';        // IDLE, FOLD, WAIT1, EXTEND
        this.t_phase = 0.0;         // 현재 페이즈 경과 시간 [s]
        this.T_f_star = 0.10;       // 솔버 산출 접기 시간 [s]
        this.T_w_star = 0.20;       // 솔버 산출 대기 시간 [s]
        this.d_fold = 0.0;          // |k₁ · β₀| (접기 각도 크기)
        this.fold_sign = 1.0;       // 접기 방향 부호
        this.delta_target = 0.0;    // 현재 목표 힙각 (θ − α)
        this.delta_offset = 0.0;    // FOLD 시작 시점의 힙각 (접기 기준점)

        // ===== 디버깅 =====
        this.cycle_count = 0;
        this.phase_log = [];
        this.solver_converged = true;
        this.last_solver_iterations = 0;

        // ===== 인터페이스 호환 =====
        this.nominalParams = null;
        this.sensorNoise = null;
        this.estimationMode = 'ideal';
        this._appliedTau = 0;
    }

    setSensorNoise(noise) { this.sensorNoise = noise; }

    setEstimationMode(mode) {
        this.estimationMode = mode;
        this.reset();
    }

    reset() {
        this.phase = 'IDLE';
        this.t_phase = 0.0;
        this.T_f_star = 0.10;
        this.T_w_star = 0.20;
        this.d_fold = 0.0;
        this.fold_sign = 1.0;
        this.delta_target = 0.0;
        this.delta_offset = 0.0;
        this.cycle_count = 0;
        this.phase_log = [];
        this._appliedTau = 0;
        console.log('[FWE V16] Controller reset');
    }

    _log(msg) {
        const logMsg = `[FWE #${this.cycle_count}] ${msg}`;
        this.phase_log.push(logMsg);
        if (this.phase_log.length > 100) this.phase_log.shift();
        console.log(logMsg);
    }

    // ================================================================
    //  물리 상수 계산
    // ================================================================
    _getPhysicsConstants(params) {
        const m1 = params.m1, m2 = params.m2;
        const L1 = params.L1, L2 = params.L2;
        const R = params.R, g = params.g || 9.81;
        const ell1 = L1 / 2, ell2 = L2 / 2;
        const Mt = m1 + m2;

        // CoM 높이 (발목 기준)
        const h_com = (m1 * ell1 + m2 * (L1 + ell2)) / Mt;

        // 모멘트 암
        const p1 = m1 * ell1 + m2 * L1;
        const p2 = m2 * ell2;
        const ps = p1 + p2;
        const c_foot = p2 / ps;

        // 관성모멘트 (발목 기준)
        const I1 = m1 * L1 * L1 / 12;
        const I2 = m2 * L2 * L2 / 12;
        const J_aa = m1 * ell1 * ell1 + I1 + m2 * L1 * L1;
        const J_tt = m2 * ell2 * ell2 + I2;
        const J_at = m2 * L1 * ell2;
        const J_tot = J_aa + J_tt + 2 * J_at;

        // 고유 진동수/낙하율
        const lambda = Math.sqrt((Mt * g * h_com) / J_tot);
        const R_minus_h = R - h_com;
        const omega = R_minus_h > 0.01 ? Math.sqrt(g / R_minus_h) : 100.0;

        return { Mt, h_com, p1, p2, ps, c_foot, lambda, omega, R, g };
    }

    // ================================================================
    //  제4 메인 식 — G₁, G₂ 연립 함수
    // ================================================================
    _equation4(T_f, T_w, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h) {
        if (T_f <= 0.001 || T_w <= 0.001) return [1e6, 1e6];

        const T_total = 2 * T_f + T_w;

        // === 줄 진자 적분 커널 ===
        const cosHalf = Math.cos(omega * T_f / 2);
        const sinHalf = Math.sin(omega * T_f / 2);
        const Phi_pos = (2 * f_bar_phi / (omega * omega * T_f * T_f)) * cosHalf * (1 - cosHalf);
        const Phi_vel = (2 * f_bar_phi / (omega * T_f * T_f)) * sinHalf * (cosHalf - 1);

        // === 몸 역진자 적분 커널 ===
        const coshF = Math.cosh(lambda * T_f);
        const sinhF = Math.sinh(lambda * T_f);
        const coshHalfF = Math.cosh(lambda * T_f / 2);
        const sinhHalfF = Math.sinh(lambda * T_f / 2);
        const B_pos = -(f_bar_beta / (lambda * lambda * T_f * T_f)) * (coshF - 2 * coshHalfF + 1);
        const B_vel = -(f_bar_beta / (lambda * T_f * T_f)) * (sinhF - 2 * sinhHalfF);

        // === d̄_φ: 외력 전파 (FOLD → WAIT → EXTEND 누적) ===
        const cosW = Math.cos(omega * T_w);
        const sinW = Math.sin(omega * T_w);
        const cosF = Math.cos(omega * T_f);
        const sinF = Math.sin(omega * T_f);

        const d_bar_phi = (Phi_pos * cosW + (Phi_vel / omega) * sinW) * cosF
                        + (-Phi_pos * omega * sinW + Phi_vel * cosW) * (sinF / omega)
                        - Phi_pos;

        // === d̄_β: 외력 전파 (FOLD → WAIT → EXTEND 누적) ===
        const coshW = Math.cosh(lambda * T_w);
        const sinhW = Math.sinh(lambda * T_w);
        const coshFe = Math.cosh(lambda * T_f);
        const sinhFe = Math.sinh(lambda * T_f);

        const d_bar_beta = (B_pos * coshW + (B_vel / lambda) * sinhW) * coshFe
                         + (B_pos * lambda * sinhW + B_vel * coshW) * (sinhFe / lambda)
                         - B_pos;

        // === 최종 연립 구속 식 ===
        const G1 = psi * Math.cos(omega * T_total) + v_phi * Math.sin(omega * T_total) + d_bar_phi * (R / h);
        const G2 = Math.cosh(lambda * T_total) + v_beta * Math.sinh(lambda * T_total) + d_bar_beta;

        return [G1, G2];
    }

    // ================================================================
    //  Newton-Raphson 2×2 연립 솔버
    // ================================================================
    _solveEquation4(phi_i, phi_dot_i, beta_0, beta_dot_i, phys) {
        const { omega, lambda, R, h_com, c_foot } = phys;
        const h = h_com;
        const k1 = this.k1;

        // 1. 무차원화
        if (Math.abs(beta_0) < 1e-8) return { T_f: 0.10, T_w: 0.20, converged: false };

        const psi = (R * phi_i) / (h * beta_0);
        const v_phi = (R * phi_dot_i) / (h * beta_0 * omega);
        const v_beta = beta_dot_i / (beta_0 * lambda);

        const f_bar_phi = (4 * c_foot * k1) / R;
        const f_bar_beta = (4 * c_foot * k1) / h;

        console.log(`[SOLVER] inputs: phi=${(phi_i*180/Math.PI).toFixed(4)}° phi_dot=${phi_dot_i.toFixed(4)} beta=${(beta_0*180/Math.PI).toFixed(4)}° beta_dot=${beta_dot_i.toFixed(6)}`);
        console.log(`[SOLVER] normalized: psi=${psi.toFixed(4)} v_phi=${v_phi.toFixed(4)} v_beta=${v_beta.toFixed(4)}`);
        console.log(`[SOLVER] forcing: f_bar_phi=${f_bar_phi.toFixed(4)} f_bar_beta=${f_bar_beta.toFixed(4)} omega=${omega.toFixed(4)} lambda=${lambda.toFixed(4)}`);

        // 2. Grid Search → 최적 초기 추정 탐색
        const N_TF = 30, N_TW = 30;
        const TF_MIN = 0.02, TF_MAX = 0.30;
        const TW_MIN = 0.02, TW_MAX = 0.80;
        let bestCost = Infinity, bestTf = 0.05, bestTw = 0.20;

        for (let i = 0; i < N_TF; i++) {
            const tf = TF_MIN + (TF_MAX - TF_MIN) * i / (N_TF - 1);
            for (let j = 0; j < N_TW; j++) {
                const tw = TW_MIN + (TW_MAX - TW_MIN) * j / (N_TW - 1);
                const F = this._equation4(tf, tw, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h);
                const cost = F[0] * F[0] + F[1] * F[1];
                if (cost < bestCost) {
                    bestCost = cost;
                    bestTf = tf;
                    bestTw = tw;
                }
            }
        }

        const F_grid = this._equation4(bestTf, bestTw, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h);
        console.log(`[SOLVER] grid best: T_f=${(bestTf*1000).toFixed(1)}ms T_w=${(bestTw*1000).toFixed(1)}ms G1=${F_grid[0].toExponential(3)} G2=${F_grid[1].toExponential(3)} cost=${bestCost.toExponential(3)}`);

        // 3. 감쇠 Newton-Raphson 미세 조정
        let T_f = bestTf;
        let T_w = bestTw;
        const eps_fd = 1e-6;
        const tol = 1e-8;
        const maxIter = 80;
        let converged = false;
        let iter = 0;

        for (iter = 0; iter < maxIter; iter++) {
            const F = this._equation4(T_f, T_w, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h);

            if (iter < 5 || iter % 20 === 0) {
                console.log(`[SOLVER] iter=${iter}: T_f=${(T_f*1000).toFixed(1)}ms T_w=${(T_w*1000).toFixed(1)}ms G1=${F[0].toExponential(4)} G2=${F[1].toExponential(4)}`);
            }

            if (Math.abs(F[0]) < tol && Math.abs(F[1]) < tol) {
                converged = true;
                console.log(`[SOLVER] CONVERGED at iter=${iter}: T_f=${(T_f*1000).toFixed(1)}ms T_w=${(T_w*1000).toFixed(1)}ms`);
                break;
            }

            // 야코비안 (유한 차분)
            const F_dTf = this._equation4(T_f + eps_fd, T_w, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h);
            const F_dTw = this._equation4(T_f, T_w + eps_fd, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h);

            const J00 = (F_dTf[0] - F[0]) / eps_fd;
            const J01 = (F_dTw[0] - F[0]) / eps_fd;
            const J10 = (F_dTf[1] - F[1]) / eps_fd;
            const J11 = (F_dTw[1] - F[1]) / eps_fd;

            const det = J00 * J11 - J01 * J10;
            if (Math.abs(det) < 1e-15) {
                console.log(`[SOLVER] det≈0 at iter=${iter}, aborting`);
                break;
            }

            let dTf = -(J11 * F[0] - J01 * F[1]) / det;
            let dTw = -(-J10 * F[0] + J00 * F[1]) / det;

            // 감쇠: 스텝 크기 제한 (최대 현재값의 50%)
            const maxStep = 0.5;
            const stepScale = Math.max(Math.abs(dTf) / (T_f * maxStep + 0.001),
                                       Math.abs(dTw) / (T_w * maxStep + 0.001), 1.0);
            dTf /= stepScale;
            dTw /= stepScale;

            // 라인 서치: 스텝이 cost를 줄이는지 확인
            const cost0 = F[0] * F[0] + F[1] * F[1];
            let alpha = 1.0;
            for (let ls = 0; ls < 8; ls++) {
                const tf_try = Math.max(TF_MIN, Math.min(TF_MAX, T_f + alpha * dTf));
                const tw_try = Math.max(TW_MIN, Math.min(TW_MAX, T_w + alpha * dTw));
                const F_try = this._equation4(tf_try, tw_try, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h);
                const cost_try = F_try[0] * F_try[0] + F_try[1] * F_try[1];
                if (cost_try < cost0 * 0.999) {
                    T_f = tf_try;
                    T_w = tw_try;
                    break;
                }
                alpha *= 0.5;
                if (ls === 7) {
                    // 라인 서치 실패 — 그냥 작은 스텝 적용
                    T_f = Math.max(TF_MIN, Math.min(TF_MAX, T_f + 0.1 * dTf));
                    T_w = Math.max(TW_MIN, Math.min(TW_MAX, T_w + 0.1 * dTw));
                }
            }
        }

        // Newton 결과 평가
        if (converged) {
            console.log(`[SOLVER] CONVERGED at iter=${iter}: T_f=${(T_f*1000).toFixed(1)}ms T_w=${(T_w*1000).toFixed(1)}ms`);
            return { T_f, T_w, converged: true };
        }

        // 수렴 실패 — 실증적 폴백 사용
        const F_final = this._equation4(T_f, T_w, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h);
        console.log(`[SOLVER] FAILED after ${iter} iter: Newton T_f=${(T_f*1000).toFixed(1)}ms T_w=${(T_w*1000).toFixed(1)}ms G1=${F_final[0].toExponential(3)} G2=${F_final[1].toExponential(3)}`);
        console.log(`[SOLVER] using empirical fallback: T_f=100ms T_w=200ms`);
        return { T_f: 0.10, T_w: 0.20, converged: false };
    }

    // ================================================================
    //  삼각형 가·감속 프로파일 (δ(t): 0 → d_fold)
    // ================================================================
    _foldProfile(t_local, T_f, d) {
        if (T_f <= 0) return d;
        const a = 4 * d / (T_f * T_f);
        const half = T_f / 2;

        if (t_local <= 0) return 0.0;
        else if (t_local < half) return 0.5 * a * t_local * t_local;
        else if (t_local < T_f) {
            const dt = t_local - half;
            return 0.5 * a * half * half + a * half * dt - 0.5 * a * dt * dt;
        }
        else return d;
    }

    // ================================================================
    //  센서/추정 처리
    // ================================================================
    _getNoisyIMU(state) {
        const imu = {
            alpha: state.alpha, theta: state.theta,
            alphaDot: state.alphaDot, thetaDot: state.thetaDot
        };
        if (this.sensorNoise && (this.sensorNoise.angleNoiseSigma > 0 || this.sensorNoise.gyroNoiseSigma > 0 || this.sensorNoise.gyroBiasDrift > 0)) {
            const noisy = this.sensorNoise.addNoise(state);
            imu.alpha = noisy.alpha;
            imu.theta = noisy.theta;
            imu.alphaDot = noisy.alphaDot;
            imu.thetaDot = noisy.thetaDot;
        }
        return imu;
    }

    _getKinematicEstimate(state, model) {
        const p = model.params;
        const R = p.R, L1 = p.L1, L2 = p.L2, m1 = p.m1, m2 = p.m2;
        const e1 = L1 / 2, e2 = L2 / 2, Mt = m1 + m2;
        const C1 = (m1 * e1 + m2 * L1) / Mt;
        const C2 = (m2 * e2) / Mt;
        return {
            phi: -(C1 * state.alpha + C2 * state.theta) / R,
            alpha: state.alpha, theta: state.theta,
            phiDot: -(C1 * state.alphaDot + C2 * state.thetaDot) / R,
            alphaDot: state.alphaDot, thetaDot: state.thetaDot
        };
    }

    // ================================================================
    //  FOLD 트리거 및 사이클 초기화
    // ================================================================
    _initiateFold(beta_0, beta_dot, phi_i, phi_dot_i, phys, delta_current) {
        // 식 4 솔버 호출
        const sol = this._solveEquation4(phi_i, phi_dot_i, beta_0, beta_dot, phys);
        this.T_f_star = sol.T_f;
        this.T_w_star = sol.T_w;
        this.solver_converged = sol.converged;

        // 접기 각도: δ_fold = k₁ · β₀
        const d_fold_raw = this.k1 * beta_0;
        this.fold_sign = Math.sign(d_fold_raw);
        if (this.fold_sign === 0) this.fold_sign = 1.0;
        this.d_fold = Math.min(Math.abs(d_fold_raw), this.max_delta);

        // 현재 힙각 기록 (접기 기준점)
        this.delta_offset = delta_current;

        // 상태 전환
        this.phase = 'FOLD';
        this.t_phase = 0.0;
        this.cycle_count++;
        this.delta_target = this.delta_offset + this.fold_sign * this._foldProfile(0.0, this.T_f_star, this.d_fold);

        this._log(`FOLD start: β₀=${(beta_0 * 180 / Math.PI).toFixed(2)}° d_fold=${(d_fold_raw * 180 / Math.PI).toFixed(1)}° δ_offset=${(delta_current * 180 / Math.PI).toFixed(1)}° T_f*=${(this.T_f_star * 1000).toFixed(0)}ms T_w*=${(this.T_w_star * 1000).toFixed(0)}ms converged=${sol.converged} iter=${this.last_solver_iterations}`);
    }

    // ================================================================
    //  메인 제어 루프 (매 dt마다 호출)
    // ================================================================
    compute(state, model, dt) {
        // 1. 센서 처리
        const imuData = this._getNoisyIMU(state);
        let estState;
        if (this.estimationMode === 'kinematic') {
            const noisyState = Object.assign({}, state, imuData);
            estState = this._getKinematicEstimate(noisyState, model);
        } else {
            estState = Object.assign({}, state, imuData);
        }

        // 2. 물리 상수 계산
        const p = this.nominalParams || model.params;
        const phys = this._getPhysicsConstants(p);

        // 3. β, β̇ 계산 (절대 CoM 기울기)
        const x_com = phys.R * Math.sin(estState.phi)
                    + (phys.p1 * Math.sin(estState.alpha) + phys.p2 * Math.sin(estState.theta)) / phys.Mt;
        const beta = x_com / phys.h_com;

        const x_com_dot = phys.R * Math.cos(estState.phi) * estState.phiDot
                        + (phys.p1 * Math.cos(estState.alpha) * estState.alphaDot
                         + phys.p2 * Math.cos(estState.theta) * estState.thetaDot) / phys.Mt;
        const beta_dot = x_com_dot / phys.h_com;

        // 4. φ, φ̇ (발 각도)
        const phi_i = estState.phi;
        const phi_dot_i = estState.phiDot;

        // 5. 현재 힙각
        const delta_actual = estState.theta - estState.alpha;
        const delta_dot_actual = estState.thetaDot - estState.alphaDot;

        // 6. 상태머신
        this.t_phase += dt;

        if (this.phase === 'IDLE') {
            // IDLE: 현재 힙각을 PD 서보로 유지
            this.delta_target = delta_actual;

            // 트리거: |β| > threshold 이고 넘어지는 방향(β와 β̇ 같은 부호)
            const is_falling = (Math.sign(beta) === Math.sign(beta_dot)) || (Math.abs(beta_dot) < 1e-5);
            if (Math.abs(beta) > this.beta_threshold && is_falling) {
                this._initiateFold(beta, beta_dot, phi_i, phi_dot_i, phys, delta_actual);
            }
        }
        else if (this.phase === 'FOLD') {
            const pos = this._foldProfile(this.t_phase, this.T_f_star, this.d_fold);
            this.delta_target = this.delta_offset + this.fold_sign * pos;

            if (this.t_phase >= this.T_f_star) {
                this.delta_target = this.delta_offset + this.fold_sign * this.d_fold;
                this.phase = 'WAIT1';
                this.t_phase = 0.0;
                this._log(`WAIT1 start: holding δ_target=${(this.delta_target * 180 / Math.PI).toFixed(1)}° for T_w*=${(this.T_w_star * 1000).toFixed(0)}ms`);
            }
        }
        else if (this.phase === 'WAIT1') {
            this.delta_target = this.delta_offset + this.fold_sign * this.d_fold;

            if (this.t_phase >= this.T_w_star) {
                this.phase = 'EXTEND';
                this.t_phase = 0.0;
                this._log(`EXTEND start: returning from δ_target=${(this.delta_target * 180 / Math.PI).toFixed(1)}°`);
            }
        }
        else if (this.phase === 'EXTEND') {
            const pos = this._foldProfile(this.t_phase, this.T_f_star, this.d_fold);
            this.delta_target = this.delta_offset + this.fold_sign * (this.d_fold - pos);

            if (this.t_phase >= this.T_f_star) {
                this.delta_target = this.delta_offset;  // 접기 전 힙각으로 복귀
                this.phase = 'IDLE';
                this.t_phase = 0.0;
                this._log(`IDLE: cycle complete. β=${(beta * 180 / Math.PI).toFixed(2)}° φ=${(phi_i * 180 / Math.PI).toFixed(2)}°`);
            }
        }

        // 7. 힙 PD 서보 토크 (FOLD/WAIT1/EXTEND에서만)
        let tau = this.Kp_servo * (this.delta_target - delta_actual) - this.Kd_servo * delta_dot_actual;
        tau = Math.max(-p.tauMax, Math.min(p.tauMax, tau));

        this._appliedTau = tau;
        return tau;
    }

    getDelayInfo() {
        const modeLabels = { 'ideal': '🎯 완벽한 센싱', 'kinematic': '📐 CoM 역기구학' };
        return {
            totalMs: 0,
            sensorMs: modeLabels[this.estimationMode] || '알 수 없음',
            actuatorMs: 0,
            estimationMode: this.estimationMode,
            phaseName: this.phase
        };
    }
};

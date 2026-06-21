/**
 * Slackline Balance Simulator V16 — FWE 4-Phase Controller (True Equation 4)
 * * [2026 최종 정렬본] 3-DOF 정밀 질량행렬 연동형 FWE 제어기
 * - model.js의 라그랑주 운동방정식 질량 행렬 스펙을 직접 상속받아 d_fold 계산
 * - UI의 k1 슬라이더를 오리지널 입실론 게인으로 해석 (epsilon = k1 / 100)
 * - 접기 시간 T_f는 0.05초(50ms) 고정 레이어로 통제하여 1차원 수렴 보장
 * - 단일 마스터 방정식 (G1 + G2 = 0)을 1차원 이분법 솔버로 칼수렴 구현
 */
var SL = SL || {};

SL.FWEController = class {
    constructor() {
        // ===== 식 4 설계 파라미터 =====
        this.k1 = 200.0;                     // UI 슬라이더 연동 (epsilon = k1 / 100)
        this.beta_threshold = 0.001 * Math.PI / 180;  // FOLD 트리거 임계값 [rad]
        this.max_delta = 45 * Math.PI / 180;  // 최대 접기 각도 제한 [rad]
        
        // 현실적인 모터의 삼각형 가·감속 구동 시간 고정 레이어 (50ms)
        this.T_f_fixed = 0.05; 

        // ===== 힙 PD 서보 게인 (이상적 추종) =====
        this.Kp_servo = 500000.0;
        this.Kd_servo = 500.0;

        // ===== 상태머신 변수 =====
        this.phase = 'IDLE';        // IDLE, FOLD, WAIT1, EXTEND
        this.t_phase = 0.0;         // 현재 페이즈 경과 시간
        this.T_f_star = 0.05;       // 고정 접기 시간
        this.T_w_star = 0.20;       // 솔버가 산출할 최적 대기 시간
        this.d_fold = 0.0;          // 해석적 유도로 결정되는 접기 각도 크기
        this.fold_sign = 1.0;       // 접기 방향 부호
        this.delta_target = 0.0;    // 목표 힙각 (θ − α)
        this.delta_offset = 0.0;    // FOLD 시작 시점의 기준 힙각

        // ===== 디버깅 변수 =====
        this.cycle_count = 0;
        this.phase_log = [];
        this.solver_converged = true;
        this.last_solver_iterations = 0;

        // ===== 인터페이스 호환 변수 =====
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
        this.T_f_star = this.T_f_fixed;
        this.T_w_star = 0.20;
        this.d_fold = 0.0;
        this.fold_sign = 1.0;
        this.delta_target = 0.0;
        this.delta_offset = 0.0;
        this.cycle_count = 0;
        this.phase_log = [];
        this._appliedTau = 0;
        console.log('[FWE 최종 개정본] 3-DOF 연동형 제어기 리셋 완료');
    }

    _log(msg) {
        const logMsg = `[FWE #${this.cycle_count}] ${msg}`;
        this.phase_log.push(logMsg);
        if (this.phase_log.length > 100) this.phase_log.shift();
        console.log(logMsg);
    }

    // ================================================================
    //  물리 상수 및 동역학 고유치 계산
    // ================================================================
    _getPhysicsConstants(params) {
        const m1 = params.m1, m2 = params.m2;
        const L1 = params.L1, L2 = params.L2;
        const R = params.R, g = params.g || 9.81;
        const ell1 = L1 / 2, ell2 = L2 / 2;
        const Mt = m1 + m2;

        const h_com = (m1 * ell1 + m2 * (L1 + ell2)) / Mt;
        const p1 = params.p1, p2 = params.p2, p3 = params.p3;
        const c_foot = p2 / (p1 + p2);

        // model.js 기반의 총 관성 모멘트 역산
        const J_tot = params.M22 + params.M33 + 2 * p3; 
        const lambda = Math.sqrt((Mt * g * h_com) / J_tot);
        const R_minus_h = R - h_com;
        const omega = R_minus_h > 0.01 ? Math.sqrt(g / R_minus_h) : 100.0;

        return { Mt, h_com, p1, p2, p3, c_foot, lambda, omega, R, g };
    }

    // ================================================================
    //  제4 메인 식 코어 커널 (G1 + G2)
    // ================================================================
    _equation4(T_f, T_w, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h) {
        if (T_f <= 0.001 || T_w <= 0.001) return [1e6, 1e6];
        const T_total = 2 * T_f + T_w;

        const cosHalf = Math.cos(omega * T_f / 2);
        const sinHalf = Math.sin(omega * T_f / 2);
        const Phi_pos = (2 * f_bar_phi / (omega * omega * T_f * T_f)) * cosHalf * (1 - cosHalf);
        const Phi_vel = (2 * f_bar_phi / (omega * T_f * T_f)) * sinHalf * (cosHalf - 1);

        const coshF = Math.cosh(lambda * T_f);
        const sinhF = Math.sinh(lambda * T_f);
        const coshHalfF = Math.cosh(lambda * T_f / 2);
        const sinhHalfF = Math.sinh(lambda * T_f / 2);
        const B_pos = -(f_bar_beta / (lambda * lambda * T_f * T_f)) * (coshF - 2 * coshHalfF + 1);
        const B_vel = -(f_bar_beta / (lambda * T_f * T_f)) * (sinhF - 2 * sinhHalfF);

        const cosW = Math.cos(omega * T_w);
        const sinW = Math.sin(omega * T_w);
        const cosF = Math.cos(omega * T_f);
        const sinF = Math.sin(omega * T_f);
        const d_bar_phi = (Phi_pos * cosW + (Phi_vel / omega) * sinW) * cosF
                        + (-Phi_pos * omega * sinW + Phi_vel * cosW) * (sinF / omega)
                        - Phi_pos;

        const coshW = Math.cosh(lambda * T_w);
        const sinhW = Math.sinh(lambda * T_w);
        const coshFe = Math.cosh(lambda * T_f);
        const sinhFe = Math.sinh(lambda * T_f);
        const d_bar_beta = (B_pos * coshW + (B_vel / lambda) * sinhW) * coshFe
                         + (B_pos * lambda * sinhW + B_vel * coshW) * (sinhFe / lambda)
                         - B_pos;

        const G1 = psi * Math.cos(omega * T_total) + v_phi * Math.sin(omega * T_total) + d_bar_phi * (R / h);
        const G2 = Math.cosh(lambda * T_total) + v_beta * Math.sinh(lambda * T_total) + d_bar_beta;

        return [G1, G2];
    }

    // ================================================================
    //  [수정 핵심] model.js 의 질량 행렬을 직접 투입한 1D 솔버 단락
    // ================================================================
    _solveEquation4(phi_i, phi_dot_i, beta_0, beta_dot_i, phys, p) {
        const { omega, lambda, R, h_com, c_foot } = phys;
        const h = h_com;
        const T_f = this.T_f_fixed;

        if (Math.abs(beta_0) < 1e-8) {
            return { T_f, T_w: 0.20, d_fold: 0.0, fold_sign: 1.0, converged: false };
        }

        // 1. 입실론 게인 역산 (k1 슬라이더 기반)
        const epsilon = this.k1 / 100.0;

        // 2. model.js의 고유 3-DOF 질량 행렬 요소를 기하 평면(q=0) 상에서 추출
        const M11 = p.M11;
        const M12 = R * p.p1;
        const M13 = R * p.p2;
        const M22 = p.M22;
        const M23 = p.p3;
        const M33 = p.M33;

        // 라그랑주 역학에 따른 상하체 토크 작용-반작용 내부 감도행렬 디코딩
        const C_alpha = M11 * (M22 + M23) - M12 * (M12 + M13);
        const C_theta = M11 * (M23 + M33) - M13 * (M12 + M13);

        // 허리 관절각 변화(d_delta) 대비 발끝의 수평 이동 민감도(d_xfoot)의 정밀 유도식
        const DX_foot_per_d_delta = R * (M12 * C_theta - M13 * C_alpha) / (M11 * (C_alpha + C_theta));

        // [방향 오류 완벽 해결] 쓰러지는 방향(beta_0)과 무조건 같은 방향으로 스케일링 일치
        let fold_sign = Math.sign(beta_0);
        if (fold_sign === 0) fold_sign = 1.0;

        // 목표 발끝 도약 거리(epsilon * h * beta_0)를 만족하기 위한 정밀 d_fold 도출
        let d_fold = (epsilon * h * Math.abs(beta_0)) / Math.abs(DX_foot_per_d_delta);

        console.log(`[SOLVER] ε=${epsilon.toFixed(2)} h=${h.toFixed(3)}m DX/dδ=${DX_foot_per_d_delta.toFixed(4)}m/rad β₀=${(beta_0*180/Math.PI).toFixed(3)}° → d_fold_raw=${(d_fold*180/Math.PI).toFixed(2)}°`);

        // 기계적 상시 포화 한계 보호
        if (d_fold > this.max_delta) {
            d_fold = this.max_delta;
        }

        // 실제 꺾이는 크기에 부합하도록 물리 외력 엔진 부하 인자(k1_effective) 동기화
        const k1_effective = d_fold / Math.abs(beta_0);

        // 3. 무차원 변수 징집
        const psi = (R * phi_i) / (h * beta_0);
        const v_phi = (R * phi_dot_i) / (h * beta_0 * omega);
        const v_beta = beta_dot_i / (beta_0 * lambda);

        const f_bar_phi = (4 * c_foot * k1_effective) / R;
        const f_bar_beta = (4 * c_foot * k1_effective) / h;

        // 4. 단일 마스터 식 타깃 선언
        const masterEquation1D = (tw) => {
            const F = this._equation4(T_f, tw, psi, v_phi, v_beta, f_bar_phi, f_bar_beta, omega, lambda, R, h);
            return F[0] + F[1];
        };

        // 5. 구간 이분법 Bracket 스캔
        const N_SCAN = 80;
        const TW_MIN = 0.01, TW_MAX = 1.4;
        let bracketL = -1, bracketR = -1;
        let prevVal = masterEquation1D(TW_MIN);

        for (let i = 1; i < N_SCAN; i++) {
            const tw = TW_MIN + (TW_MAX - TW_MIN) * i / (N_SCAN - 1);
            const val = masterEquation1D(tw);
            if (prevVal * val <= 0) {
                bracketL = tw - (TW_MAX - TW_MIN) / (N_SCAN - 1);
                bracketR = tw;
                break;
            }
            prevVal = val;
        }

        let T_w = 0.20;
        let converged = false;
        this.last_solver_iterations = 0;

        // 6. 이분법 루프 작동
        if (bracketL !== -1 && bracketR !== -1) {
            let left = bracketL;
            let right = bracketR;
            for (let iter = 0; iter < 40; iter++) {
                this.last_solver_iterations++;
                T_w = (left + right) / 2;
                const val = masterEquation1D(T_w);
                
                if (Math.abs(val) < 1e-7) {
                    break;
                }
                if (masterEquation1D(left) * val <= 0) {
                    right = T_w;
                } else {
                    left = T_w;
                }
            }
            converged = true;
        } else {
            let bestCost = Infinity;
            for (let i = 0; i < N_SCAN; i++) {
                const tw = TW_MIN + (TW_MAX - TW_MIN) * i / (N_SCAN - 1);
                const val = Math.abs(masterEquation1D(tw));
                if (val < bestCost) {
                    bestCost = val;
                    T_w = tw;
                }
            }
            converged = false;
        }

        return { T_f, T_w, d_fold, fold_sign, converged };
    }

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

    _initiateFold(beta_0, beta_dot, phi_i, phi_dot_i, phys, delta_current, p) {
        const sol = this._solveEquation4(phi_i, phi_dot_i, beta_0, beta_dot, phys, p);
        
        this.T_f_star = sol.T_f;
        this.T_w_star = sol.T_w;
        this.d_fold = sol.d_fold;
        this.fold_sign = sol.fold_sign;
        this.solver_converged = sol.converged;

        this.delta_offset = delta_current;
        this.phase = 'FOLD';
        this.t_phase = 0.0;
        this.cycle_count++;
        
        this.delta_target = this.delta_offset + this.fold_sign * this._foldProfile(0.0, this.T_f_star, this.d_fold);

        this._log(`[트리거 성공] β₀=${(beta_0 * 180 / Math.PI).toFixed(2)}° d_fold=${(this.fold_sign * this.d_fold * 180 / Math.PI).toFixed(1)}° T_w*=${(this.T_w_star * 1000).toFixed(0)}ms`);
    }

    compute(state, model, dt) {
        const imuData = this._getNoisyIMU(state);
        let estState;
        if (this.estimationMode === 'kinematic') {
            const noisyState = Object.assign({}, state, imuData);
            estState = this._getKinematicEstimate(noisyState, model);
        } else {
            estState = Object.assign({}, state, imuData);
        }

        const p = this.nominalParams || model.params;
        const phys = this._getPhysicsConstants(p);

        // β: 발 기준 상대 CoM 기울기 (힙토크는 내력 → phi 항 제외)
        const x_com_rel = (phys.p1 * Math.sin(estState.alpha) + phys.p2 * Math.sin(estState.theta)) / phys.Mt;
        const beta = x_com_rel / phys.h_com;

        const x_com_rel_dot = (phys.p1 * Math.cos(estState.alpha) * estState.alphaDot
                             + phys.p2 * Math.cos(estState.theta) * estState.thetaDot) / phys.Mt;
        const beta_dot = x_com_rel_dot / phys.h_com;

        const phi_i = estState.phi;
        const phi_dot_i = estState.phiDot;

        const delta_actual = estState.theta - estState.alpha;
        const delta_dot_actual = estState.thetaDot - estState.alphaDot;

        this.t_phase += dt;

        if (this.phase === 'IDLE') {
            this.delta_target = delta_actual;

            const is_falling = (Math.sign(beta) === Math.sign(beta_dot)) || (Math.abs(beta_dot) < 1e-5);
            if (Math.abs(beta) > this.beta_threshold && is_falling) {
                this._initiateFold(beta, beta_dot, phi_i, phi_dot_i, phys, delta_actual, p);
            }
        }
        else if (this.phase === 'FOLD') {
            const pos = this._foldProfile(this.t_phase, this.T_f_star, this.d_fold);
            this.delta_target = this.delta_offset + this.fold_sign * pos;

            if (this.t_phase >= this.T_f_star) {
                this.delta_target = this.delta_offset + this.fold_sign * this.d_fold;
                this.phase = 'WAIT1';
                this.t_phase = 0.0;
            }
        }
        else if (this.phase === 'WAIT1') {
            this.delta_target = this.delta_offset + this.fold_sign * this.d_fold;

            if (this.t_phase >= this.T_w_star) {
                this.phase = 'EXTEND';
                this.t_phase = 0.0;
            }
        }
        else if (this.phase === 'EXTEND') {
            const pos = this._foldProfile(this.t_phase, this.T_f_star, this.d_fold);
            this.delta_target = this.delta_offset + this.fold_sign * (this.d_fold - pos);

            if (this.t_phase >= this.T_f_star) {
                this.delta_target = this.delta_offset;
                this.phase = 'IDLE';
                this.t_phase = 0.0;
            }
        }

        let tau = this.Kp_servo * (this.delta_target - delta_actual) - this.Kd_servo * delta_dot_actual;

        // 종합 진단 (첫 2사이클, 5ms 간격)
        if (this.cycle_count <= 2 && this.phase !== 'IDLE') {
            const step_ms = this.t_phase * 1000;
            if (step_ms < 1.5 || Math.floor(step_ms) % 5 === 0) {
                const err_deg = (this.delta_target - delta_actual) * 180 / Math.PI;
                const sat = Math.abs(tau) > p.tauMax ? ' ⚠️SAT' : '';
                const toDeg = r => (r * 180 / Math.PI).toFixed(2);
                console.log(`[DIAG] ${this.phase} t=${step_ms.toFixed(0)}ms | δ_tgt=${toDeg(this.delta_target)}° δ_act=${toDeg(delta_actual)}° err=${err_deg.toFixed(3)}° | φ=${toDeg(estState.phi)}° α=${toDeg(estState.alpha)}° θ=${toDeg(estState.theta)}° | β=${toDeg(beta)}° | τ=${tau.toFixed(0)}/${p.tauMax}${sat}`);
            }
        }

        tau = Math.max(-p.tauMax, Math.min(p.tauMax, tau));

        this._appliedTau = tau;
        return tau;
    }

    getDelayInfo() {
        return {
            totalMs: 0,
            sensorMs: this.estimationMode === 'kinematic' ? '📐 CoM 역기구학' : '🎯 완벽한 센싱',
            actuatorMs: 0,
            estimationMode: this.estimationMode,
            phaseName: this.phase
        };
    }
};

/**
 * Slackline Balance Simulator V15 — FWE 4-Phase Controller + Hip PD Servo
 * 
 * Fold-Wait₁-Extend-Wait₂ state machine for slackline balancing.
 * Outputs hip torque using PD servo tracking of delta_target.
 */
var SL = SL || {};

SL.FWEController = class {
    constructor() {
        // FWE 4단계 제어 파라미터 기본값 (V15 대형 로봇 스윕 결과 및 사용자 피드백 반영)
        this.eps = 0.5;             // 접기 강도 스케일 (디폴트 축소)
        this.T_f = 0.150;           // FOLD/EXTEND 소요 시간 [s]
        this.T_w1 = 0.080;          // WAIT₁ 대기 시간 [s] (많이 기다리는 문제 방지하기 위해 80ms로 단축)
        this.beta_threshold = 1.0 * Math.PI / 180; // FOLD 트리거 임계값 [rad] (1.0도)
        this.c_pred = 1.0;          // 적응적 예측 계수 [s/rad] (T_pred = c_pred × |beta_eff|)
        this.T_w2_min = 0.100;      // WAIT₂ 최소 잠금 시간 [s]
        this.max_delta = 30 * Math.PI / 180; // 힙각 최대 제한 [rad] (비선형 구역 회피 및 오버슈트 방지를 위해 30도로 조정)
        this.c_beta_dot = 0.3;      // beta_dot 감쇠 계수 (두 번째 사이클에서 과접기 방지)

        // 힙 관절 PD 서보 게인 (높은 질량을 강하게 추종하기 위해 고게인 사용)
        this.Kp_servo = 35000.0;
        this.Kd_servo = 1000.0;

        // 제어 상태 변수
        this.phase = 'IDLE';        // IDLE, FOLD, WAIT1, EXTEND, WAIT2
        this.t_phase = 0.0;         // 현재 페이즈 진행 시간 [s]
        this.d_fold = 0.0;          // 이번 사이클의 목표 접기 각도 [rad]
        this.fold_sign = 1.0;       // 접는 방향 (+1: 오른쪽, -1: 왼쪽)
        this.delta_target = 0.0;    // 목표 상대 힙각 (theta - alpha)
        this.beta_prev = null;      // 수치 미분용 이전 beta

        // 영점 감지용
        this.phi_prev = null;
        this.phi_crossed = false;

        // 디버깅 및 통계
        this.cycle_count = 0;
        this.phase_log = [];

        // 인터페이스 호환성용
        this.sensorNoise = null;
        this.estimationMode = 'ideal'; // 'ideal' | 'kinematic'
        this._appliedTau = 0;

        // V15 추가 제어 옵션 기본값
        this.skipWait2OnExtendZero = true;
        this.skipWait2Entirely = true;
    }

    setSensorNoise(noise) {
        this.sensorNoise = noise;
    }

    setEstimationMode(mode) {
        this.estimationMode = mode;
        this.reset();
    }

    reset() {
        this.phase = 'IDLE';
        this.t_phase = 0.0;
        this.d_fold = 0.0;
        this.fold_sign = 1.0;
        this.delta_target = 0.0;
        this.beta_prev = null;
        this.phi_prev = null;
        this.phi_crossed = false;
        this.cycle_count = 0;
        this.phase_log = [];
        this._appliedTau = 0;
        console.log('FWE Controller Reset completed.');
    }

    _log(msg) {
        const timestamp = this.t_phase.toFixed(3);
        const logMsg = `[FWE #${this.cycle_count}] [t_phase=${timestamp}s] ${msg}`;
        this.phase_log.push(logMsg);
        if (this.phase_log.length > 100) this.phase_log.shift(); // 로그 갯수 제한
        console.log(logMsg);
    }

    /** 접기 프로파일 (삼각 가속도 프로파일) */
    _fold_profile(t_local) {
        const Tf = this.T_f;
        const d = Math.abs(this.d_fold);
        if (Tf <= 0) return d;
        
        const a = 4 * d / (Tf * Tf);
        const half = Tf / 2;

        if (t_local <= 0) {
            return 0.0;
        } else if (t_local < half) {
            return 0.5 * a * t_local * t_local;
        } else if (t_local < Tf) {
            const dt = t_local - half;
            return 0.5 * a * half * half + a * half * dt - 0.5 * a * dt * dt;
        } else {
            return d;
        }
    }

    /** 센서 노이즈가 반영된 IMU 데이터 획득 */
    _getNoisyIMU(state) {
        const imu = {
            alpha: state.alpha,
            theta: state.theta,
            alphaDot: state.alphaDot,
            thetaDot: state.thetaDot
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

    /** CoM 역기구학에 기반한 phi 추정 */
    _getKinematicEstimate(state, model) {
        const p = model.params;
        const R = p.R, L1 = p.L1, L2 = p.L2, m1 = p.m1, m2 = p.m2;
        const e1 = L1 / 2, e2 = L2 / 2;
        const Mt = m1 + m2;

        const C1 = (m1 * e1 + m2 * L1) / Mt;
        const C2 = (m2 * e2) / Mt;

        const phi_est = -(C1 * state.alpha + C2 * state.theta) / R;
        const phiDot_est = -(C1 * state.alphaDot + C2 * state.thetaDot) / R;

        return {
            phi: phi_est,
            alpha: state.alpha,
            theta: state.theta,
            phiDot: phiDot_est,
            alphaDot: state.alphaDot,
            thetaDot: state.thetaDot
        };
    }

    /** FWE 제4식 기반 FOLD 상태 전환 초기화 */
    _initiateFold(beta, beta_dot, model, x_com, fromPhase) {
        const p = this.nominalParams || model.params;
        const m1 = p.m1, m2 = p.m2;
        const L1 = p.L1, L2 = p.L2;
        const ell1 = L1 / 2, ell2 = L2 / 2;
        const Mt = m1 + m2;
        const h_com = (m1 * ell1 + m2 * (L1 + ell2)) / Mt;
        const p1 = m1 * ell1 + m2 * L1;
        const p2 = m2 * ell2;
        const ps = p1 + p2;
        const c_foot = p2 / ps;

        // 가상 역진자 쓰러짐율 lambda 및 관성모멘트 J_tot 연산
        const I1 = m1 * L1 * L1 / 12;
        const I2 = m2 * L2 * L2 / 12;
        const J_aa = m1 * ell1 * ell1 + I1 + m2 * L1 * L1;
        const J_tt = m2 * ell2 * ell2 + I2;
        const J_at = m2 * L1 * ell2;
        const J_tot = J_aa + J_tt + 2 * J_at;
        
        const lambda = Math.sqrt((Mt * p.g * h_com) / J_tot);
        
        // 제4식의 삼각형 프로파일 외력 전파 항 D_coef_beta 계산
        const Tf = this.T_f;
        const Tw1 = this.T_w1;
        const T_total = 2 * Tf + Tw1;
        
        // k_1 = 1 기준의 f_bar_beta tilde
        const f_bar_beta = (4.0 * c_foot) / h_com; 
        
        // B_pos, B_vel (k1 = 1 기준 tilde 항)
        const B_pos = -(f_bar_beta / (lambda * lambda * Tf * Tf)) * (Math.cosh(lambda * Tf) - 2 * Math.cosh(lambda * Tf / 2) + 1);
        const B_vel = -(f_bar_beta / (lambda * Tf * Tf)) * (Math.sinh(lambda * Tf) - 2 * Math.sinh(lambda * Tf / 2));
        
        // d_bar_beta_tilde
        const term1 = (B_pos * Math.cosh(lambda * Tw1) + (B_vel / lambda) * Math.sinh(lambda * Tw1)) * Math.cosh(lambda * Tf);
        const term2 = (B_pos * lambda * Math.sinh(lambda * Tw1) + B_vel * Math.cosh(lambda * Tw1)) * (Math.sinh(lambda * Tf) / lambda);
        const d_bar_beta_tilde = term1 + term2 - B_pos;
        
        // 분모 보호 (Singularity 회피)
        let denom = d_bar_beta_tilde;
        if (Math.abs(denom) < 1e-6) {
            denom = Math.sign(denom) * 1e-6;
            if (denom === 0) denom = -1e-6; // 음수/양수 방향 보존
        }
        
        // 분자 계산 (+beta * cosh(lambda * Tf) 항 복원 및 beta_dot 감쇠 반영)
        const beta_dot_scaled = beta_dot * (this.c_beta_dot !== undefined ? this.c_beta_dot : 1.0);
        const num = beta * Math.cosh(lambda * T_total) + (beta_dot_scaled / lambda) * Math.sinh(lambda * T_total) + beta * Math.cosh(lambda * Tf);
        
        // 진짜 제4식 접기 공식 적용 (d_fold = - (1 + eps) * num / denom)
        const d_fold_val = - (1.0 + this.eps) * num / denom;
        
        // 방향 및 크기 할당
        this.fold_sign = Math.sign(d_fold_val);
        if (this.fold_sign === 0) this.fold_sign = 1.0;
        this.d_fold = Math.abs(d_fold_val);

        if (this.d_fold > this.max_delta) {
            this.d_fold = this.max_delta;
        }

        this.phase = 'FOLD';
        this.t_phase = 0.0;
        this.cycle_count++;
        this.delta_target = this.fold_sign * this._fold_profile(0.0);
        
        const loc = fromPhase ? ` (from ${fromPhase})` : "";
        this._log(`FOLD start${loc}: x_com=${(x_com*100).toFixed(1)}cm beta=${(beta*180/Math.PI).toFixed(2)}° d_fold=${(d_fold_val*180/Math.PI).toFixed(1)}°`);
    }

    /** 4단계 제어 계산 수행 및 토크 반환 */
    compute(state, model, dt) {
        // 1. 센서 노이즈 및 측정 모드 처리
        const imuData = this._getNoisyIMU(state);
        let estState;
        if (this.estimationMode === 'kinematic') {
            const noisyState = Object.assign({}, state, imuData);
            estState = this._getKinematicEstimate(noisyState, model);
        } else {
            estState = Object.assign({}, state, imuData); // ideal (또는 노이즈만 반영)
        }

        const phi_current = estState.phi;
        if (this.phi_prev === null) {
            this.phi_prev = phi_current;
        }
        const phi_crossed_zero = (this.phi_prev !== null && Math.sign(this.phi_prev) !== Math.sign(phi_current) && this.phi_prev !== phi_current);

        // 2. 물리 상수에 기반한 기하학 계산
        const p = this.nominalParams || model.params; // 플랜트 또는 공칭 파라미터
        const m1 = p.m1, m2 = p.m2;
        const L1 = p.L1, L2 = p.L2;
        const ell1 = L1 / 2, ell2 = L2 / 2;
        const Mt = m1 + m2;
        const h_com = (m1 * ell1 + m2 * (L1 + ell2)) / Mt;
        const p1 = m1 * ell1 + m2 * L1;
        const p2 = m2 * ell2;
        const ps = p1 + p2;
        const c_foot = p2 / ps;

        // CoM 절대 수평 위치 (중앙=0 기준) — 글로벌 신호
        const R = p.R;
        const x_com = R * Math.sin(estState.phi)
                     + (p1 * Math.sin(estState.alpha) + p2 * Math.sin(estState.theta)) / Mt;

        // CoM 절대 기울기 (절대 연직선 기준)
        const beta = x_com / h_com;

        // 분석적인 절대 CoM 기울기 속도 (beta_dot)
        const x_com_dot = R * Math.cos(estState.phi) * estState.phiDot
                        + (p1 * Math.cos(estState.alpha) * estState.alphaDot + p2 * Math.cos(estState.theta) * estState.thetaDot) / Mt;
        const beta_dot = x_com_dot / h_com;
        this.beta_prev = beta; 

        // 3. FWE 4단계 상태머신
        this.t_phase += dt;

        // 접기 조건 헬퍼: 절대 CoM 기울기 beta가 임계치를 초과하며 더 넘어지는 방향일 때 FOLD 트리거
        const T_pred = Math.min(0.08, this.c_pred * Math.abs(beta));
        const beta_pred = beta + T_pred * beta_dot;
        const is_falling = (Math.sign(beta) === Math.sign(beta_dot)) || (Math.abs(beta_dot) < 1e-5);
        const need_fold = Math.abs(beta_pred) > this.beta_threshold && is_falling;

        // 재트리거 조건: 동일 방향 재트리거 방지 및 40ms 과도기 쿨다운 가드 적용
        const need_fold_retrigger = need_fold && (Math.sign(beta_pred) !== this.fold_sign) && (this.t_phase > 0.04);

        if (this.phase === 'IDLE') {
            this.delta_target = 0.0;
            
            if (need_fold) {
                this._initiateFold(beta, beta_dot, model, x_com, 'IDLE');
            }
        } 
        else if (this.phase === 'FOLD') {
            const pos = this._fold_profile(this.t_phase);
            this.delta_target = this.fold_sign * pos;

            if (this.t_phase >= this.T_f) {
                this.delta_target = this.fold_sign * Math.abs(this.d_fold);
                this.phase = 'WAIT1';
                this.t_phase = 0.0;
                this._log(`WAIT1 start: holding delta=${(this.delta_target*180/Math.PI).toFixed(1)}°`);
            }
        } 
        else if (this.phase === 'WAIT1') {
            this.delta_target = this.fold_sign * Math.abs(this.d_fold);
            if (need_fold_retrigger) {
                this.phase = 'EXTEND';
                this.t_phase = 0.0;
                this._log("WAIT1 early cut (need_fold_retrigger) -> EXTEND start");
            } else if (this.t_phase >= this.T_w1) {
                this.phase = 'EXTEND';
                this.t_phase = 0.0;
                this._log(`EXTEND start: returning from delta=${(this.delta_target*180/Math.PI).toFixed(1)}°`);
            }
        } 
        else if (this.phase === 'EXTEND') {
            const pos = this._fold_profile(this.t_phase);
            this.delta_target = this.fold_sign * (Math.abs(this.d_fold) - pos);

            if (phi_crossed_zero) {
                this._log(`EXTEND early cut: phi crossed zero. x_com=${(x_com*100).toFixed(1)}cm beta=${(beta*180/Math.PI).toFixed(2)}°`);
                this.delta_target = 0.0;
                this.phase = 'IDLE';
                this.t_phase = 0.0;
                this._log(`IDLE entry (from EXTEND early cut): x_com=${(x_com*100).toFixed(1)}cm`);
            } else if (this.t_phase >= this.T_f) {
                if (this.skipWait2Entirely) {
                    this.delta_target = 0.0;
                    this.phase = 'IDLE';
                    this.t_phase = 0.0;
                    this._log(`EXTEND complete: skipping WAIT2 → IDLE.`);
                } else {
                    this.delta_target = 0.0;
                    this.phase = 'WAIT2';
                    this.t_phase = 0.0;
                    this._log(`WAIT2 start: phi=${(phi_current*180/Math.PI).toFixed(2)}°`);
                }
            }
        } 
        else if (this.phase === 'WAIT2') {
            this.delta_target = 0.0;
            if (need_fold_retrigger) {
                this._initiateFold(beta, beta_dot, model, x_com, 'WAIT2 (need_fold)');
            } else if (this.t_phase >= this.T_w2_min && phi_crossed_zero) {
                this._log(`WAIT2 complete (phi crossed zero): x_com=${(x_com*100).toFixed(1)}cm beta=${(beta*180/Math.PI).toFixed(2)}°`);
                this.phase = 'IDLE';
                this.t_phase = 0.0;
                this._log(`IDLE entry (from WAIT2): x_com=${(x_com*100).toFixed(1)}cm`);
            } else if (this.t_phase > 2.0) {
                this._log(`WAIT2 timeout → IDLE. phi=${(phi_current*180/Math.PI).toFixed(2)}°`);
                this.phase = 'IDLE';
                this.t_phase = 0.0;
            }
        }

        // 4. 힙 서보 추종 PD 토크 계산
        // delta_actual = theta - alpha
        const delta_actual = estState.theta - estState.alpha;
        const delta_dot_actual = estState.thetaDot - estState.alphaDot;

        let tau = this.Kp_servo * (this.delta_target - delta_actual) - this.Kd_servo * delta_dot_actual;
        
        // 물리적 최대 토크 제한 클램프
        tau = Math.max(-p.tauMax, Math.min(p.tauMax, tau));
        
        this._appliedTau = tau;
        this.phi_prev = phi_current;
        return tau;
    }

    getDelayInfo() {
        const modeLabels = {
            'ideal': '🎯 완벽한 센싱',
            'kinematic': '📐 CoM 역기구학'
        };
        return {
            totalMs: 0,
            sensorMs: modeLabels[this.estimationMode] || '알 수 없음',
            actuatorMs: 0,
            estimationMode: this.estimationMode,
            augmented: false,
            phaseName: this.phase
        };
    }
};

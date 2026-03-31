/**
 * Slackline Balance Simulator V8 — Discrete Kalman Filter State Observer
 * 
 * IMU 센서(α, θ, α̇, θ̇)와 제어 입력(τ)만으로
 * 직접 측정 불가능한 발 위치(φ, φ̇)를 추정합니다.
 * 
 * 알고리즘:
 *   1. Predict: x̂⁻ = Ad · x̂ + Bd · u
 *   2. Correct: x̂ = x̂⁻ + L · (y - C · x̂⁻)
 * 
 * 여기서:
 *   x = [φ, α, θ, φ̇, α̇, θ̇] (6차원 상태)
 *   u = τ (힙 토크, 1차원)
 *   y = [α, θ, α̇, θ̇] (IMU 관측, 4차원)
 *   C = 관측행렬 (4×6) — α, θ, α̇, θ̇만 관측
 *   L = 관측기 게인 (6×4) — 정상상태 칼만 게인
 */
var SL = SL || {};

SL.StateObserver = class {
    constructor() {
        this.n = 6;  // 상태 차원
        this.m = 4;  // 관측 차원
        this.initialized = false;

        // 이산 시스템 행렬
        this.Ad = null;  // 6×6
        this.Bd = null;  // 6×1
        this.C = null;   // 4×6
        this.L = null;   // 6×4 관측기 게인

        // 추정 상태
        this.xHat = new Float64Array(6);  // [φ̂, α̂, θ̂, φ̂̇, α̂̇, θ̂̇]

        // Kalman 필터 공분산
        this.P = null;   // 6×6 오차 공분산
        this.Q = null;   // 6×6 프로세스 노이즈
        this.R_noise = null;  // 4×4 관측 노이즈
    }

    /**
     * 관측기 초기화
     * @param {Object} params - 물리 파라미터 (SL.Params)
     * @param {Object} lqrController - LQR 제어기 (linearize, discretize 메서드 사용)
     * @param {Object} model - 물리 모델 (computeDerivatives, rk4Step 사용)
     */
    init(params, lqrController, model) {
        this._params = params;
        this._model = model;  // 비선형 예측에 사용
        const M = SL.Mat;
        const n = this.n;
        const m = this.m;

        // 1. 선형화 → 이산화 (LQR 제어기의 기존 인프라 재활용)
        const lin = lqrController.linearize(params);
        if (!lin) {
            console.error('Observer: linearization failed');
            return;
        }

        const dt = params.dt;  // 1ms
        const disc = lqrController.discretize(lin.Ac, lin.Bc, dt);
        this.Ad = disc.Ad;
        this.Bd = disc.Bd;

        // 2. 관측 행렬 C (4×6): y = [α, θ, α̇, θ̇]
        //    x = [φ(0), α(1), θ(2), φ̇(3), α̇(4), θ̇(5)]
        this.C = M.zeros(m, n);
        this.C[0][1] = 1;  // α
        this.C[1][2] = 1;  // θ
        this.C[2][4] = 1;  // α̇
        this.C[3][5] = 1;  // θ̇

        // 3. 관측 가능성 검증 (옵션 - 로그만 출력)
        this._checkObservability();

        // 4. 칼만 필터 노이즈 행렬 설정
        //    Q: 프로세스 노이즈 (모델 불확실성)
        this.Q = M.zeros(n, n);
        // φ, φ̇는 직접 관측 불가하므로 불확실성을 크게
        this.Q[0][0] = 1e-4;   // φ 위치 불확실성
        this.Q[1][1] = 1e-6;   // α (IMU 직접 측정)
        this.Q[2][2] = 1e-6;   // θ (IMU 직접 측정)
        this.Q[3][3] = 1e-3;   // φ̇ 속도 불확실성
        this.Q[4][4] = 1e-5;   // α̇ 
        this.Q[5][5] = 1e-5;   // θ̇

        //    R: 관측 노이즈 (센서 측정 오차)
        this.R_noise = M.zeros(m, m);
        this.R_noise[0][0] = 1e-4;  // α 측정 노이즈
        this.R_noise[1][1] = 1e-4;  // θ 측정 노이즈
        this.R_noise[2][2] = 1e-3;  // α̇ 측정 노이즈
        this.R_noise[3][3] = 1e-3;  // θ̇ 측정 노이즈

        // 5. 초기 공분산 (불확실성 높게 시작)
        this.P = M.zeros(n, n);
        for (let i = 0; i < n; i++) this.P[i][i] = 1.0;

        // 6. 정상상태 게인 사전 계산 (DARE 풀이)
        this._computeSteadyStateGain();

        // 7. 추정 상태 초기화
        this.xHat = new Float64Array(6);

        this.initialized = true;
        console.log('V8 State Observer initialized (Kalman Filter)');
    }

    /**
     * 관측 가능성 검증
     * rank([C; C·A; C·A²; ...]) = n 이면 관측 가능
     */
    _checkObservability() {
        const M = SL.Mat;
        const n = this.n, m = this.m;
        // 간이 검증: C·A^k (k=0..n-1) 의 조건수 확인
        // 풀 랭크 검증은 SVD가 필요하므로, 여기서는 행렬식 기반 간이 테스트
        let O = M.zeros(m * n, n);  // 관측가능성 행렬 (24×6)
        let CA_k = M.copy(this.C, m, n);  // C·A^0 = C
        for (let k = 0; k < n; k++) {
            for (let i = 0; i < m; i++)
                for (let j = 0; j < n; j++)
                    O[k * m + i][j] = CA_k[i][j];
            if (k < n - 1) {
                CA_k = M.mul(CA_k, this.Ad, m, n, n);
            }
        }
        // O^T · O 의 행렬식으로 간이 판정
        const OtO = M.mul(M.transpose(O, m * n, n), O, n, m * n, n);
        let det = this._det6x6(OtO);
        const observable = Math.abs(det) > 1e-20;
        console.log(`Observer: Observability matrix det = ${det.toExponential(3)}, observable = ${observable}`);
    }

    /** 6×6 행렬식 (LU 분해 기반) */
    _det6x6(A) {
        const n = 6;
        // LU 분해로 행렬식 계산
        const L = SL.Mat.zeros(n, n);
        const U = SL.Mat.copy(A, n, n);
        let det = 1;
        for (let k = 0; k < n; k++) {
            if (Math.abs(U[k][k]) < 1e-30) return 0;
            det *= U[k][k];
            for (let i = k + 1; i < n; i++) {
                const factor = U[i][k] / U[k][k];
                for (let j = k; j < n; j++) {
                    U[i][j] -= factor * U[k][j];
                }
            }
        }
        return det;
    }

    /**
     * 정상상태 칼만 게인 계산
     * DARE(이산 대수 리카티 방정식)를 풀어 정상상태 P를 구하고,
     * L = P·Cᵀ·(C·P·Cᵀ + R)⁻¹ 를 계산합니다.
     * 
     * 쌍대성: LQR의 DARE와 동일한 구조
     *   LQR: A, B, Q_lqr, R_lqr → K
     *   LQE: Aᵀ, Cᵀ, Q, R → Lᵀ
     */
    _computeSteadyStateGain() {
        const M = SL.Mat;
        const n = this.n, m = this.m;

        // 쌍대 DARE: P = Ad · P · Adᵀ - Ad · P · Cᵀ · (C·P·Cᵀ + R)⁻¹ · C · P · Adᵀ + Q
        // 반복법으로 수렴시킴
        let P = M.copy(this.P, n, n);
        const At = this.Ad;
        const AtT = M.transpose(At, n, n);
        const Ct = this.C;
        const CtT = M.transpose(Ct, m, n);  // 6×4
        const Qn = this.Q;
        const Rn = this.R_noise;

        const maxIter = 500;
        const tol = 1e-10;

        for (let iter = 0; iter < maxIter; iter++) {
            // S = C · P · Cᵀ + R  (4×4)
            const CP = M.mul(Ct, P, m, n, n);      // 4×6
            const S = M.add(M.mul(CP, CtT, m, n, m), Rn, m, m);  // 4×4

            // S⁻¹ (4×4 역행렬)
            const Sinv = this._inv4x4(S);
            if (!Sinv) {
                console.warn('Observer: S matrix singular at iter', iter);
                break;
            }

            // K_gain = P · Cᵀ · S⁻¹  (6×4)
            const PCtT = M.mul(P, CtT, n, n, m);    // 6×4
            const Kgain = M.mul(PCtT, Sinv, n, m, m); // 6×4

            // P_new = (I - K·C) · P   → Joseph form for numerical stability
            // P_new = A · P · Aᵀ - A · P · Cᵀ · S⁻¹ · C · P · Aᵀ + Q
            const AP = M.mul(At, P, n, n, n);        // 6×6
            const APAt = M.mul(AP, AtT, n, n, n);    // 6×6
            const APCtT = M.mul(AP, CtT, n, n, m);   // 6×4
            const SinvCPAt = M.mul(M.mul(Sinv, CP, m, m, n), AtT, m, n, n);  // 4×6
            const correction = M.mul(APCtT, SinvCPAt, n, m, n); // 6×6

            const Pnew = M.add(M.sub(APAt, correction, n, n), Qn, n, n);

            // 수렴 확인
            const diff = M.normDiff(Pnew, P, n, n);
            P = Pnew;

            if (diff < tol) {
                console.log(`Observer: DARE converged at iter ${iter}, diff=${diff.toExponential(3)}`);
                break;
            }
        }

        // 정상상태 칼만 게인: L = P · Cᵀ · (C·P·Cᵀ + R)⁻¹
        const CP_final = M.mul(Ct, P, m, n, n);
        const S_final = M.add(M.mul(CP_final, CtT, m, n, m), Rn, m, m);
        const Sinv_final = this._inv4x4(S_final);

        if (Sinv_final) {
            const PCtT_final = M.mul(P, CtT, n, n, m);
            this.L = M.mul(PCtT_final, Sinv_final, n, m, m);  // 6×4

            // 게인 로그
            console.log('Observer gain L:');
            for (let i = 0; i < n; i++) {
                const row = [];
                for (let j = 0; j < m; j++) row.push(this.L[i][j].toFixed(6));
                console.log(`  L[${i}] = [${row.join(', ')}]`);
            }
        } else {
            console.error('Observer: Failed to compute steady-state gain');
            // 폴백: 고정 게인 사용
            this.L = M.zeros(n, m);
            this.L[0][0] = 0.1; this.L[0][1] = 0.05;
            this.L[3][2] = 0.1; this.L[3][3] = 0.05;
            // IMU 직접 관측 상태는 강하게 보정
            this.L[1][0] = 0.9; this.L[2][1] = 0.9;
            this.L[4][2] = 0.9; this.L[5][3] = 0.9;
        }

        this.P = P;
    }

    /** 4×4 역행렬 (Gauss-Jordan) */
    _inv4x4(A) {
        const n = 4;
        const aug = [];
        for (let i = 0; i < n; i++) {
            aug[i] = new Float64Array(2 * n);
            for (let j = 0; j < n; j++) aug[i][j] = A[i][j];
            aug[i][n + i] = 1;
        }

        for (let col = 0; col < n; col++) {
            // 피봇 선택
            let maxVal = Math.abs(aug[col][col]);
            let maxRow = col;
            for (let row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > maxVal) {
                    maxVal = Math.abs(aug[row][col]);
                    maxRow = row;
                }
            }
            if (maxVal < 1e-30) return null;

            // 행 교환
            if (maxRow !== col) {
                const tmp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = tmp;
            }

            // 소거
            const pivot = aug[col][col];
            for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
            for (let row = 0; row < n; row++) {
                if (row === col) continue;
                const factor = aug[row][col];
                for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
            }
        }

        const inv = SL.Mat.zeros(n, n);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                inv[i][j] = aug[i][n + j];
        return inv;
    }

    /**
     * 추정 상태 초기화 (리셋 또는 초기 조건 설정)
     */
    resetState(initialState) {
        if (initialState) {
            this.xHat[0] = initialState.phi || 0;
            this.xHat[1] = initialState.alpha || 0;
            this.xHat[2] = initialState.theta || 0;
            this.xHat[3] = initialState.phiDot || 0;
            this.xHat[4] = initialState.alphaDot || 0;
            this.xHat[5] = initialState.thetaDot || 0;
        } else {
            this.xHat.fill(0);
        }
    }

    /**
     * 한 스텝 실행: Predict + Correct (결합)
     * @param {number} u - 제어 입력 (토크 τ)
     * @param {Object} imuState - IMU 측정값 {alpha, theta, alphaDot, thetaDot}
     * @returns {Object} 추정된 전체 상태
     */
    step(u, imuState) {
        if (!this.initialized) return null;

        const M = SL.Mat;
        const n = this.n, m = this.m;
        const dt = this._params.dt;

        // === NaN/Inf 방어: 입력 검증 ===
        if (!isFinite(u)) u = 0;
        if (!isFinite(imuState.alpha) || !isFinite(imuState.theta)) {
            return this.getEstimatedState();
        }

        // === 1단계: Predict (비선형 모델 기반 시간 업데이트) ===
        const estState = {
            phi: this.xHat[0],
            alpha: this.xHat[1],
            theta: this.xHat[2],
            phiDot: this.xHat[3],
            alphaDot: this.xHat[4],
            thetaDot: this.xHat[5]
        };

        // RK4 예측 전 입력 토크 클램프 (물리적 한계)
        const uClamped = Math.max(-500, Math.min(500, u));
        const nextState = this._model.rk4Step(estState, uClamped, dt);
        const xPred = new Float64Array([
            nextState.phi, nextState.alpha, nextState.theta,
            nextState.phiDot, nextState.alphaDot, nextState.thetaDot
        ]);

        // === RK4 결과 NaN 체크 ===
        let hasNaN = false;
        for (let i = 0; i < n; i++) {
            if (!isFinite(xPred[i])) { hasNaN = true; break; }
        }
        if (hasNaN) {
            // RK4가 발산 → IMU 측정값으로 직접 상태 리셋
            console.warn('Observer: RK4 prediction diverged, resetting to IMU');
            this.xHat[1] = imuState.alpha;
            this.xHat[2] = imuState.theta;
            this.xHat[4] = imuState.alphaDot;
            this.xHat[5] = imuState.thetaDot;
            // φ, φ̇는 0으로 리셋
            this.xHat[0] = 0;
            this.xHat[3] = 0;
            return this.getEstimatedState();
        }

        // === 2단계: Correct (선형 게인으로 측정 업데이트) ===
        const y = new Float64Array([
            imuState.alpha, imuState.theta,
            imuState.alphaDot, imuState.thetaDot
        ]);
        const yPred = M.mulVec(this.C, xPred, m, n);

        // innovation (측정 잔차): e = y - ŷ
        const innovation = new Float64Array(m);
        for (let i = 0; i < m; i++) {
            innovation[i] = y[i] - yPred[i];
        }

        // === Innovation 클램핑 (교란 시 과대 보정 방지) ===
        // 각도 잔차: ±0.5 rad (≈29°), 각속도 잔차: ±5 rad/s
        const innovLimits = [0.5, 0.5, 5.0, 5.0];
        for (let i = 0; i < m; i++) {
            innovation[i] = Math.max(-innovLimits[i], Math.min(innovLimits[i], innovation[i]));
        }

        // x̂ = x̂⁻ + L · e
        const correction = M.mulVec(this.L, innovation, n, m);
        for (let i = 0; i < n; i++) {
            this.xHat[i] = xPred[i] + correction[i];
        }

        // === 상태 포화 (물리적 한계 내로 유지) ===
        // 각도: ±π/2 (90°), 각속도: ±30 rad/s
        const angleLim = Math.PI / 2;
        const velLim = 30;
        for (let i = 0; i < 3; i++) {
            this.xHat[i] = Math.max(-angleLim, Math.min(angleLim, this.xHat[i]));
        }
        for (let i = 3; i < 6; i++) {
            this.xHat[i] = Math.max(-velLim, Math.min(velLim, this.xHat[i]));
        }

        // === 최종 NaN 체크 ===
        for (let i = 0; i < n; i++) {
            if (!isFinite(this.xHat[i])) {
                console.warn('Observer: NaN detected, full reset');
                this.xHat[0] = 0; this.xHat[3] = 0;
                this.xHat[1] = imuState.alpha;
                this.xHat[2] = imuState.theta;
                this.xHat[4] = imuState.alphaDot;
                this.xHat[5] = imuState.thetaDot;
                break;
            }
        }

        return this.getEstimatedState();
    }

    /**
     * 추정된 전체 상태 반환
     */
    getEstimatedState() {
        return {
            phi: this.xHat[0],
            alpha: this.xHat[1],
            theta: this.xHat[2],
            phiDot: this.xHat[3],
            alphaDot: this.xHat[4],
            thetaDot: this.xHat[5]
        };
    }

    /**
     * 추정 오차 정보 (디버깅용)
     */
    getEstimationInfo() {
        return {
            phiEst: this.xHat[0],
            phiDotEst: this.xHat[3],
            initialized: this.initialized
        };
    }
};

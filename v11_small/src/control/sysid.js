/**
 * Slackline Balance Simulator V10 — 온라인 시스템 식별
 * 
 * 예측 오차 방법 (Prediction Error Method):
 *   1. 현재 추정 파라미터로 K스텝 시뮬레이션 → 예측 궤적
 *   2. 실제 관측 궤적과 비교 → 예측 오차
 *   3. 파라미터를 미소 변동하여 수치 Jacobian 계산
 *   4. 경사 하강으로 파라미터 업데이트
 * 
 * 장점: 운동방정식의 커플링을 완벽히 반영 (풀 모델 사용)
 */
var SL = SL || {};

SL.SystemIdentifier = class {
    constructor() {
        // 추정 대상: m1, m2 스케일 (공칭 대비 비율)
        this.nParams = 2;
        this.paramScale = new Float64Array([1.0, 1.0]); // [m1_scale, m2_scale]

        // 학습률
        this.learningRate = 0.05;

        // 데이터 버퍼 (최근 상태/토크 이력)
        this._buffer = [];        // [{state, tau}]
        this._bufferSize = 800;   // 800ms 분량
        this._predictionHorizon = 500; // 500스텝(500ms) 앞 예측

        // 카운터
        this._updateCount = 0;
        this._warmupCount = 0;
        this._identifyInterval = 300; // 300스텝마다 식별 실행

        // 공칭 파라미터
        this._nominalParams = null;

        // 활성 상태
        this.active = false;
        this.converged = false;
        this._errorHistory = [];
        this._paramHistory = { m1: [], m2: [], time: [] };

        // 수렴 판정
        this._recentErrors = [];
        this._maxRecentErrors = 10;
    }

    init(nominalParams) {
        this._nominalParams = Object.assign({}, nominalParams);
        this.paramScale[0] = 1.0;
        this.paramScale[1] = 1.0;
        this._buffer = [];
        this._updateCount = 0;
        this._warmupCount = 0;
        this.converged = false;
        this._errorHistory = [];
        this._recentErrors = [];
        this._paramHistory = { m1: [], m2: [], time: [] };
    }

    reset() {
        if (this._nominalParams) this.init(this._nominalParams);
    }

    /**
     * 매 시뮬레이션 스텝마다 호출
     */
    update(state, tau, dt) {
        if (!this.active || !this._nominalParams) return;

        // 버퍼에 상태/토크 저장
        this._buffer.push({
            state: {
                phi: state.phi, alpha: state.alpha, theta: state.theta,
                phiDot: state.phiDot, alphaDot: state.alphaDot, thetaDot: state.thetaDot,
                time: state.time
            },
            tau: tau
        });

        // 버퍼 크기 제한
        while (this._buffer.length > this._bufferSize) {
            this._buffer.shift();
        }

        this._updateCount++;

        // 워밍업 (버퍼가 충분히 찰 때까지 대기)
        if (this._buffer.length < this._predictionHorizon + 10) return;

        // identifyInterval 스텝마다 식별 실행
        if (this._updateCount % this._identifyInterval !== 0) return;

        this._runIdentification(dt);
    }

    /**
     * 예측 오차 기반 파라미터 업데이트
     */
    _runIdentification(dt) {
        const horizon = this._predictionHorizon;
        const bufLen = this._buffer.length;

        // 시작점: 버퍼의 중간 지점에서 horizon 만큼 전진 비교
        const startIdx = bufLen - horizon - 10;
        if (startIdx < 0) return;

        const startState = this._buffer[startIdx].state;

        // 1. 현재 추정 파라미터로 예측 오차 계산
        const errorBase = this._computePredictionError(startIdx, horizon, this.paramScale, dt);
        if (!isFinite(errorBase)) return;

        // 2. 수치 Jacobian: 각 파라미터를 미소 변동 (중앙 차분)
        const delta = 0.02; // 2% 변동
        const grad = new Float64Array(this.nParams);

        for (let p = 0; p < this.nParams; p++) {
            const scalePlus = new Float64Array(this.paramScale);
            const scaleMinus = new Float64Array(this.paramScale);
            scalePlus[p] += delta;
            scaleMinus[p] -= delta;

            const errPlus = this._computePredictionError(startIdx, horizon, scalePlus, dt);
            const errMinus = this._computePredictionError(startIdx, horizon, scaleMinus, dt);

            if (isFinite(errPlus) && isFinite(errMinus)) {
                grad[p] = (errPlus - errMinus) / (2 * delta); // 중앙 차분
            } else if (isFinite(errPlus)) {
                grad[p] = (errPlus - errorBase) / delta;
            } else {
                grad[p] = 0;
            }
        }

        // 3. 경사 하강 업데이트 (Adam-like adaptive)
        let gradNorm = 0;
        for (let p = 0; p < this.nParams; p++) gradNorm += grad[p] * grad[p];
        gradNorm = Math.sqrt(gradNorm);

        if (gradNorm > 1e-12) {
            // 그래디언트가 클 때만 정규화, 작으면 그대로 사용
            const lr = gradNorm > 1 ? this.learningRate / gradNorm : this.learningRate;
            for (let p = 0; p < this.nParams; p++) {
                this.paramScale[p] -= lr * grad[p];
                this.paramScale[p] = Math.max(0.3, Math.min(3.0, this.paramScale[p]));
            }
            console.log(`SysID: grad=[${grad[0].toExponential(2)}, ${grad[1].toExponential(2)}], ` +
                `scale=[${this.paramScale[0].toFixed(4)}, ${this.paramScale[1].toFixed(4)}], err=${errorBase.toExponential(2)}`);
        }

        // 4. 이력 기록
        const nom = this._nominalParams;
        this._paramHistory.m1.push(this.paramScale[0] * nom.m1);
        this._paramHistory.m2.push(this.paramScale[1] * nom.m2);
        this._paramHistory.time.push(this._updateCount);

        // 5. 수렴 판정: 최근 오차가 충분히 작고 안정적이면
        this._recentErrors.push(errorBase);
        if (this._recentErrors.length > this._maxRecentErrors) {
            this._recentErrors.shift();
        }
        if (this._recentErrors.length >= this._maxRecentErrors) {
            const avgError = this._recentErrors.reduce((a, b) => a + b, 0) / this._recentErrors.length;
            const errorVariance = this._recentErrors.reduce((a, b) => a + (b - avgError) ** 2, 0) / this._recentErrors.length;
            // 오차 분산이 작으면 수렴
            this.converged = (errorVariance < avgError * 0.01 + 1e-8) && this._recentErrors.length >= this._maxRecentErrors;
        }
    }

    /**
     * 예측 오차 계산: startIdx에서 horizon 스텝 전진 시뮬레이션하여
     * 실제 관측 궤적과의 차이를 MSE로 반환
     */
    _computePredictionError(startIdx, horizon, paramScales, dt) {
        const nom = this._nominalParams;

        // 임시 파라미터 생성 (Object.assign은 getter를 복사하지 않으므로 직접 추가)
        const testParams = Object.assign({}, nom);
        testParams.m1 = nom.m1 * paramScales[0];
        testParams.m2 = nom.m2 * paramScales[1];

        // 파생 상수를 getter로 재정의 (model.js의 computeDerivatives에서 사용)
        Object.defineProperties(testParams, {
            ell1:   { get() { return this.L1 / 2; } },
            ell2:   { get() { return this.L2 / 2; } },
            I1:     { get() { return this.m1 * this.L1 * this.L1 / 12; } },
            I2:     { get() { return this.m2 * this.L2 * this.L2 / 12; } },
            p1:     { get() { return this.m1 * this.ell1 + this.m2 * this.L1; } },
            p2:     { get() { return this.m2 * this.ell2; } },
            p3:     { get() { return this.m2 * this.L1 * this.ell2; } },
            Mtotal: { get() { return this.m1 + this.m2; } },
            M11:    { get() { return this.Mtotal * this.R * this.R; } },
            M22:    { get() { return this.m1 * this.ell1 * this.ell1 + this.I1 + this.m2 * this.L1 * this.L1; } },
            M33:    { get() { return this.m2 * this.ell2 * this.ell2 + this.I2; } }
        });

        // 임시 모델 생성
        const testModel = new SL.Model(testParams);

        // 시작 상태
        let simState = Object.assign({}, this._buffer[startIdx].state);

        let totalError = 0;
        let count = 0;

        // horizon 스텝 전진 시뮬레이션
        for (let k = 0; k < horizon; k++) {
            const realIdx = startIdx + k;
            if (realIdx >= this._buffer.length - 1) break;

            const tau = this._buffer[realIdx].tau;
            const realNext = this._buffer[realIdx + 1].state;

            // 시뮬레이션 1스텝 전진
            simState = testModel.rk4Step(simState, tau, dt);

            // NaN 체크
            if (!isFinite(simState.alpha) || !isFinite(simState.theta)) {
                return Infinity;
            }

            // 관측 가능한 상태(α, θ)만 비교
            const eAlpha = simState.alpha - realNext.alpha;
            const eTheta = simState.theta - realNext.theta;
            totalError += eAlpha * eAlpha + eTheta * eTheta;
            count++;
        }

        return count > 0 ? totalError / count : Infinity;
    }

    getEstimatedParams() {
        if (!this._nominalParams) return null;
        const nom = this._nominalParams;
        return {
            m1: this.paramScale[0] * nom.m1,
            m2: this.paramScale[1] * nom.m2,
            m1_scale: this.paramScale[0],
            m2_scale: this.paramScale[1]
        };
    }

    getConvergenceInfo() {
        return {
            p0: this.paramScale[0],
            p1: this.paramScale[1],
            converged: this.converged,
            updates: Math.floor(this._updateCount / this._identifyInterval),
            error: this._recentErrors.length > 0 ?
                this._recentErrors[this._recentErrors.length - 1] : Infinity
        };
    }
};

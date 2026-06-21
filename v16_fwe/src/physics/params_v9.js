/**
 * Slackline Balance Simulator V9 — 파라미터 관리 (Plant/Nominal 분리)
 * 
 * nominalParams: 제어기/옵저버가 사용하는 설계 파라미터 (고정)
 * plantParams:   실제 물리 시뮬레이션이 사용하는 파라미터 (편차 적용)
 */
var SL = SL || {};

SL.ParamManager = class {
    constructor() {
        // 편차 비율 (0 = 편차 없음, 0.1 = +10%, -0.2 = -20%)
        this.mismatch = {
            m1: 0, m2: 0,
            L1: 0, L2: 0,
            R: 0
        };

        // 공칭 파라미터 스냅샷 (init 시 SL.Params에서 복사)
        this._nominalSnapshot = {};
    }

    /** SL.Params 현재값을 공칭 기준으로 저장 */
    captureNominal() {
        // 먼저 공칭값을 복원하여 편차 누적 방지
        // (SL.Params에 이미 편차가 적용되어 있을 수 있으므로)
        if (Object.keys(this._nominalSnapshot).length > 0) {
            // 기존 공칭값 복원 후 새 공칭값 캐쳐
            ['m1', 'm2', 'L1', 'L2', 'R'].forEach(k => {
                SL.Params[k] = this._nominalSnapshot[k];
            });
        }
        const keys = ['m1', 'm2', 'L1', 'L2', 'R',
                       'b_phi', 'b_alpha', 'b_theta', 'g',
                       'dt', 'stepsPerFrame', 'speedMultiplier',
                       'phi0', 'alpha0', 'theta0',
                       'phiDot0', 'alphaDot0', 'thetaDot0',
                       'phiMax', 'tauMax', 'controllerOn',
                       'lowerWidth', 'upperWidth',
                       'L_pole', 'L0', 'bodyDepth', 'showArc'];
        this._nominalSnapshot = {};
        keys.forEach(k => {
            this._nominalSnapshot[k] = SL.Params[k];
        });
        // 편차 재적용
        this.applyPlantToGlobal();
    }

    /** 공칭 파라미터 반환 (제어기/옵저버용) — getter 포함 프록시 */
    get nominalParams() {
        const snap = Object.assign({}, this._nominalSnapshot);
        // getter 프로퍼티 바인딩
        return this._attachDerivedGetters(snap);
    }

    /** 플랜트 파라미터 반환 (물리 시뮬레이션용) — 편차 적용 */
    get plantParams() {
        const plant = Object.assign({}, this._nominalSnapshot);
        // 편차 적용
        ['m1', 'm2', 'L1', 'L2', 'R'].forEach(k => {
            plant[k] = this._nominalSnapshot[k] * (1 + this.mismatch[k]);
        });
        return this._attachDerivedGetters(plant);
    }

    /** 플랜트 파라미터를 SL.Params에 동기화 (물리엔진이 SL.Params 직접 참조하므로) */
    applyPlantToGlobal() {
        const pp = this.plantParams;
        ['m1', 'm2', 'L1', 'L2', 'R'].forEach(k => {
            SL.Params[k] = pp[k];
        });
    }

    /** 공칭 파라미터를 SL.Params에 복원 */
    restoreNominalToGlobal() {
        ['m1', 'm2', 'L1', 'L2', 'R'].forEach(k => {
            SL.Params[k] = this._nominalSnapshot[k];
        });
    }

    /** 파생 상수 getter 부착 */
    _attachDerivedGetters(obj) {
        Object.defineProperties(obj, {
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
            M33:    { get() { return this.m2 * this.ell2 * this.ell2 + this.I2; } },
            ranges: { get() { return SL.Params.ranges; } }
        });
        return obj;
    }

    /** 편차 설정 */
    setMismatch(key, ratio) {
        if (key in this.mismatch) {
            this.mismatch[key] = ratio;
        }
    }

    /** 현재 편차 요약 문자열 */
    getMismatchSummary() {
        const parts = [];
        for (const k of Object.keys(this.mismatch)) {
            const pct = (this.mismatch[k] * 100).toFixed(0);
            if (pct !== '0') parts.push(`${k}: ${pct > 0 ? '+' : ''}${pct}%`);
        }
        return parts.length > 0 ? parts.join(', ') : '편차 없음';
    }
};

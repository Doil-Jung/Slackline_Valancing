/**
 * Slackline Balance Simulator V8 — LQR Controller + State Observer
 * 
 * 세 가지 φ 관측 모드:
 *   1. ideal: 완벽한 센싱 (기존 시뮬레이터와 동일)
 *   2. kinematic: 정적 CoM 역기구학 추정 (V7)
 *   3. observer: 이산 칼만 필터 상태 관측기 (V8 신규)
 * 
 * 상태증강 방법:
 *   - 시스템을 평형점에서 선형화 → A_c, B_c
 *   - dt_aug (5ms) 간격으로 이산화 → A_d, B_d
 *   - 입력 지연 N = totalDelayMs / dt_aug
 *   - 증강 상태: z = [x, u(t-1), ..., u(t-N)]  (6+N 차원)
 *   - 이산 대수 리카티 방정식(DARE) 풀이 → 최적 게인 K_aug
 */
var SL = SL || {};

// ============================================================
//  행렬 유틸리티 (2D 배열 기반, 소규모 행렬용)
// ============================================================
SL.Mat = {
    /** n×m 영행렬 생성 */
    zeros(n, m) {
        const A = [];
        for (let i = 0; i < n; i++) { A[i] = new Float64Array(m); }
        return A;
    },

    /** n×n 단위행렬 */
    eye(n) {
        const I = this.zeros(n, n);
        for (let i = 0; i < n; i++) I[i][i] = 1;
        return I;
    },

    /** C = A × B,  A(n×m), B(m×p) → C(n×p) */
    mul(A, B, n, m, p) {
        const C = this.zeros(n, p);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < p; j++) {
                let s = 0;
                for (let k = 0; k < m; k++) s += A[i][k] * B[k][j];
                C[i][j] = s;
            }
        return C;
    },

    /** C = A + B (n×m) */
    add(A, B, n, m) {
        const C = this.zeros(n, m);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < m; j++)
                C[i][j] = A[i][j] + B[i][j];
        return C;
    },

    /** C = A − B (n×m) */
    sub(A, B, n, m) {
        const C = this.zeros(n, m);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < m; j++)
                C[i][j] = A[i][j] - B[i][j];
        return C;
    },

    /** C = s × A (n×m) */
    scale(A, s, n, m) {
        const C = this.zeros(n, m);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < m; j++)
                C[i][j] = A[i][j] * s;
        return C;
    },

    /** Aᵀ (n×m → m×n) */
    transpose(A, n, m) {
        const T = this.zeros(m, n);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < m; j++)
                T[j][i] = A[i][j];
        return T;
    },

    /** 내적: a·b (길이 n) */
    dot(a, b, n) {
        let s = 0;
        for (let i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    },

    /** 행렬-벡터 곱 A(n×m) × v(m×1) → w(n×1), 벡터는 1D 배열 */
    mulVec(A, v, n, m) {
        const w = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            let s = 0;
            for (let j = 0; j < m; j++) s += A[i][j] * v[j];
            w[i] = s;
        }
        return w;
    },

    /** 대칭 3×3 역행렬 */
    inv3x3sym(M) {
        const [a, b, c, d, e, f] = [M[0][0], M[0][1], M[0][2], M[1][1], M[1][2], M[2][2]];
        const det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
        if (Math.abs(det) < 1e-15) return null;
        const id = 1.0 / det;
        const R = this.zeros(3, 3);
        R[0][0] = (d * f - e * e) * id;
        R[0][1] = R[1][0] = -(b * f - c * e) * id;
        R[0][2] = R[2][0] = (b * e - c * d) * id;
        R[1][1] = (a * f - c * c) * id;
        R[1][2] = R[2][1] = -(a * e - b * c) * id;
        R[2][2] = (a * d - b * b) * id;
        return R;
    },

    /** 행렬 복사 */
    copy(A, n, m) {
        const C = this.zeros(n, m);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < m; j++)
                C[i][j] = A[i][j];
        return C;
    },

    /** Frobenius norm 차이 */
    normDiff(A, B, n, m) {
        let s = 0;
        for (let i = 0; i < n; i++)
            for (let j = 0; j < m; j++) {
                const d = A[i][j] - B[i][j];
                s += d * d;
            }
        return Math.sqrt(s);
    }
};

// ============================================================
//  LQR 제어기 (기본 + 상태증강)
// ============================================================
SL.LQRController = class {
    constructor() {
        // 기본 LQR 게인 (지연 0용, analysis/controllability.html에서 계산)
        this.K_base = [-6454.691363, -5908.227575, -2148.820176, -1312.073937, -1095.998565, -579.948137];

        // 현재 사용 중인 게인
        this.K = this.K_base.slice();

        // 게인 스케일링
        this.gainScale = 1.0;

        // 적분 보정
        this.integral = 0;
        this.maxIntegral = 50;
        this.Ki = 0;

        // === 발위치 추정 모드 (V8) ===
        this.dtMs = 1;
        this.totalDelayMs = 0;
        this.actuatorDelaySteps = 0;
        this.estimationMode = 'ideal'; // 'ideal' | 'kinematic' | 'observer'
        this.observer = null; // SL.StateObserver 인스턴스 (외부에서 주입)
        this.sensorNoise = null; // V9: SL.SensorNoise 인스턴스

        // Buffers
        this.stateBuffer = [];
        this.actuatorBuffer = [];

        // === 상태증강 모드 ===
        this.augmented = false;        // 상태증강 모드 ON/OFF
        this.K_aug = null;             // 증강 게인 (1 × (6+N) 배열)
        this.N_aug = 0;                // 증강 지연 스텝 수
        this.dtAug = 5;               // 증강 이산화 간격 (ms)
        this.uPipeline = [];           // 입력 파이프라인 [u(t-1), ..., u(t-N)]
        this.augValid = false;         // 증강 게인이 유효한지

        // 캐시 (파라미터 변경 시 재계산 방지)
        this._lastAugParams = null;

        // V10: R 스케일 (LQR 튜닝)
        this.rScale = 1.0;
    }

    // ============================================================
    //  선형화 & 이산화 & DARE
    // ============================================================

    /**
     * 시스템을 평형점(x=0, u=0)에서 선형화
     * @returns {{Ac: Array, Bc: Array}} 연속시간 상태공간 행렬
     */
    linearize(params) {
        const { m1, m2, R, g, L1, L2, b_phi, b_alpha, b_theta } = params;
        const ell1 = L1 / 2, ell2 = L2 / 2;
        const I1 = m1 * L1 * L1 / 12, I2 = m2 * L2 * L2 / 12;
        const p1 = m1 * ell1 + m2 * L1;
        const p2 = m2 * ell2;
        const p3 = m2 * L1 * ell2;
        const Mt = m1 + m2;
        const M = SL.Mat;

        // 질량행렬 (평형점) ← 대칭
        const Meq = M.zeros(3, 3);
        Meq[0][0] = Mt * R * R;
        Meq[0][1] = Meq[1][0] = R * p1;    // cos(0) = 1
        Meq[0][2] = Meq[2][0] = R * p2;
        Meq[1][1] = m1 * ell1 * ell1 + I1 + m2 * L1 * L1;
        Meq[1][2] = Meq[2][1] = p3;        // cos(0) = 1
        Meq[2][2] = m2 * ell2 * ell2 + I2;

        const Minv = M.inv3x3sym(Meq);
        if (!Minv) return null;

        // 중력 선형화: G·q (∂f/∂q)
        const G = M.zeros(3, 3);
        G[0][0] = -Mt * g * R;    // ∂f₁/∂φ
        G[1][1] = p1 * g;         // ∂f₂/∂α
        G[2][2] = p2 * g;         // ∂f₃/∂θ

        // 감쇠 선형화: D·q̇ (∂f/∂q̇)
        const D = M.zeros(3, 3);
        D[0][0] = -b_phi;
        D[1][1] = -b_alpha;
        D[2][2] = -b_theta;

        // 입력 벡터: E (∂f/∂τ)
        const E = [[0], [-1], [1]];

        // q̈ = Minv·G·q + Minv·D·q̇ + Minv·E·τ
        const MinvG = M.mul(Minv, G, 3, 3, 3);
        const MinvD = M.mul(Minv, D, 3, 3, 3);
        const MinvE = M.mul(Minv, E, 3, 3, 1);

        // 상태공간: x = [φ,α,θ, φ̇,α̇,θ̇]
        // ẋ = Ac·x + Bc·u
        const Ac = M.zeros(6, 6);
        const Bc = M.zeros(6, 1);

        // 상단 3행: q̇ = [0₃ | I₃]·x
        for (let i = 0; i < 3; i++) Ac[i][i + 3] = 1;

        // 하단 3행: q̈ = MinvG·q + MinvD·q̇
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                Ac[i + 3][j] = MinvG[i][j];
                Ac[i + 3][j + 3] = MinvD[i][j];
            }
            Bc[i + 3][0] = MinvE[i][0];
        }

        return { Ac, Bc };
    }

    /**
     * 연속 → 이산 변환 (4차 행렬지수 근사)
     * A_d ≈ I + A·dt + A²·dt²/2 + A³·dt³/6 + A⁴·dt⁴/24
     */
    discretize(Ac, Bc, dt) {
        const M = SL.Mat;
        const n = 6;

        // A_d = expm(Ac * dt) ≈ I + Σ (Ac·dt)^k / k!
        const Adt = M.scale(Ac, dt, n, n);
        const Adt2 = M.mul(Adt, Adt, n, n, n);
        const Adt3 = M.mul(Adt2, Adt, n, n, n);
        const Adt4 = M.mul(Adt3, Adt, n, n, n);

        let Ad = M.eye(n);
        Ad = M.add(Ad, Adt, n, n);
        Ad = M.add(Ad, M.scale(Adt2, 0.5, n, n), n, n);
        Ad = M.add(Ad, M.scale(Adt3, 1 / 6, n, n), n, n);
        Ad = M.add(Ad, M.scale(Adt4, 1 / 24, n, n), n, n);

        // B_d ≈ (I·dt + Ac·dt²/2 + Ac²·dt³/6 + Ac³·dt⁴/24) · Bc
        let Bint = M.scale(M.eye(n), dt, n, n);
        Bint = M.add(Bint, M.scale(Adt, dt / 2, n, n), n, n);
        Bint = M.add(Bint, M.scale(Adt2, dt / 6, n, n), n, n);
        Bint = M.add(Bint, M.scale(Adt3, dt / 24, n, n), n, n);
        const Bd = M.mul(Bint, Bc, n, n, 1);

        return { Ad, Bd };
    }

    /**
     * 상태 증강 시스템 구성
     * z = [x, u(t-1), u(t-2), ..., u(t-N)]  (6+N 차원)
     * 
     * z(t+1) = A_aug · z(t) + B_aug · u(t)
     */
    buildAugmented(Ad, Bd, N) {
        const M = SL.Mat;
        const n = 6;
        const nAug = n + N;

        const Aaug = M.zeros(nAug, nAug);
        const Baug = M.zeros(nAug, 1);

        // 상단 6행: x(t+1) = Ad·x(t) + Bd·u(t-N)
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                Aaug[i][j] = Ad[i][j];
            }
            // u(t-N)은 증강 상태의 마지막 원소 (인덱스 6+N-1)
            if (N > 0) {
                Aaug[i][nAug - 1] = Bd[i][0];
            }
        }

        // 파이프라인 시프트 레지스터
        // z[6] = u(t-1) ← u(t) (새 입력)
        // z[7] = u(t-2) ← z[6] = u(t-1) (이전 상태)
        // ...
        // z[6+N-1] = u(t-N) ← z[6+N-2] = u(t-N+1)
        if (N > 0) {
            Baug[n][0] = 1;         // 새 입력 u(t)가 z[6]으로
        }
        for (let i = 1; i < N; i++) {
            Aaug[n + i][n + i - 1] = 1;  // 시프트
        }

        // N=0이면 지연 없음: Bd 직접 적용
        if (N === 0) {
            for (let i = 0; i < n; i++) {
                Baug[i][0] = Bd[i][0];
            }
        }

        return { Aaug, Baug, nAug };
    }

    /**
     * DARE (이산 대수 리카티 방정식) 반복 풀이
     * P = Aᵀ P A − Aᵀ P B (R + Bᵀ P B)⁻¹ Bᵀ P A + Q
     * 
     * 입력이 스칼라(1차원)이므로 역행렬은 스칼라 나눗셈으로 처리
     */
    solveDARE(A, B, Q, R_scalar, n, maxIter, tol) {
        const M = SL.Mat;
        maxIter = maxIter || 1000;
        tol = tol || 1e-8;

        // B를 1D 벡터로 변환 (첫 열)
        const b = new Float64Array(n);
        for (let i = 0; i < n; i++) b[i] = B[i][0];

        let P = M.copy(Q, n, n);
        const At = M.transpose(A, n, n);

        for (let iter = 0; iter < maxIter; iter++) {
            // PA = P × A
            const PA = M.mul(P, A, n, n, n);
            // AtPA = Aᵀ × PA
            const AtPA = M.mul(At, PA, n, n, n);

            // BtPB = bᵀ P b (스칼라)
            const Pb = M.mulVec(P, b, n, n);
            const BtPB = M.dot(b, Pb, n);

            // BtPA = bᵀ × PA (1×n 벡터)
            const BtPA = new Float64Array(n);
            for (let j = 0; j < n; j++) {
                let s = 0;
                for (let i = 0; i < n; i++) s += b[i] * PA[i][j];
                BtPA[j] = s;
            }

            // scale = 1 / (R + BtPB)
            const scale = 1.0 / (R_scalar + BtPB);

            // rankUpdate = BtPAᵀ × BtPA × scale  (n×n, rank-1)
            const Pnew = M.zeros(n, n);
            for (let i = 0; i < n; i++)
                for (let j = 0; j < n; j++)
                    Pnew[i][j] = AtPA[i][j] - BtPA[i] * BtPA[j] * scale + Q[i][j];

            // 대칭 강제
            for (let i = 0; i < n; i++)
                for (let j = i + 1; j < n; j++)
                    Pnew[i][j] = Pnew[j][i] = (Pnew[i][j] + Pnew[j][i]) * 0.5;

            // 수렴 확인
            const diff = M.normDiff(Pnew, P, n, n);
            P = Pnew;
            if (diff < tol) {
                console.log(`DARE converged in ${iter + 1} iterations (diff=${diff.toExponential(2)})`);
                break;
            }
        }
        return P;
    }

    /**
     * DARE 솔루션으로부터 최적 게인 K 계산
     * K = (R + Bᵀ P B)⁻¹ Bᵀ P A   (1×n 벡터)
     */
    computeGainFromP(A, B, P, R_scalar, n) {
        const M = SL.Mat;
        const b = new Float64Array(n);
        for (let i = 0; i < n; i++) b[i] = B[i][0];

        const Pb = M.mulVec(P, b, n, n);
        const BtPB = M.dot(b, Pb, n);
        const scale = 1.0 / (R_scalar + BtPB);

        const PA = M.mul(P, A, n, n, n);
        const K = new Float64Array(n);
        for (let j = 0; j < n; j++) {
            let s = 0;
            for (let i = 0; i < n; i++) s += b[i] * PA[i][j];
            K[j] = s * scale;
        }
        return K;
    }

    /**
     * V10: R 스케일을 반영하여 기본 LQR 게인 재계산
     * 
     * @param {Object} params - 파라미터 (공칭 또는 식별 파라미터)
     * @param {number} rScale - R 스케일 (1.0 = 기본, 100 = 보수적)
     */
    recomputeBaseGain(params, rScale) {
        this.rScale = rScale || 1.0;
        
        // R_scalar 자동 스케일링: 인체 스케일(M33≈8.5)에서 R=0.001이 적절했으므로
        // 다른 스케일에서는 R을 M33_ref/M33_actual 비율로 조정
        const m2 = params.m2 || 40, L2 = params.L2 || 0.8;
        const ell2 = L2 / 2;
        const M33_actual = m2 * ell2 * ell2 + m2 * L2 * L2 / 12;
        const M33_ref = 8.5;  // 인체 스케일 기준
        const baseR = 0.001; // V14 matching: Use fixed R_cost instead of auto-scaling
        const R_actual = baseR * this.rScale;
        
        console.log(`LQR R_scalar: baseR=${baseR.toFixed(4)} (M33=${M33_actual.toFixed(6)}), R_actual=${R_actual.toFixed(4)}`);

        console.time('BaseGain');

        // 1. 선형화
        const lin = this.linearize(params);
        if (!lin) {
            console.warn('recomputeBaseGain: linearize failed');
            return;
        }

        // 2. 이산화 (1ms dt)
        const dt = params.dt || 0.001;
        const { Ad, Bd } = this.discretize(lin.Ac, lin.Bc, dt);

        // 3. Q 행렬 (기본값)
        const M = SL.Mat;
        const Q = M.zeros(6, 6);
        const qDiag = [1, 50, 50, 0.1, 5, 5];
        for (let i = 0; i < 6; i++) Q[i][i] = qDiag[i];

        // 4. DARE 풀이
        const P = this.solveDARE(Ad, Bd, Q, R_actual, 6, 2000, 1e-9);

        // 5. 게인 계산
        const K = this.computeGainFromP(Ad, Bd, P, R_actual, 6);
        this.K_base = Array.from(K);

        console.timeEnd('BaseGain');
        console.log(`BaseGain (R×${rScale.toFixed(1)}): K=[${this.K_base.map(v => v.toFixed(1)).join(', ')}]`);

        // augmented 모드면 증강 게인도 재계산
        if (this.augmented && this.augValid) {
            this._lastAugParams = null; // 캐시 무효화
            this.computeAugmentedGains(params, this.totalDelayMs);
        }
    }

    /**
     * 상태증강 LQR 게인 계산 (지연 반영)
     */
    computeAugmentedGains(params, totalDelayMs) {
        const dtAugSec = this.dtAug / 1000;
        const N = Math.round(totalDelayMs / this.dtAug);

        // 캐시 확인 (같은 조건이면 재계산 불필요)
        const cacheKey = `${params.m1}_${params.m2}_${params.R}_${params.L1}_${params.L2}_${params.g}_${params.b_phi}_${params.b_alpha}_${params.b_theta}_${N}`;
        if (this._lastAugParams === cacheKey && this.augValid) return;
        this._lastAugParams = cacheKey;

        console.time('AugLQR');

        // 1. 선형화
        const lin = this.linearize(params);
        if (!lin) { this.augValid = false; return; }

        // 2. 이산화 (dtAug 간격)
        const { Ad, Bd } = this.discretize(lin.Ac, lin.Bc, dtAugSec);

        // 3. 상태 증강
        const { Aaug, Baug, nAug } = this.buildAugmented(Ad, Bd, N);

        // 4. Q, R 설정
        // 기본 Q: diag(1, 50, 50, 0.1, 5, 5)  증강 부분: 0
        const M = SL.Mat;
        const Qaug = M.zeros(nAug, nAug);
        const qDiag = [1, 50, 50, 0.1, 5, 5];
        for (let i = 0; i < 6; i++) Qaug[i][i] = qDiag[i];
        // 파이프라인의 u에 약간의 페널티 → 제어 입력 변동 억제
        for (let i = 6; i < nAug; i++) Qaug[i][i] = 0.0001;
        // R_scalar 자동 스케일링 (recomputeBaseGain과 동일한 방식)
        const m2 = params.m2 || 40, L2 = params.L2 || 0.8;
        const ell2 = L2 / 2;
        const M33_actual = m2 * ell2 * ell2 + m2 * L2 * L2 / 12;
        const M33_ref = 8.5;
        const R_scalar = 0.001; // V14 matching

        // 5. DARE 풀이
        const P = this.solveDARE(Aaug, Baug, Qaug, R_scalar, nAug, 2000, 1e-9);

        // 6. 게인 계산
        this.K_aug = this.computeGainFromP(Aaug, Baug, P, R_scalar, nAug);
        this.N_aug = N;
        this.augValid = true;

        console.timeEnd('AugLQR');
        console.log(`Augmented LQR: N=${N}, nAug=${nAug}, K_aug[0..5]=[${Array.from(this.K_aug).slice(0, 6).map(v => v.toFixed(1)).join(', ')}]`);
    }

    // ============================================================
    //  지연 설정
    // ============================================================

    setDelay(totalDelayMs, dtMs) {
        this.totalDelayMs = totalDelayMs;
        if (dtMs) this.dtMs = dtMs;

        // 지연 처리는 단순화를 위해 100% 액추에이터 지연으로 할당 (관측기 파이프라인 동기화 편의)
        this.actuatorDelaySteps = Math.round(this.totalDelayMs / this.dtMs);
        this._resizeBuffer('actuatorBuffer', this.actuatorDelaySteps);

        if (this.augmented) {
            this.uPipeline = new Array(this.actuatorDelaySteps).fill(0);
            this.computeAugmentedGains(this._lastAugParams_raw || SL.Params, this.totalDelayMs);
        }
    }

    setEstimationMode(mode) {
        this.estimationMode = mode; // 'ideal', 'kinematic', 'observer'
        this.reset();
    }

    setObserver(observer) {
        this.observer = observer;
    }

    setSensorNoise(noise) {
        this.sensorNoise = noise;
    }

    /** 센서 노이즈 적용된 IMU 데이터 생성 */
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

    _resizeBuffer(bufName, targetSize) {
        const buf = this[bufName];
        while (buf.length > targetSize + 1) buf.shift();
    }

    setAugmented(on, params) {
        this.augmented = on;
        if (params) this._lastAugParams_raw = params;
        if (on && params) {
            this.computeAugmentedGains(params, this.totalDelayMs);
        }
        this.reset();
    }

    // ============================================================
    //  기본 모드 버퍼 (센서/액추에이터 지연)
    // ============================================================

    _getKinematicEstimate(state, model) {
        // 정적 질량중심(CoM) 역기구학 추정 (V7 방식)
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

    _getDelayedTorque(immediateTau) {
        this.actuatorBuffer.push(immediateTau);
        if (this.actuatorDelaySteps === 0 || this.actuatorBuffer.length <= 1) {
            if (this.actuatorBuffer.length > this.actuatorDelaySteps + 1) this.actuatorBuffer.shift();
            return immediateTau;
        }
        if (this.actuatorBuffer.length > this.actuatorDelaySteps + 1) return this.actuatorBuffer.shift();
        return 0;
    }

    // ============================================================
    //  토크 계산
    // ============================================================

    compute(state, model, dt) {
        if (this.augmented && this.augValid && this.N_aug > 0) {
            return this._computeAugmented(state, model, dt);
        } else if (this.augmented && this.totalDelayMs === 0) {
            // 지연 0이면 기본 게인 사용
            return this._computeBasic(state, model, dt);
        }
        return this._computeBasic(state, model, dt);
    }

    /** 기본 모드 (발 위치 추정 적용) */
    _computeBasic(state, model, dt) {
        // V9: 센서 노이즈 적용된 IMU 데이터
        const imuData = this._getNoisyIMU(state);

        let estState;
        if (this.estimationMode === 'observer' && this.observer && this.observer.initialized) {
            // 칼만 필터에 노이즈 포함 IMU 전달 + 실제 인가된 토크
            estState = this.observer.step(this._appliedTau || 0, imuData);
        } else if (this.estimationMode === 'kinematic') {
            // 역기구학에도 노이즈 포함 측정값 사용
            const noisyState = Object.assign({}, state, imuData);
            estState = this._getKinematicEstimate(noisyState, model);
        } else {
            estState = state; // ideal: 노이즈 없이 실제값 직접 사용
        }

        const x = [estState.phi, estState.alpha, estState.theta,
        estState.phiDot, estState.alphaDot, estState.thetaDot];

        let tau = 0;
        for (let i = 0; i < 6; i++) tau -= this.K_base[i] * x[i];
        tau *= this.gainScale;

        if (this.Ki > 0) {
            const comError = model.getCoMError(state);
            this.integral += comError * dt;
            this.integral = Math.max(-this.maxIntegral, Math.min(this.maxIntegral, this.integral));
            tau -= this.Ki * this.integral;
        }

        this._lastTau = tau;
        const applied = this._getDelayedTorque(tau);
        this._appliedTau = applied;
        return applied;
    }

    /** 상태증강 모드 */
    _computeAugmented(state, model, dt) {
        // V9: 센서 노이즈 적용된 IMU 데이터
        const imuData = this._getNoisyIMU(state);

        let estState;
        if (this.estimationMode === 'observer' && this.observer && this.observer.initialized) {
            estState = this.observer.step(this._appliedTau || 0, imuData);
        } else if (this.estimationMode === 'kinematic') {
            const noisyState = Object.assign({}, state, imuData);
            estState = this._getKinematicEstimate(noisyState, model);
        } else {
            estState = state;
        }

        // 증강 상태 벡터 구성: [x, u_pipeline]
        const x = [estState.phi, estState.alpha, estState.theta,
        estState.phiDot, estState.alphaDot, estState.thetaDot];

        const N = this.actuatorDelaySteps;
        const nAug = 6 + this.N_aug;

        const zAug = new Float64Array(nAug);
        for (let i = 0; i < 6; i++) zAug[i] = x[i];

        // uPipeline에서 N_aug개 샘플 추출 (등간격)
        if (this.N_aug > 0 && this.uPipeline.length > 0) {
            const ratio = this.uPipeline.length / this.N_aug;
            for (let i = 0; i < this.N_aug; i++) {
                const idx = Math.min(Math.floor(i * ratio), this.uPipeline.length - 1);
                zAug[6 + i] = this.uPipeline[idx];
            }
        }

        // τ = −K_aug · z_aug
        let tau = 0;
        for (let i = 0; i < nAug; i++) tau -= this.K_aug[i] * zAug[i];
        tau *= this.gainScale;

        // 적분 보정
        if (this.Ki > 0) {
            const comError = model.getCoMError(state);
            this.integral += comError * dt;
            this.integral = Math.max(-this.maxIntegral, Math.min(this.maxIntegral, this.integral));
            tau -= this.Ki * this.integral;
        }

        // 파이프라인에 새 토크 추가, 지연된 토크 반환
        this.uPipeline.push(tau);
        let appliedTau = 0;
        if (this.uPipeline.length > N) {
            appliedTau = this.uPipeline.shift();
        }

        this._appliedTau = appliedTau;
        return appliedTau;
    }

    // ============================================================
    //  기타
    // ============================================================

    setGainScale(scale) { this.gainScale = scale; }
    setKi(Ki) { this.Ki = Ki; }

    reset() {
        this.integral = 0;
        this._lastTau = 0;
        this._appliedTau = 0;
        this.stateBuffer = [];
        this.actuatorBuffer = [];
        this.uPipeline = new Array(this.actuatorDelaySteps).fill(0);
    }

    getDelayInfo() {
        const modeLabels = {
            'ideal': '🎯 완벽한 센싱',
            'kinematic': '📐 CoM 역기구학 (V7)',
            'observer': '🔬 칼만 필터 옵저버 (V8)'
        };
        return {
            totalMs: this.totalDelayMs,
            sensorMs: modeLabels[this.estimationMode] || '알 수 없음',
            actuatorMs: this.totalDelayMs,
            estimationMode: this.estimationMode,
            augmented: this.augmented,
            augN: this.N_aug,
            augValid: this.augValid
        };
    }
};

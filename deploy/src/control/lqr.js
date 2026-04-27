/**
 * Deploy Toolkit — LQR Controller
 * 행렬유틸 + 선형화 + 이산화 + DARE + 게인 계산
 */
var DT = DT || {};

// ============================================================
//  행렬 유틸리티
// ============================================================
DT.Mat = {
    zeros(n, m) { const A = []; for (let i = 0; i < n; i++) A[i] = new Float64Array(m); return A; },
    eye(n) { const I = this.zeros(n, n); for (let i = 0; i < n; i++) I[i][i] = 1; return I; },
    mul(A, B, n, m, p) { const C = this.zeros(n, p); for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) { let s = 0; for (let k = 0; k < m; k++) s += A[i][k] * B[k][j]; C[i][j] = s; } return C; },
    add(A, B, n, m) { const C = this.zeros(n, m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) C[i][j] = A[i][j] + B[i][j]; return C; },
    sub(A, B, n, m) { const C = this.zeros(n, m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) C[i][j] = A[i][j] - B[i][j]; return C; },
    scale(A, s, n, m) { const C = this.zeros(n, m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) C[i][j] = A[i][j] * s; return C; },
    transpose(A, n, m) { const T = this.zeros(m, n); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j]; return T; },
    dot(a, b, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; },
    mulVec(A, v, n, m) { const w = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < m; j++) s += A[i][j] * v[j]; w[i] = s; } return w; },
    copy(A, n, m) { const C = this.zeros(n, m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) C[i][j] = A[i][j]; return C; },
    normDiff(A, B, n, m) { let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const d = A[i][j] - B[i][j]; s += d * d; } return Math.sqrt(s); },
    inv3x3sym(M) {
        const [a, b, c, d, e, f] = [M[0][0], M[0][1], M[0][2], M[1][1], M[1][2], M[2][2]];
        const det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
        if (Math.abs(det) < 1e-15) return null;
        const id = 1.0 / det, R = this.zeros(3, 3);
        R[0][0] = (d * f - e * e) * id; R[0][1] = R[1][0] = -(b * f - c * e) * id;
        R[0][2] = R[2][0] = (b * e - c * d) * id; R[1][1] = (a * f - c * c) * id;
        R[1][2] = R[2][1] = -(a * e - b * c) * id; R[2][2] = (a * d - b * b) * id;
        return R;
    }
};

// ============================================================
//  선형화 (평형점 φ=α=θ=0 근처)
// ============================================================
DT.linearize = function(p) {
    const M = DT.Mat;
    const { m1, m2, R, g, L1, L2, ell1, ell2, I1, I2, b_phi, b_alpha, b_theta } = p;
    const pv1 = m1 * ell1 + m2 * L1;
    const pv2 = m2 * ell2;
    const pv3 = m2 * L1 * ell2;
    const Mt = m1 + m2;

    const Meq = [
        [Mt * R * R, R * pv1, R * pv2],
        [R * pv1, m1 * ell1 * ell1 + I1 + m2 * L1 * L1, pv3],
        [R * pv2, pv3, m2 * ell2 * ell2 + I2]
    ];
    const Minv = M.inv3x3sym(Meq);
    if (!Minv) return null;

    const G = [[-Mt * g * R, 0, 0], [0, pv1 * g, 0], [0, 0, pv2 * g]];
    const D = [[-b_phi, 0, 0], [0, -b_alpha, 0], [0, 0, -b_theta]];
    const E = [[0], [-1], [1]];

    const MG = M.mul(Minv, G, 3, 3, 3);
    const MD = M.mul(Minv, D, 3, 3, 3);
    const ME = M.mul(Minv, E, 3, 3, 1);

    const Ac = M.zeros(6, 6), Bc = M.zeros(6, 1);
    for (let i = 0; i < 3; i++) Ac[i][i + 3] = 1;
    for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
            Ac[i + 3][j] = MG[i][j];
            Ac[i + 3][j + 3] = MD[i][j];
        }
        Bc[i + 3][0] = ME[i][0];
    }
    return { Ac, Bc };
};

// ============================================================
//  이산화 (행렬 지수 근사, 4차 테일러)
// ============================================================
DT.discretize = function(Ac, Bc, dt) {
    const M = DT.Mat, n = 6;
    const Adt = M.scale(Ac, dt, n, n);
    const Adt2 = M.mul(Adt, Adt, n, n, n);
    const Adt3 = M.mul(Adt2, Adt, n, n, n);
    const Adt4 = M.mul(Adt3, Adt, n, n, n);

    let Ad = M.eye(n);
    Ad = M.add(Ad, Adt, n, n);
    Ad = M.add(Ad, M.scale(Adt2, 0.5, n, n), n, n);
    Ad = M.add(Ad, M.scale(Adt3, 1 / 6, n, n), n, n);
    Ad = M.add(Ad, M.scale(Adt4, 1 / 24, n, n), n, n);

    let Bint = M.scale(M.eye(n), dt, n, n);
    Bint = M.add(Bint, M.scale(Adt, dt / 2, n, n), n, n);
    Bint = M.add(Bint, M.scale(Adt2, dt / 6, n, n), n, n);
    Bint = M.add(Bint, M.scale(Adt3, dt / 24, n, n), n, n);
    const Bd = M.mul(Bint, Bc, n, n, 1);

    return { Ad, Bd };
};

// ============================================================
//  DARE (이산 대수 리카티 방정식) — 단일 입력
// ============================================================
DT.solveDARE = function(A, B, Q, R_scalar, n) {
    const M = DT.Mat;
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) b[i] = B[i][0];

    let P = M.copy(Q, n, n);
    const At = M.transpose(A, n, n);

    for (let iter = 0; iter < 3000; iter++) {
        const PA = M.mul(P, A, n, n, n);
        const AtPA = M.mul(At, PA, n, n, n);
        const Pb = M.mulVec(P, b, n, n);
        const BtPB = M.dot(b, Pb, n);
        const BtPA = new Float64Array(n);
        for (let j = 0; j < n; j++) {
            let s = 0;
            for (let i = 0; i < n; i++) s += b[i] * PA[i][j];
            BtPA[j] = s;
        }
        const sc = 1.0 / (R_scalar + BtPB);
        const Pn = M.zeros(n, n);
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                Pn[i][j] = AtPA[i][j] - BtPA[i] * BtPA[j] * sc + Q[i][j];
        // 대칭 강제
        for (let i = 0; i < n; i++)
            for (let j = i + 1; j < n; j++)
                Pn[i][j] = Pn[j][i] = (Pn[i][j] + Pn[j][i]) * 0.5;

        const diff = M.normDiff(Pn, P, n, n);
        P = Pn;
        if (diff < 1e-9) break;
    }
    return P;
};

// ============================================================
//  게인 계산: K = (R + BᵀPB)⁻¹ BᵀPA
// ============================================================
DT.computeGain = function(A, B, P, R_scalar, n) {
    const M = DT.Mat;
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) b[i] = B[i][0];
    const Pb = M.mulVec(P, b, n, n);
    const BtPB = M.dot(b, Pb, n);
    const sc = 1.0 / (R_scalar + BtPB);
    const PA = M.mul(P, A, n, n, n);
    const K = new Float64Array(n);
    for (let j = 0; j < n; j++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += b[i] * PA[i][j];
        K[j] = s * sc;
    }
    return K;
};

// ============================================================
//  전체 파이프라인: 파라미터 → LQR 게인
// ============================================================
DT.computeLQRGain = function(params, Q_diag, R_lqr) {
    const lin = DT.linearize(params);
    if (!lin) return null;
    const disc = DT.discretize(lin.Ac, lin.Bc, params.dt || 0.001);
    const n = 6;
    const Q = DT.Mat.zeros(n, n);
    Q_diag.forEach((v, i) => Q[i][i] = v);
    const P = DT.solveDARE(disc.Ad, disc.Bd, Q, R_lqr, n);
    const K = DT.computeGain(disc.Ad, disc.Bd, P, R_lqr, n);
    return { K, P, Ad: disc.Ad, Bd: disc.Bd, Ac: lin.Ac, Bc: lin.Bc };
};

// ============================================================
//  미니 시뮬: 게인으로 시뮬 돌려서 토크/각도 이력 반환
// ============================================================
DT.runMiniSim = function(params, K, duration, theta0, perturbs) {
    const model = new DT.Model(params);
    model.state.theta = theta0 || 0.15;
    const dt = params.dt || 0.001;
    const totalSteps = Math.round(duration / dt);
    const pf = (perturbs || []).map(() => false);

    let maxTau = 0, maxHipVel = 0, maxHipAng = 0;
    const history = [];
    const sampleInterval = Math.max(1, Math.round(totalSteps / 200)); // ~200 samples

    for (let s = 0; s < totalSteps; s++) {
        const st = model.state;
        // 교란
        if (perturbs) {
            for (let pi = 0; pi < perturbs.length; pi++) {
                if (st.time >= perturbs[pi].t && !pf[pi]) {
                    st.thetaDot += perturbs[pi].v;
                    pf[pi] = true;
                }
            }
        }
        // LQR
        const x = [st.phi, st.alpha, st.theta, st.phiDot, st.alphaDot, st.thetaDot];
        let tau = 0;
        for (let i = 0; i < 6; i++) tau -= K[i] * x[i];
        tau = Math.max(-params.tauMax, Math.min(params.tauMax, tau));

        const at = Math.abs(tau);
        if (at > maxTau) maxTau = at;
        const hv = Math.abs(st.thetaDot - st.alphaDot);
        if (hv > maxHipVel) maxHipVel = hv;
        const ha = Math.abs(st.theta - st.alpha);
        if (ha > maxHipAng) maxHipAng = ha;

        if (s % sampleInterval === 0) {
            history.push({ t: st.time, tau, theta: st.theta, alpha: st.alpha, phi: st.phi, hipVel: hv, hipAng: ha });
        }

        model.state = model.rk4Step(st, tau, dt);
        if (Math.abs(model.state.phi) > params.phiMax) {
            model.state.phi = Math.sign(model.state.phi) * params.phiMax;
            model.state.phiDot *= -0.3;
        }
        if (!isFinite(model.state.phi) || !isFinite(model.state.theta)) {
            return { success: false, diverged: true, maxTau, maxHipVel, maxHipAng, history, step: s };
        }
    }

    const st = model.state;
    const balanced = Math.abs(st.theta) < 0.05 && Math.abs(st.alpha) < 0.05;
    return { success: balanced, diverged: false, maxTau, maxHipVel, maxHipAng, history, finalState: st, saturated: Math.abs(maxTau - params.tauMax) < 0.001 };
};

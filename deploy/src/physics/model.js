/**
 * Deploy Toolkit — 3-DOF Physics Model
 * V10에서 이식, I와 ℓ을 독립 파라미터로 분리
 */
var DT = DT || {};

DT.Model = class {
    constructor(params) {
        this.params = params;
        this.reset();
    }

    reset(params) {
        if (params) this.params = params;
        const p = this.params;
        this.state = {
            phi: p.phi0 || 0, alpha: p.alpha0 || 0, theta: p.theta0 || 0.15,
            phiDot: 0, alphaDot: 0, thetaDot: 0, time: 0
        };
        this.tau = 0;
    }

    solve3x3(M, v) {
        const [a, b, c, d, e, f] = [M[0][0], M[0][1], M[0][2], M[1][1], M[1][2], M[2][2]];
        const det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
        if (Math.abs(det) < 1e-12) return [0, 0, 0];
        const id = 1.0 / det;
        const A11 = d * f - e * e, A12 = -(b * f - c * e), A13 = b * e - c * d;
        const A22 = a * f - c * c, A23 = -(a * e - b * c), A33 = a * d - b * b;
        return [
            (A11 * v[0] + A12 * v[1] + A13 * v[2]) * id,
            (A12 * v[0] + A22 * v[1] + A23 * v[2]) * id,
            (A13 * v[0] + A23 * v[1] + A33 * v[2]) * id
        ];
    }

    computeDerivatives(state, tau) {
        const p = this.params;
        const { phi, alpha, theta, phiDot, alphaDot, thetaDot } = state;
        const { R, g, p1, p2, p3, Mtotal } = p;

        const sinPA = Math.sin(phi + alpha), cosPA = Math.cos(phi + alpha);
        const sinPT = Math.sin(phi + theta), cosPT = Math.cos(phi + theta);
        const sinAT = Math.sin(alpha - theta), cosAT = Math.cos(alpha - theta);

        const M11 = Mtotal * R * R;
        const M12 = R * p1 * cosPA;
        const M13 = R * p2 * cosPT;
        const M22 = p.I1_total; // m₁ℓ₁² + I₁ + m₂L₁²
        const M23 = p3 * cosAT;
        const M33 = p.I2_total; // m₂ℓ₂² + I₂

        const M = [[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]];

        const f1 = R * p1 * sinPA * alphaDot * alphaDot
            + R * p2 * sinPT * thetaDot * thetaDot
            - Mtotal * g * R * Math.sin(phi) - p.b_phi * phiDot;
        const f2 = R * p1 * sinPA * phiDot * phiDot
            - p3 * sinAT * thetaDot * thetaDot
            + p1 * g * Math.sin(alpha) - tau - p.b_alpha * alphaDot;
        const f3 = R * p2 * sinPT * phiDot * phiDot
            + p3 * sinAT * alphaDot * alphaDot
            + p2 * g * Math.sin(theta) + tau - p.b_theta * thetaDot;

        const accel = this.solve3x3(M, [f1, f2, f3]);
        return [phiDot, alphaDot, thetaDot, accel[0], accel[1], accel[2]];
    }

    rk4Step(state, tau, dt) {
        const s = [state.phi, state.alpha, state.theta, state.phiDot, state.alphaDot, state.thetaDot];
        const toObj = (a) => ({ phi: a[0], alpha: a[1], theta: a[2], phiDot: a[3], alphaDot: a[4], thetaDot: a[5] });
        const k1 = this.computeDerivatives(toObj(s), tau);
        const s2 = s.map((v, i) => v + 0.5 * dt * k1[i]);
        const k2 = this.computeDerivatives(toObj(s2), tau);
        const s3 = s.map((v, i) => v + 0.5 * dt * k2[i]);
        const k3 = this.computeDerivatives(toObj(s3), tau);
        const s4 = s.map((v, i) => v + dt * k3[i]);
        const k4 = this.computeDerivatives(toObj(s4), tau);
        const ns = s.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
        return { phi: ns[0], alpha: ns[1], theta: ns[2], phiDot: ns[3], alphaDot: ns[4], thetaDot: ns[5], time: state.time + dt };
    }
};

/**
 * 파라미터 객체 생성 (독립 I, ℓ 지원)
 * opts: { m1, m2, L1, L2, ell1, ell2, I1, I2, R, g, b_phi, b_alpha, b_theta, tauMax }
 * ell1, I1 등이 없으면 균일 막대 기본값 사용
 */
DT.makeParams = function(opts) {
    const m1 = opts.m1, m2 = opts.m2;
    const L1 = opts.L1, L2 = opts.L2;
    const ell1 = opts.ell1 !== undefined ? opts.ell1 : L1 / 2;
    const ell2 = opts.ell2 !== undefined ? opts.ell2 : L2 / 2;
    const I1 = opts.I1 !== undefined ? opts.I1 : m1 * L1 * L1 / 12;
    const I2 = opts.I2 !== undefined ? opts.I2 : m2 * L2 * L2 / 12;
    const R = opts.R || 0.35;
    const g = opts.g || 9.81;

    const p1 = m1 * ell1 + m2 * L1;
    const p2 = m2 * ell2;
    const p3 = m2 * L1 * ell2;
    const Mtotal = m1 + m2;

    return {
        m1, m2, L1, L2, ell1, ell2, I1, I2, R, g,
        p1, p2, p3, Mtotal,
        I1_total: m1 * ell1 * ell1 + I1 + m2 * L1 * L1,
        I2_total: m2 * ell2 * ell2 + I2,
        b_phi: opts.b_phi || 0.01, b_alpha: opts.b_alpha || 0.005, b_theta: opts.b_theta || 0.003,
        tauMax: opts.tauMax || 1.47,
        dt: opts.dt || 0.001,
        phi0: opts.phi0 || 0, alpha0: opts.alpha0 || 0, theta0: opts.theta0 || 0.15,
        phiMax: 80 * Math.PI / 180
    };
};

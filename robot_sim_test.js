/**
 * 실제 20cm 로봇 파라미터로 시뮬레이션 돌려보기
 * 
 * 목적: 스케일링이 아닌 실제 물리 파라미터로 시뮬레이션을 돌려
 *       필요한 토크(Nm, kgf·cm)와 힙 각속도(deg/s)를 직접 측정
 */

// ========== 파라미터 ==========
const Params = {
    m1: 0.060,       // 60g
    m2: 0.120,       // 120g
    L1: 0.20,        // 20cm
    L2: 0.20,        // 20cm
    R: 0.30,         // 30cm
    g: 9.81,
    b_phi: 0.005,
    b_alpha: 0.003,
    b_theta: 0.002,
    dt: 0.001,
    tauMax: 10,
    phi0: 0, alpha0: 0, theta0: 0.15,
    phiMax: 80 * Math.PI / 180,
    get ell1() { return this.L1 / 2; },
    get ell2() { return this.L2 / 2; },
    get I1() { return this.m1 * this.L1 * this.L1 / 12; },
    get I2() { return this.m2 * this.L2 * this.L2 / 12; },
    get p1() { return this.m1 * this.ell1 + this.m2 * this.L1; },
    get p2() { return this.m2 * this.ell2; },
    get p3() { return this.m2 * this.L1 * this.ell2; },
    get Mtotal() { return this.m1 + this.m2; },
    get M11() { return this.Mtotal * this.R * this.R; },
    get M22() { return this.m1 * this.ell1 * this.ell1 + this.I1 + this.m2 * this.L1 * this.L1; },
    get M33() { return this.m2 * this.ell2 * this.ell2 + this.I2; },
};

// ========== 행렬 유틸 ==========
function zeros(n, m) { return Array.from({ length: n }, () => new Float64Array(m)); }
function eye(n) { const I = zeros(n, n); for (let i = 0; i < n; i++) I[i][i] = 1; return I; }
function matMul(A, B, n, m, p) { const C = zeros(n, p); for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) { let s = 0; for (let k = 0; k < m; k++) s += A[i][k] * B[k][j]; C[i][j] = s; } return C; }
function matAdd(A, B, n, m) { const C = zeros(n, m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) C[i][j] = A[i][j] + B[i][j]; return C; }
function matScale(A, s, n, m) { const C = zeros(n, m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) C[i][j] = A[i][j] * s; return C; }
function matT(A, n, m) { const T = zeros(m, n); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j]; return T; }
function matCopy(A, n, m) { const C = zeros(n, m); for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) C[i][j] = A[i][j]; return C; }
function matNormDiff(A, B, n, m) { let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) { const d = A[i][j] - B[i][j]; s += d * d; } return Math.sqrt(s); }
function dot(a, b, n) { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; }
function mulVec(A, v, n, m) { const w = new Float64Array(n); for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < m; j++) s += A[i][j] * v[j]; w[i] = s; } return w; }

function inv3x3sym(M) {
    const [a, b, c, d, e, f] = [M[0][0], M[0][1], M[0][2], M[1][1], M[1][2], M[2][2]];
    const det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
    if (Math.abs(det) < 1e-15) return null;
    const id = 1.0 / det;
    const R = zeros(3, 3);
    R[0][0] = (d * f - e * e) * id; R[0][1] = R[1][0] = -(b * f - c * e) * id;
    R[0][2] = R[2][0] = (b * e - c * d) * id; R[1][1] = (a * f - c * c) * id;
    R[1][2] = R[2][1] = -(a * e - b * c) * id; R[2][2] = (a * d - b * b) * id;
    return R;
}

function solve3x3(M, v) {
    const [a, b, c, d, e, f] = [M[0][0], M[0][1], M[0][2], M[1][1], M[1][2], M[2][2]];
    const det = a * (d * f - e * e) - b * (b * f - e * c) + c * (b * e - d * c);
    if (Math.abs(det) < 1e-15) return [0, 0, 0];
    const id = 1.0 / det;
    const A11 = d * f - e * e, A12 = -(b * f - c * e), A13 = b * e - c * d;
    const A22 = a * f - c * c, A23 = -(a * e - b * c), A33 = a * d - b * b;
    return [(A11*v[0]+A12*v[1]+A13*v[2])*id, (A12*v[0]+A22*v[1]+A23*v[2])*id, (A13*v[0]+A23*v[1]+A33*v[2])*id];
}

// ========== 물리 ==========
function computeDerivatives(state, tau, p) {
    const { phi, alpha, theta, phiDot, alphaDot, thetaDot } = state;
    const R = p.R, g = p.g, p1 = p.p1, p2 = p.p2, p3 = p.p3, Mt = p.Mtotal;
    const sinPA = Math.sin(phi+alpha), cosPA = Math.cos(phi+alpha);
    const sinPT = Math.sin(phi+theta), cosPT = Math.cos(phi+theta);
    const sinAT = Math.sin(alpha-theta), cosAT = Math.cos(alpha-theta);
    const M = [[p.M11, R*p1*cosPA, R*p2*cosPT],[R*p1*cosPA, p.M22, p3*cosAT],[R*p2*cosPT, p3*cosAT, p.M33]];
    const f1 = R*p1*sinPA*alphaDot*alphaDot + R*p2*sinPT*thetaDot*thetaDot - Mt*g*R*Math.sin(phi) - p.b_phi*phiDot;
    const f2 = R*p1*sinPA*phiDot*phiDot - p3*sinAT*thetaDot*thetaDot + p1*g*Math.sin(alpha) - tau - p.b_alpha*alphaDot;
    const f3 = R*p2*sinPT*phiDot*phiDot + p3*sinAT*alphaDot*alphaDot + p2*g*Math.sin(theta) + tau - p.b_theta*thetaDot;
    const accel = solve3x3(M, [f1, f2, f3]);
    return [phiDot, alphaDot, thetaDot, accel[0], accel[1], accel[2]];
}

function rk4Step(state, tau, dt, p) {
    const s = [state.phi, state.alpha, state.theta, state.phiDot, state.alphaDot, state.thetaDot];
    const toObj = (a) => ({phi:a[0],alpha:a[1],theta:a[2],phiDot:a[3],alphaDot:a[4],thetaDot:a[5]});
    const k1 = computeDerivatives(toObj(s), tau, p);
    const s2 = s.map((v,i)=>v+0.5*dt*k1[i]); const k2 = computeDerivatives(toObj(s2), tau, p);
    const s3 = s.map((v,i)=>v+0.5*dt*k2[i]); const k3 = computeDerivatives(toObj(s3), tau, p);
    const s4 = s.map((v,i)=>v+dt*k3[i]); const k4 = computeDerivatives(toObj(s4), tau, p);
    const ns = s.map((v,i)=>v+(dt/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i]));
    return {phi:ns[0],alpha:ns[1],theta:ns[2],phiDot:ns[3],alphaDot:ns[4],thetaDot:ns[5],time:state.time+dt};
}

// ========== 선형화 + LQR ==========
function linearize(p) {
    const { m1, m2, R, g, L1, L2, b_phi, b_alpha, b_theta } = p;
    const e1 = L1/2, e2 = L2/2, I1 = m1*L1*L1/12, I2 = m2*L2*L2/12;
    const pv1 = m1*e1+m2*L1, pv2 = m2*e2, pv3 = m2*L1*e2, Mt = m1+m2;
    const Meq = [[Mt*R*R,R*pv1,R*pv2],[R*pv1,m1*e1*e1+I1+m2*L1*L1,pv3],[R*pv2,pv3,m2*e2*e2+I2]];
    const Minv = inv3x3sym(Meq); if (!Minv) return null;
    const G = [[-Mt*g*R,0,0],[0,pv1*g,0],[0,0,pv2*g]];
    const D = [[-b_phi,0,0],[0,-b_alpha,0],[0,0,-b_theta]];
    const E = [[0],[-1],[1]];
    const MG = matMul(Minv,G,3,3,3), MD = matMul(Minv,D,3,3,3), ME = matMul(Minv,E,3,3,1);
    const Ac = zeros(6,6), Bc = zeros(6,1);
    for (let i=0;i<3;i++) Ac[i][i+3]=1;
    for (let i=0;i<3;i++) { for (let j=0;j<3;j++) { Ac[i+3][j]=MG[i][j]; Ac[i+3][j+3]=MD[i][j]; } Bc[i+3][0]=ME[i][0]; }
    return {Ac,Bc};
}

function discretize(Ac, Bc, dt) {
    const n=6, Adt=matScale(Ac,dt,n,n), Adt2=matMul(Adt,Adt,n,n,n), Adt3=matMul(Adt2,Adt,n,n,n), Adt4=matMul(Adt3,Adt,n,n,n);
    let Ad=eye(n); Ad=matAdd(Ad,Adt,n,n); Ad=matAdd(Ad,matScale(Adt2,0.5,n,n),n,n); Ad=matAdd(Ad,matScale(Adt3,1/6,n,n),n,n); Ad=matAdd(Ad,matScale(Adt4,1/24,n,n),n,n);
    let Bi=matScale(eye(n),dt,n,n); Bi=matAdd(Bi,matScale(Adt,dt/2,n,n),n,n); Bi=matAdd(Bi,matScale(Adt2,dt/6,n,n),n,n); Bi=matAdd(Bi,matScale(Adt3,dt/24,n,n),n,n);
    return {Ad, Bd:matMul(Bi,Bc,n,n,1)};
}

function solveDARE(A,B,Q,Rs,n) {
    const b=new Float64Array(n); for(let i=0;i<n;i++) b[i]=B[i][0];
    let P=matCopy(Q,n,n); const At=matT(A,n,n);
    for(let iter=0;iter<2000;iter++){
        const PA=matMul(P,A,n,n,n), AtPA=matMul(At,PA,n,n,n);
        const Pb=mulVec(P,b,n,n), BtPB=dot(b,Pb,n);
        const BtPA=new Float64Array(n);
        for(let j=0;j<n;j++){let s=0;for(let i=0;i<n;i++)s+=b[i]*PA[i][j];BtPA[j]=s;}
        const sc=1/(Rs+BtPB);
        const Pn=zeros(n,n);
        for(let i=0;i<n;i++)for(let j=0;j<n;j++)Pn[i][j]=AtPA[i][j]-BtPA[i]*BtPA[j]*sc+Q[i][j];
        for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)Pn[i][j]=Pn[j][i]=(Pn[i][j]+Pn[j][i])*0.5;
        if(matNormDiff(Pn,P,n,n)<1e-9){console.log(`DARE converged at iter ${iter+1}`);P=Pn;break;}
        P=Pn;
    }
    return P;
}

function computeGain(A,B,P,Rs,n) {
    const b=new Float64Array(n); for(let i=0;i<n;i++) b[i]=B[i][0];
    const Pb=mulVec(P,b,n,n), sc=1/(Rs+dot(b,Pb,n)), PA=matMul(P,A,n,n,n);
    const K=new Float64Array(n);
    for(let j=0;j<n;j++){let s=0;for(let i=0;i<n;i++)s+=b[i]*PA[i][j];K[j]=s*sc;}
    return K;
}

// ================================================================
//   메인
// ================================================================
console.log('======================================================================');
console.log('  슬랙라인 로봇 하드웨어 검증 시뮬레이션');
console.log('  실제 파라미터: m1=60g, m2=120g, L1=L2=20cm, R=30cm');
console.log('======================================================================\n');

console.log('[파라미터]');
console.log(`  m1=${Params.m1}kg (${Params.m1*1000}g), m2=${Params.m2}kg (${Params.m2*1000}g), 총=${Params.Mtotal}kg`);
console.log(`  L1=${Params.L1}m, L2=${Params.L2}m, R=${Params.R}m`);
console.log(`  I1=${Params.I1.toExponential(4)}, I2=${Params.I2.toExponential(4)}\n`);

// LQR
const lin = linearize(Params);
if(!lin){console.error('선형화 실패!');process.exit(1);}
const disc = discretize(lin.Ac, lin.Bc, Params.dt);
const Q = zeros(6,6); [1,50,50,0.1,5,5].forEach((v,i)=>Q[i][i]=v);
const P = solveDARE(disc.Ad, disc.Bd, Q, 0.001, 6);
const K = computeGain(disc.Ad, disc.Bd, P, 0.001, 6);
console.log(`[LQR 게인] K = [${Array.from(K).map(v=>v.toFixed(4)).join(', ')}]\n`);

// 시뮬레이션
let state = {phi:0,alpha:0,theta:0.15,phiDot:0,alphaDot:0,thetaDot:0,time:0};
let maxAbsTau=0, maxHipAngVel=0, maxAlphaDot=0, maxThetaDot=0, maxPhiDot=0, maxHipAngle=0;
let pertFlags = [false,false,false];
const pertTimes = [1.0, 2.0, 3.0];
const pertValues = [2.0, 3.0, -4.0];
let tauHist = [], hipVelHist = [];

for(let step=0; step < 5000; step++){
    // 교란
    for(let pi=0;pi<3;pi++){
        if(state.time>=pertTimes[pi] && !pertFlags[pi]){
            state.thetaDot += pertValues[pi];
            pertFlags[pi]=true;
            console.log(`[t=${state.time.toFixed(2)}s] 교란 #${pi+1}: thetaDot += ${pertValues[pi]} rad/s`);
        }
    }
    // LQR
    const x = [state.phi,state.alpha,state.theta,state.phiDot,state.alphaDot,state.thetaDot];
    let tau=0; for(let i=0;i<6;i++) tau -= K[i]*x[i];
    tau = Math.max(-Params.tauMax, Math.min(Params.tauMax, tau));

    const absTau=Math.abs(tau);
    if(absTau>maxAbsTau) maxAbsTau=absTau;
    const hipAV=Math.abs(state.thetaDot-state.alphaDot);
    if(hipAV>maxHipAngVel) maxHipAngVel=hipAV;
    if(Math.abs(state.alphaDot)>maxAlphaDot) maxAlphaDot=Math.abs(state.alphaDot);
    if(Math.abs(state.thetaDot)>maxThetaDot) maxThetaDot=Math.abs(state.thetaDot);
    if(Math.abs(state.phiDot)>maxPhiDot) maxPhiDot=Math.abs(state.phiDot);
    const hipA=Math.abs(state.theta-state.alpha);
    if(hipA>maxHipAngle) maxHipAngle=hipA;

    if(step%100===0){tauHist.push({t:state.time,tau});hipVelHist.push({t:state.time,vel:hipAV});}

    state = rk4Step(state, tau, Params.dt, Params);
    if(Math.abs(state.phi)>Params.phiMax){state.phi=Math.sign(state.phi)*Params.phiMax;state.phiDot*=-0.3;}
}

console.log('\n======================================================================');
console.log('  시뮬레이션 결과 (5초, 초기 기울기 8.6° + 교란 3회)');
console.log('======================================================================\n');

const tKgf = maxAbsTau * 100 / 9.81;
const tMnm = maxAbsTau * 1000;
console.log('[토크 요구량]');
console.log(`  최대 |τ| = ${maxAbsTau.toFixed(6)} N·m = ${tKgf.toFixed(4)} kgf·cm = ${tMnm.toFixed(3)} mN·m`);

const reqSpeed = 60 / (maxHipAngVel * 180 / Math.PI);
console.log('\n[속도 요구량]');
console.log(`  최대 힙 각속도 |θ̇-α̇| = ${maxHipAngVel.toFixed(2)} rad/s = ${(maxHipAngVel*180/Math.PI).toFixed(1)} °/s`);
console.log(`  → 필요 서보 속도: ${reqSpeed.toFixed(4)} sec/60°`);
console.log(`  최대 |α̇| = ${(maxAlphaDot*180/Math.PI).toFixed(1)} °/s`);
console.log(`  최대 |θ̇| = ${(maxThetaDot*180/Math.PI).toFixed(1)} °/s`);
console.log(`  최대 힙각 |θ-α| = ${(maxHipAngle*180/Math.PI).toFixed(1)} °`);

console.log('\n[서보 비교]');
console.log(`  필요: 토크 ≥ ${(tKgf*2).toFixed(3)} kgf·cm (×2 안전여유), 속도 ≤ ${(reqSpeed*0.7).toFixed(4)} sec/60°`);
const servos = [
    {name:'SG90',t:1.8,s:0.12,w:9},{name:'MG90S',t:2.2,s:0.10,w:14},
    {name:'MG92B',t:3.5,s:0.08,w:14},{name:'MG996R',t:13,s:0.15,w:55},
    {name:'DS3218',t:21,s:0.07,w:60},{name:'STS3215',t:15,s:0.05,w:67}
];
servos.forEach(sv=>{
    const tOK=sv.t>=tKgf*2, sOK=sv.s<=reqSpeed*0.7;
    console.log(`  ${sv.name.padEnd(8)} ${String(sv.t).padStart(5)}kgf·cm ${sv.s}s/60° ${String(sv.w).padStart(2)}g  ${tOK&&sOK?'✅ 적합':(!tOK?'❌ 토크부족':'❌ 속도부족')}`);
});

console.log('\n[토크 시계열 (100ms 간격, mN·m)]');
console.log('  시간(s)  토크(mN·m)   힙각속도(°/s)');
tauHist.forEach((it,i)=>{
    const v=hipVelHist[i];
    console.log(`  ${it.t.toFixed(2).padStart(6)}  ${(it.tau*1000).toFixed(3).padStart(10)}  ${(v.vel*180/Math.PI).toFixed(1).padStart(10)}`);
});

console.log('\n[최종 상태 (t=5s)]');
console.log(`  φ=${(state.phi*180/Math.PI).toFixed(4)}° α=${(state.alpha*180/Math.PI).toFixed(4)}° θ=${(state.theta*180/Math.PI).toFixed(4)}°`);
console.log(`  균형 유지: ${Math.abs(state.theta)<0.01&&Math.abs(state.alpha)<0.01?'✅ 성공':'⚠️ 확인 필요'}`);

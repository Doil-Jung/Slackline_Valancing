/**
 * Slackline Balance Simulator — Canvas Renderer (3-DOF + Latency)
 * 
 * 하체(α 자유도)+상체(θ 자유도), 힙 관절, 원호 트랙, CoM, 지연시간, 실시간 그래프
 */
var SL = SL || {};

SL.Renderer = class {
    constructor(simCanvas, graphCanvas, params) {
        this.simCanvas = simCanvas;
        this.graphCanvas = graphCanvas;
        this.simCtx = simCanvas.getContext('2d');
        this.graphCtx = graphCanvas.getContext('2d');
        this.params = params;
        this.graphData = [];
        this.maxDataPoints = 600;

        this.colors = {
            bgGrad1: '#0a0a1a', bgGrad2: '#1a1a3e',
            arc: '#3a86ff', arcGlow: 'rgba(58,134,255,0.25)',
            foot: '#ff006e', footGlow: 'rgba(255,0,110,0.35)',
            lower: '#06d6a0', lowerOutline: '#34d399',
            upper: '#8338ec', upperOutline: '#a855f7',
            hip: '#ffbe0b', hipGlow: 'rgba(255,190,11,0.35)',
            com: '#ff006e',
            refLine: 'rgba(255,255,255,0.15)',
            grid: 'rgba(255,255,255,0.04)',
            text: '#e0e0e0', textDim: 'rgba(255,255,255,0.4)',
            graphTheta: '#ff006e', graphPhi: '#3a86ff',
            graphAlpha: '#06d6a0', graphTau: '#ffbe0b',
            graphBg: '#0d0d24',
        };
        this.resize();
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        [this.simCanvas, this.graphCanvas].forEach(c => {
            const r = c.getBoundingClientRect();
            c.width = r.width * dpr;
            c.height = r.height * dpr;
            c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
        });
        this.simW = this.simCanvas.getBoundingClientRect().width;
        this.simH = this.simCanvas.getBoundingClientRect().height;
        this.graphW = this.graphCanvas.getBoundingClientRect().width;
        this.graphH = this.graphCanvas.getBoundingClientRect().height;
        this.computeTransform();
    }

    computeTransform() {
        const { R, L1, L2 } = this.params;
        const totalH = R + L1 + L2 + 0.3;
        const totalW = 2.5 * R;
        this.scale = Math.min(this.simW * 0.7 / totalW, this.simH * 0.7 / totalH);
        this.originX = this.simW / 2;
        this.originY = this.simH * 0.78;
    }

    toCanvas(x, y) {
        return { cx: this.originX + x * this.scale, cy: this.originY - y * this.scale };
    }

    pushData(t, phi, alpha, theta, tau) {
        this.graphData.push({ time: t, phi, alpha, theta, tau });
        if (this.graphData.length > this.maxDataPoints) this.graphData.shift();
    }
    clearData() { this.graphData = []; }

    draw(state, model, controller) {
        this.computeTransform();
        this.drawSim(state, model, controller);
        this.drawGraphs();
    }

    drawSim(state, model, controller) {
        const ctx = this.simCtx;
        const p = this.params;

        // 배경
        const grad = ctx.createLinearGradient(0, 0, 0, this.simH);
        grad.addColorStop(0, this.colors.bgGrad2);
        grad.addColorStop(1, this.colors.bgGrad1);
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, this.simW, this.simH);
        this.drawGrid(ctx);

        // 원호
        this.drawArc(ctx, p.R);

        // 위치 계산
        const foot = model.getFootPos(state);
        const hip = model.getHipPos(state);
        const head = model.getHeadPos(state);
        const com = model.getTotalCoM(state);

        const footC = this.toCanvas(foot.x, foot.y);
        const hipC = this.toCanvas(hip.x, hip.y);
        const headC = this.toCanvas(head.x, head.y);
        const comC = this.toCanvas(com.x, com.y);

        // 기저면 수직선 (발 위치에서 위로)
        ctx.beginPath();
        ctx.setLineDash([4, 6]);
        ctx.strokeStyle = this.colors.refLine;
        ctx.lineWidth = 1;
        const refTop = this.toCanvas(foot.x, foot.y + p.L1 + p.L2 + 0.3);
        ctx.moveTo(footC.cx, footC.cy);
        ctx.lineTo(refTop.cx, refTop.cy);
        ctx.stroke();
        ctx.setLineDash([]);

        // === 하체 (발→힙) ===
        this.drawSegment(ctx, foot, hip, p.lowerWidth,
            'rgba(6,214,160,0.55)', this.colors.lowerOutline);

        // === 상체 (힙→머리) ===
        this.drawSegment(ctx, hip, head, p.upperWidth,
            'rgba(131,56,236,0.65)', this.colors.upperOutline);

        // === 힙 관절 ===
        ctx.beginPath();
        ctx.arc(hipC.cx, hipC.cy, 9, 0, Math.PI * 2);
        ctx.fillStyle = this.colors.hipGlow;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(hipC.cx, hipC.cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = this.colors.hip;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // === 발 ===
        ctx.beginPath();
        ctx.arc(footC.cx, footC.cy, 10, 0, Math.PI * 2);
        ctx.fillStyle = this.colors.footGlow;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(footC.cx, footC.cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = this.colors.foot;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // === 전체 CoM (십자 마크) ===
        ctx.strokeStyle = this.colors.com;
        ctx.lineWidth = 2.5;
        const s = 6;
        ctx.beginPath();
        ctx.moveTo(comC.cx - s, comC.cy - s); ctx.lineTo(comC.cx + s, comC.cy + s);
        ctx.moveTo(comC.cx + s, comC.cy - s); ctx.lineTo(comC.cx - s, comC.cy + s);
        ctx.stroke();

        // 토크 표시
        if (Math.abs(model.tau) > 1) this.drawTorque(ctx, hipC, model.tau);

        // 정보
        this.drawInfo(ctx, state, model, controller);
    }

    drawSegment(ctx, from, to, width, fillColor, strokeColor) {
        const fromC = this.toCanvas(from.x, from.y);
        const toC = this.toCanvas(to.x, to.y);
        const midCx = (fromC.cx + toC.cx) / 2;
        const midCy = (fromC.cy + toC.cy) / 2;
        const len = Math.sqrt((toC.cx - fromC.cx) ** 2 + (toC.cy - fromC.cy) ** 2);
        const w = width * this.scale;

        const drawAngle = Math.atan2(toC.cy - fromC.cy, toC.cx - fromC.cx) + Math.PI / 2;

        ctx.save();
        ctx.translate(midCx, midCy);
        ctx.rotate(drawAngle);

        ctx.fillStyle = fillColor;
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(-w / 2, -len / 2, w, len, 4);
        ctx.fill();
        ctx.stroke();
        ctx.restore();
    }

    drawArc(ctx, R) {
        const phiMax = this.params.phiMax;
        const steps = 60;
        // 글로우
        ctx.beginPath();
        ctx.strokeStyle = this.colors.arcGlow;
        ctx.lineWidth = 8;
        for (let i = 0; i <= steps; i++) {
            const phi = -phiMax + (2 * phiMax * i) / steps;
            const p = this.toCanvas(R * Math.sin(phi), R * (1 - Math.cos(phi)));
            i === 0 ? ctx.moveTo(p.cx, p.cy) : ctx.lineTo(p.cx, p.cy);
        }
        ctx.stroke();
        // 점선
        ctx.beginPath();
        ctx.strokeStyle = this.colors.arc;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([8, 4]);
        for (let i = 0; i <= steps; i++) {
            const phi = -phiMax + (2 * phiMax * i) / steps;
            const p = this.toCanvas(R * Math.sin(phi), R * (1 - Math.cos(phi)));
            i === 0 ? ctx.moveTo(p.cx, p.cy) : ctx.lineTo(p.cx, p.cy);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    drawGrid(ctx) {
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;
        for (let y = -2; y <= 5; y += 0.5) {
            const a = this.toCanvas(-5, y), b = this.toCanvas(5, y);
            ctx.beginPath(); ctx.moveTo(a.cx, a.cy); ctx.lineTo(b.cx, b.cy); ctx.stroke();
        }
        for (let x = -5; x <= 5; x += 0.5) {
            const a = this.toCanvas(x, -2), b = this.toCanvas(x, 5);
            ctx.beginPath(); ctx.moveTo(a.cx, a.cy); ctx.lineTo(b.cx, b.cy); ctx.stroke();
        }
    }

    drawTorque(ctx, hipC, tau) {
        const strength = Math.min(Math.abs(tau) / 400, 1);
        const dir = tau > 0 ? 1 : -1;
        const radius = 16 + strength * 8;
        ctx.beginPath();
        const sa = -Math.PI / 2 - dir * 0.3;
        const ea = -Math.PI / 2 + dir * (0.5 + strength * 1.5);
        ctx.arc(hipC.cx, hipC.cy, radius, sa, ea, dir < 0);
        ctx.strokeStyle = `rgba(255,190,11,${0.3 + strength * 0.5})`;
        ctx.lineWidth = 2 + strength * 2;
        ctx.stroke();
    }

    drawInfo(ctx, state, model, controller) {
        ctx.font = '13px "Inter", monospace';
        const thetaDeg = (state.theta * 180 / Math.PI).toFixed(1);
        const alphaDeg = (state.alpha * 180 / Math.PI).toFixed(1);
        const phiDeg = (state.phi * 180 / Math.PI).toFixed(1);
        const hipDeg = ((state.theta - state.alpha) * 180 / Math.PI).toFixed(1);
        const comErr = (model.getCoMError(state) * 100).toFixed(1); // cm 단위
        const energy = model.getEnergy(state);

        const lines = [
            `t = ${state.time.toFixed(2)} s`,
            `θ = ${thetaDeg}° (상체)`,
            `α = ${alphaDeg}° (하체)`,
            `φ = ${phiDeg}° (발)`,
            `힙각 = ${hipDeg}°`,
            `τ = ${model.tau.toFixed(0)} N·m`,
            `CoM오차 = ${comErr} cm`,
            `E = ${energy.total.toFixed(1)} J`,
        ];

        // 지연시간 정보 추가
        if (controller && controller.getDelayInfo) {
            const d = controller.getDelayInfo();
            if (d.totalMs > 0) {
                lines.push(`⏱ 지연 = ${d.totalMs} ms`);
                if (d.augmented) {
                    lines.push(`  📐 상태증강 (${6 + d.augN}차원)`);
                } else {
                    lines.push(`  센서 ${d.sensorMs}ms + 구동 ${d.actuatorMs}ms`);
                }
            }
            if (d.augmented && d.totalMs === 0) {
                lines.push(`📐 상태증강 모드 (지연=0)`);
            }
        }

        const x = 15; let y = 25;
        ctx.fillStyle = 'rgba(0,0,0,0.4)';
        ctx.fillRect(x - 5, y - 16, 190, lines.length * 20 + 8);
        ctx.fillStyle = this.colors.text;
        lines.forEach(l => { ctx.fillText(l, x, y); y += 20; });
    }

    drawGraphs() {
        const ctx = this.graphCtx;
        const w = this.graphW, h = this.graphH;
        ctx.fillStyle = this.colors.graphBg;
        ctx.fillRect(0, 0, w, h);
        if (this.graphData.length < 2) return;

        const graphs = [
            { key: 'theta', label: 'θ (rad)', color: this.colors.graphTheta, range: 0.5 },
            { key: 'alpha', label: 'α (rad)', color: this.colors.graphAlpha, range: 0.5 },
            { key: 'phi', label: 'φ (rad)', color: this.colors.graphPhi, range: 1.0 },
            { key: 'tau', label: 'τ (N·m)', color: this.colors.graphTau, range: 800 },
        ];
        const pad = { left: 55, right: 15, top: 8, bottom: 5 };
        const gH = (h - pad.top - pad.bottom) / graphs.length;
        const gW = w - pad.left - pad.right;

        graphs.forEach((g, idx) => {
            const gy = pad.top + idx * gH;
            if (idx % 2 === 0) { ctx.fillStyle = 'rgba(255,255,255,0.02)'; ctx.fillRect(pad.left, gy, gW, gH); }

            const cy = gy + gH / 2;
            ctx.beginPath(); ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1;
            ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + gW, cy); ctx.stroke();

            ctx.font = '11px "Inter",sans-serif'; ctx.fillStyle = g.color; ctx.textAlign = 'left';
            ctx.fillText(g.label, 5, cy + 4);

            let maxVal = g.range;
            this.graphData.forEach(d => { const v = Math.abs(d[g.key]); if (v > maxVal) maxVal = v * 1.2; });

            ctx.beginPath(); ctx.strokeStyle = g.color; ctx.lineWidth = 1.5;
            this.graphData.forEach((d, i) => {
                const x = pad.left + (i / (this.maxDataPoints - 1)) * gW;
                const y = cy - (d[g.key] / maxVal) * (gH / 2 - 4);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();
        });
    }
};

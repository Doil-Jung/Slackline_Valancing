/**
 * Slackline Balance Simulator V1 — Canvas Renderer (1-Body)
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
            body: '#8338ec', bodyOutline: '#a855f7',
            com: '#ffbe0b',
            refLine: 'rgba(255,255,255,0.15)',
            grid: 'rgba(255,255,255,0.04)',
            text: '#e0e0e0', textDim: 'rgba(255,255,255,0.4)',
            graphTheta: '#ff006e', graphPhi: '#3a86ff', graphTau: '#ffbe0b',
            graphBg: '#0d0d24',
        };
        this.resize();
    }

    // (V1 코스트를 줄이기 위해 필수 기능만 남김: DrawSim 중심)
    resize() {
        const dpr = window.devicePixelRatio || 1;
        [this.simCanvas, this.graphCanvas].forEach(c => {
            const r = c.getBoundingClientRect();
            c.width = r.width * dpr; c.height = r.height * dpr;
            c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
        });
        this.simW = this.simCanvas.getBoundingClientRect().width;
        this.simH = this.simCanvas.getBoundingClientRect().height;
        this.graphW = this.graphCanvas.getBoundingClientRect().width;
        this.graphH = this.graphCanvas.getBoundingClientRect().height;
        this.computeTransform();
    }

    computeTransform() {
        const p = this.params;
        const totalH = p.R + p.L * 2 + 0.3;
        const totalW = 2.5 * p.R;
        this.scale = Math.min(this.simW * 0.7 / totalW, this.simH * 0.7 / totalH);
        this.originX = this.simW / 2;
        this.originY = this.simH * 0.78;
    }

    toCanvas(x, y) { return { cx: this.originX + x * this.scale, cy: this.originY - y * this.scale }; }

    pushData(t, phi, theta, tau) {
        this.graphData.push({ time: t, phi, theta, tau });
        if (this.graphData.length > this.maxDataPoints) this.graphData.shift();
    }
    clearData() { this.graphData = []; }

    draw(state, model) {
        this.computeTransform();
        this.drawSim(state, model);
        this.drawGraphs();
    }

    drawSim(state, model) {
        const ctx = this.simCtx; const p = this.params;
        const grad = ctx.createLinearGradient(0, 0, 0, this.simH);
        grad.addColorStop(0, this.colors.bgGrad2); grad.addColorStop(1, this.colors.bgGrad1);
        ctx.fillStyle = grad; ctx.fillRect(0, 0, this.simW, this.simH);

        this.drawArc(ctx, p.R);

        const foot = model.getFootPos(state);
        const com = model.getCoMPos(state);
        const footC = this.toCanvas(foot.x, foot.y);
        const comC = this.toCanvas(com.x, com.y);

        // 점선
        ctx.beginPath(); ctx.setLineDash([4, 6]); ctx.strokeStyle = this.colors.refLine;
        const refTop = this.toCanvas(foot.x, foot.y + p.L * 2 + 0.3);
        ctx.moveTo(footC.cx, footC.cy); ctx.lineTo(refTop.cx, refTop.cy);
        ctx.stroke(); ctx.setLineDash([]);

        // 단일 몸체
        const bw = p.bodyWidth * this.scale;
        const bh = p.L * 2 * this.scale;
        ctx.save();
        ctx.translate(footC.cx, footC.cy);
        ctx.rotate(state.theta);
        ctx.fillStyle = 'rgba(131,56,236,0.55)'; ctx.strokeStyle = this.colors.bodyOutline;
        ctx.lineWidth = 2; ctx.beginPath();
        ctx.roundRect(-bw / 2, -bh, bw, bh, 8);
        ctx.fill(); ctx.stroke();
        ctx.restore();

        // 발
        ctx.beginPath(); ctx.arc(footC.cx, footC.cy, 10, 0, Math.PI * 2);
        ctx.fillStyle = this.colors.footGlow; ctx.fill();
        ctx.beginPath(); ctx.arc(footC.cx, footC.cy, 5, 0, Math.PI * 2);
        ctx.fillStyle = this.colors.foot; ctx.fill(); ctx.stroke();

        // CoM
        ctx.strokeStyle = this.colors.com; ctx.lineWidth = 2.5; const s = 6;
        ctx.beginPath(); ctx.moveTo(comC.cx - s, comC.cy - s); ctx.lineTo(comC.cx + s, comC.cy + s);
        ctx.moveTo(comC.cx + s, comC.cy - s); ctx.lineTo(comC.cx - s, comC.cy + s); ctx.stroke();

        this.drawInfo(ctx, state, model);
    }

    drawArc(ctx, R) {
        const phiMax = this.params.phiMax; const steps = 60;
        ctx.beginPath(); ctx.strokeStyle = this.colors.arcGlow; ctx.lineWidth = 8;
        for (let i = 0; i <= steps; i++) {
            const phi = -phiMax + (2 * phiMax * i) / steps;
            const p = this.toCanvas(R * Math.sin(phi), R * (1 - Math.cos(phi)));
            i === 0 ? ctx.moveTo(p.cx, p.cy) : ctx.lineTo(p.cx, p.cy);
        } ctx.stroke();
        ctx.beginPath(); ctx.strokeStyle = this.colors.arc; ctx.lineWidth = 2.5; ctx.setLineDash([8, 4]);
        for (let i = 0; i <= steps; i++) {
            const phi = -phiMax + (2 * phiMax * i) / steps;
            const p = this.toCanvas(R * Math.sin(phi), R * (1 - Math.cos(phi)));
            i === 0 ? ctx.moveTo(p.cx, p.cy) : ctx.lineTo(p.cx, p.cy);
        } ctx.stroke(); ctx.setLineDash([]);
    }

    drawInfo(ctx, state, model) {
        ctx.font = '13px "Inter", monospace';
        const lines = [
            `t = ${state.time.toFixed(2)} s`,
            `θ = ${(state.theta * 180 / Math.PI).toFixed(1)}°`,
            `φ = ${(state.phi * 180 / Math.PI).toFixed(1)}°`,
            `τ = ${model.tau.toFixed(0)} N·m`
        ];
        const x = 15; let y = 25;
        ctx.fillStyle = 'rgba(0,0,0,0.4)'; ctx.fillRect(x - 5, y - 16, 120, lines.length * 20 + 8);
        ctx.fillStyle = this.colors.text;
        lines.forEach(l => { ctx.fillText(l, x, y); y += 20; });
    }

    drawGraphs() {
        const ctx = this.graphCtx; const w = this.graphW, h = this.graphH;
        ctx.fillStyle = this.colors.graphBg; ctx.fillRect(0, 0, w, h);
        if (this.graphData.length < 2) return;
        const graphs = [
            { key: 'theta', color: this.colors.graphTheta, range: 0.5 },
            { key: 'phi', color: this.colors.graphPhi, range: 1.0 },
            { key: 'tau', color: this.colors.graphTau, range: 800 }
        ];
        const pad = { left: 55, right: 15, top: 8, bottom: 5 };
        const gH = (h - pad.top - pad.bottom) / graphs.length, gW = w - pad.left - pad.right;

        graphs.forEach((g, idx) => {
            const gy = pad.top + idx * gH, cy = gy + gH / 2;
            ctx.beginPath(); ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1;
            ctx.moveTo(pad.left, cy); ctx.lineTo(pad.left + gW, cy); ctx.stroke();

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

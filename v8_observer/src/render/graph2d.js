/**
 * Slackline Balance Simulator V6 — 2D Graph Renderer
 * 
 * v5 renderer.js에서 그래프 렌더링 부분을 분리
 * θ, α, φ, τ 실시간 그래프 표시
 */
var SL = SL || {};

SL.GraphRenderer = class {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.graphData = [];
        this.maxDataPoints = 600;

        this.colors = {
            graphTheta: '#ff006e',
            graphPhi: '#3a86ff',
            graphAlpha: '#06d6a0',
            graphTau: '#ffbe0b',
            graphBg: '#0d0d24',
        };
        this.resize();
    }

    resize() {
        const dpr = window.devicePixelRatio || 1;
        const r = this.canvas.getBoundingClientRect();
        this.canvas.width = r.width * dpr;
        this.canvas.height = r.height * dpr;
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.w = r.width;
        this.h = r.height;
    }

    pushData(t, phi, alpha, theta, tau) {
        this.graphData.push({ time: t, phi, alpha, theta, tau });
        if (this.graphData.length > this.maxDataPoints) this.graphData.shift();
    }

    clearData() { this.graphData = []; }

    draw() {
        const ctx = this.ctx;
        const w = this.w, h = this.h;
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
            if (idx % 2 === 0) {
                ctx.fillStyle = 'rgba(255,255,255,0.02)';
                ctx.fillRect(pad.left, gy, gW, gH);
            }

            const cy = gy + gH / 2;
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(255,255,255,0.08)';
            ctx.lineWidth = 1;
            ctx.moveTo(pad.left, cy);
            ctx.lineTo(pad.left + gW, cy);
            ctx.stroke();

            ctx.font = '11px "Inter",sans-serif';
            ctx.fillStyle = g.color;
            ctx.textAlign = 'left';
            ctx.fillText(g.label, 5, cy + 4);

            let maxVal = g.range;
            this.graphData.forEach(d => {
                const v = Math.abs(d[g.key]);
                if (v > maxVal) maxVal = v * 1.2;
            });

            ctx.beginPath();
            ctx.strokeStyle = g.color;
            ctx.lineWidth = 1.5;
            this.graphData.forEach((d, i) => {
                const x = pad.left + (i / (this.maxDataPoints - 1)) * gW;
                const y = cy - (d[g.key] / maxVal) * (gH / 2 - 4);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();
        });
    }
};

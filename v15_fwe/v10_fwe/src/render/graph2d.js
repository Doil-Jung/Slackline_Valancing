/**
 * Slackline Balance Simulator V15 — 2D Graph Renderer
 * 
 * φ, β, δ, τ 실시간 그래프 + FWE 페이즈 배경 표시
 */
var SL = SL || {};

SL.GraphRenderer = class {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.graphData = [];
        this.maxDataPoints = 600;

        this.colors = {
            graphPhi: '#3a86ff',
            graphBeta: '#fb7185',
            graphDelta: '#c084fc',
            graphTau: '#ffbe0b',
            graphBg: '#0d0d24',
        };

        this.phaseColors = {
            'IDLE':    null,
            'FOLD':    'rgba(255, 215, 0, 0.15)',
            'WAIT1':   'rgba(6, 214, 160, 0.15)',
            'EXTEND':  'rgba(58, 134, 255, 0.15)',
            'WAIT2':   'rgba(131, 56, 236, 0.15)',
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

    pushData(t, phi, alpha, theta, beta, delta, tau, phase) {
        this.graphData.push({ time: t, phi, alpha, theta, beta, delta, tau, phase: phase || 'IDLE' });
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
            { key: 'phi', label: 'φ (rad)', color: this.colors.graphPhi, range: 1.0 },
            { key: 'beta', label: 'β (rad)', color: this.colors.graphBeta, range: 0.5 },
            { key: 'delta', label: 'δ (rad)', color: this.colors.graphDelta, range: 0.5 },
            { key: 'tau', label: 'τ (N·m)', color: this.colors.graphTau, range: 800 },
        ];
        const pad = { left: 55, right: 15, top: 8, bottom: 5 };
        const gH = (h - pad.top - pad.bottom) / graphs.length;
        const gW = w - pad.left - pad.right;

        // 페이즈 배경 그리기 (모든 그래프 영역에 공통 적용)
        this._drawPhaseBackground(ctx, pad, gH, gW, graphs.length);

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

        // 페이즈 레이블 (하단 축에 표시)
        this._drawPhaseLabels(ctx, pad, gH, gW, graphs.length);
    }

    /** 페이즈 배경 색상 밴드 */
    _drawPhaseBackground(ctx, pad, gH, gW, nGraphs) {
        const data = this.graphData;
        const totalH = nGraphs * gH;
        let prevPhase = data[0].phase;
        let startI = 0;

        for (let i = 1; i <= data.length; i++) {
            const curPhase = i < data.length ? data[i].phase : null;
            if (curPhase !== prevPhase || i === data.length) {
                const color = this.phaseColors[prevPhase];
                if (color) {
                    const x0 = pad.left + (startI / (this.maxDataPoints - 1)) * gW;
                    const x1 = pad.left + ((i - 1) / (this.maxDataPoints - 1)) * gW;
                    ctx.fillStyle = color;
                    ctx.fillRect(x0, pad.top, x1 - x0, totalH);
                }
                prevPhase = curPhase;
                startI = i;
            }
        }
    }

    /** 페이즈 레이블 (최하단에 약어 표시) */
    _drawPhaseLabels(ctx, pad, gH, gW, nGraphs) {
        const data = this.graphData;
        const labelY = pad.top + nGraphs * gH - 3;
        const phaseLabels = { 'FOLD': 'F', 'WAIT1': 'W₁', 'EXTEND': 'E', 'WAIT2': 'W₂' };
        const phaseLabelColors = {
            'FOLD': '#FFD700', 'WAIT1': '#06d6a0', 'EXTEND': '#3a86ff', 'WAIT2': '#8338ec'
        };

        let prevPhase = data[0].phase;
        let startI = 0;

        for (let i = 1; i <= data.length; i++) {
            const curPhase = i < data.length ? data[i].phase : null;
            if (curPhase !== prevPhase || i === data.length) {
                const label = phaseLabels[prevPhase];
                if (label) {
                    const x0 = pad.left + (startI / (this.maxDataPoints - 1)) * gW;
                    const x1 = pad.left + ((i - 1) / (this.maxDataPoints - 1)) * gW;
                    const xMid = (x0 + x1) / 2;
                    if (x1 - x0 > 12) {
                        ctx.font = 'bold 10px "Inter",sans-serif';
                        ctx.fillStyle = phaseLabelColors[prevPhase] || '#aaa';
                        ctx.textAlign = 'center';
                        ctx.fillText(label, xMid, labelY);
                    }
                }
                prevPhase = curPhase;
                startI = i;
            }
        }
    }
};

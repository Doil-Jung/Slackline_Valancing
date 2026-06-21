/**
 * Slackline Balance Simulator V16 — 2D Graph Renderer
 * 
 * φ, β, δ, τ 실시간 그래프 + FWE 페이즈 배경 표시
 * 도(°) 단위 표시, Y축 눈금, X축 2초 윈도우 (현재 시간 중앙)
 * 뒤로 가기 시 미래 데이터 유지
 */
var SL = SL || {};

SL.GraphRenderer = class {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.graphData = [];
        this.maxDataPoints = 10000;
        this.windowSize = 2.0;    // X축 시간 윈도우 (초)
        this.currentTime = 0;     // 현재 재생 커서 위치

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
        this.currentTime = t;
    }

    /** 특정 시간 이후의 데이터를 제거 (뒤로 간 후 다시 진행 시) */
    trimAfter(t) {
        const eps = 1e-6;
        while (this.graphData.length > 0 && this.graphData[this.graphData.length - 1].time > t + eps) {
            this.graphData.pop();
        }
    }

    clearData() {
        this.graphData = [];
        this.currentTime = 0;
    }

    draw() {
        const ctx = this.ctx;
        const w = this.w, h = this.h;
        ctx.fillStyle = this.colors.graphBg;
        ctx.fillRect(0, 0, w, h);
        if (this.graphData.length < 2) return;

        const RAD2DEG = 180 / Math.PI;
        const halfWin = this.windowSize / 2;

        const graphs = [
            { key: 'phi',   label: 'φ (°)',    color: this.colors.graphPhi,   range: 0.5,   unit: 'deg' },
            { key: 'beta',  label: 'β (°)',    color: this.colors.graphBeta,  range: 0.5,   unit: 'deg' },
            { key: 'delta', label: 'δ (°)',    color: this.colors.graphDelta, range: 0.5,   unit: 'deg' },
            { key: 'tau',   label: 'τ (N·m)',  color: this.colors.graphTau,   range: 2,     unit: 'Nm'  },
        ];

        const pad = { left: 62, right: 15, top: 8, bottom: 22 };
        const gH = (h - pad.top - pad.bottom) / graphs.length;
        const gW = w - pad.left - pad.right;

        // 시간 윈도우: 현재 시간 중앙
        const tCenter = this.currentTime;
        const tStart = tCenter - halfWin;
        const tEnd = tCenter + halfWin;

        // 윈도우 내 데이터 필터 (약간의 여유)
        const visibleData = this.graphData.filter(d => d.time >= tStart - 0.1 && d.time <= tEnd + 0.1);
        if (visibleData.length < 1) return;

        // 페이즈 배경 그리기
        this._drawPhaseBackground(ctx, pad, gH, gW, graphs.length, visibleData, tStart, tEnd);

        graphs.forEach((g, idx) => {
            const gy = pad.top + idx * gH;

            // 교차 배경
            if (idx % 2 === 0) {
                ctx.fillStyle = 'rgba(255,255,255,0.02)';
                ctx.fillRect(pad.left, gy, gW, gH);
            }

            const cy = gy + gH / 2;

            // 동적 범위 계산 (보이는 데이터 기준)
            let maxVal = g.range;
            visibleData.forEach(d => {
                let v = Math.abs(d[g.key]);
                if (g.unit === 'deg') v *= RAD2DEG;
                if (v > maxVal) maxVal = v * 1.3;
            });

            // Y축 눈금 계산
            const tickStep = this._niceStep(maxVal);
            const nTicks = Math.floor(maxVal / tickStep);

            // Y축 눈금선 & 라벨
            ctx.font = '9px "Inter",monospace';
            ctx.textAlign = 'right';
            for (let i = -nTicks; i <= nTicks; i++) {
                const val = i * tickStep;
                const y = cy - (val / maxVal) * (gH / 2 - 6);
                if (y < gy + 2 || y > gy + gH - 2) continue;

                ctx.beginPath();
                ctx.strokeStyle = i === 0 ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.05)';
                ctx.lineWidth = i === 0 ? 1 : 0.5;
                ctx.moveTo(pad.left, y);
                ctx.lineTo(pad.left + gW, y);
                ctx.stroke();

                if (i !== 0) {
                    ctx.fillStyle = 'rgba(255,255,255,0.3)';
                    const label = g.unit === 'deg' ? val.toFixed(1) + '°' : val.toFixed(2);
                    ctx.fillText(label, pad.left - 4, y + 3);
                }
            }

            // 0선 강조
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(255,255,255,0.15)';
            ctx.lineWidth = 1;
            ctx.moveTo(pad.left, cy);
            ctx.lineTo(pad.left + gW, cy);
            ctx.stroke();

            // 그래프 라벨
            ctx.font = '11px "Inter",sans-serif';
            ctx.fillStyle = g.color;
            ctx.textAlign = 'left';
            ctx.fillText(g.label, 5, cy + 4);

            // 현재 시간에서의 값 표시 (가장 가까운 데이터)
            let curVal = 0;
            let minDist = Infinity;
            visibleData.forEach(d => {
                const dist = Math.abs(d.time - tCenter);
                if (dist < minDist) { minDist = dist; curVal = d[g.key]; }
            });
            if (g.unit === 'deg') curVal *= RAD2DEG;
            ctx.font = 'bold 10px "Inter",monospace';
            ctx.textAlign = 'left';
            ctx.fillStyle = g.color;
            const curLabel = g.unit === 'deg' ? curVal.toFixed(2) + '°' : curVal.toFixed(2);
            ctx.fillText(curLabel, 5, cy + 16);

            // 데이터 라인 — 과거(현재 이전) 실선, 미래(현재 이후) 반투명
            this._drawDataLine(ctx, visibleData, g, tStart, tEnd, tCenter, pad, gW, cy, gH, maxVal, RAD2DEG);

            // 구분선
            if (idx < graphs.length - 1) {
                ctx.beginPath();
                ctx.strokeStyle = 'rgba(255,255,255,0.08)';
                ctx.lineWidth = 1;
                ctx.moveTo(pad.left, gy + gH);
                ctx.lineTo(pad.left + gW, gy + gH);
                ctx.stroke();
            }
        });

        // 현재 시간 수직선 (NOW 인디케이터)
        this._drawNowLine(ctx, pad, gH, gW, graphs.length, tStart, tEnd, tCenter);

        // X축 시간 눈금
        this._drawTimeAxis(ctx, pad, gH, gW, graphs.length, tStart, tEnd, tCenter);

        // 페이즈 레이블
        this._drawPhaseLabels(ctx, pad, gH, gW, graphs.length, visibleData, tStart, tEnd);
    }

    /** 데이터 라인 그리기 (현재 이전=실선, 현재 이후=반투명 점선) */
    _drawDataLine(ctx, visibleData, g, tStart, tEnd, tCenter, pad, gW, cy, gH, maxVal, RAD2DEG) {
        const duration = tEnd - tStart;
        const eps = 0.0001;

        // 과거 데이터 (현재 시간까지)
        const pastData = visibleData.filter(d => d.time <= tCenter + eps);
        if (pastData.length > 0) {
            ctx.beginPath();
            ctx.strokeStyle = g.color;
            ctx.lineWidth = 1.8;
            pastData.forEach((d, i) => {
                const x = pad.left + ((d.time - tStart) / duration) * gW;
                let val = d[g.key];
                if (g.unit === 'deg') val *= RAD2DEG;
                const y = cy - (val / maxVal) * (gH / 2 - 6);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();
        }

        // 미래 데이터 (현재 시간 이후) — 반투명 점선
        const futureData = visibleData.filter(d => d.time >= tCenter - eps);
        if (futureData.length > 1) {
            ctx.beginPath();
            ctx.strokeStyle = g.color + '55';  // 33% opacity
            ctx.lineWidth = 1.2;
            ctx.setLineDash([4, 3]);
            futureData.forEach((d, i) => {
                const x = pad.left + ((d.time - tStart) / duration) * gW;
                let val = d[g.key];
                if (g.unit === 'deg') val *= RAD2DEG;
                const y = cy - (val / maxVal) * (gH / 2 - 6);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    /** NOW 수직선 */
    _drawNowLine(ctx, pad, gH, gW, nGraphs, tStart, tEnd, tCenter) {
        const duration = tEnd - tStart;
        const x = pad.left + ((tCenter - tStart) / duration) * gW;
        const yTop = pad.top;
        const yBottom = pad.top + nGraphs * gH;

        // 수직선
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1.5;
        ctx.moveTo(x, yTop);
        ctx.lineTo(x, yBottom);
        ctx.stroke();

        // 시간 라벨
        ctx.font = 'bold 9px "Inter",monospace';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.textAlign = 'center';
        ctx.fillText('▼ ' + tCenter.toFixed(2) + 's', x, yTop - 1);
    }

    /** 적절한 눈금 간격 계산 */
    _niceStep(maxVal) {
        if (maxVal <= 0.2) return 0.05;
        if (maxVal <= 0.5) return 0.1;
        if (maxVal <= 1.0) return 0.2;
        if (maxVal <= 2.0) return 0.5;
        if (maxVal <= 5.0) return 1.0;
        if (maxVal <= 10) return 2.0;
        if (maxVal <= 50) return 10;
        if (maxVal <= 100) return 20;
        if (maxVal <= 500) return 100;
        return 200;
    }

    /** X축 시간 눈금 */
    _drawTimeAxis(ctx, pad, gH, gW, nGraphs, tStart, tEnd, tCenter) {
        const y = pad.top + nGraphs * gH;
        const duration = tEnd - tStart;

        let tStep = 0.5;
        if (duration > 5) tStep = 1.0;
        if (duration > 20) tStep = 5.0;

        ctx.font = '9px "Inter",monospace';
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.textAlign = 'center';

        const firstTick = Math.ceil(tStart / tStep) * tStep;
        for (let t = firstTick; t <= tEnd; t += tStep) {
            const x = pad.left + ((t - tStart) / duration) * gW;

            // 시간 눈금 수직선
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(255,255,255,0.06)';
            ctx.lineWidth = 0.5;
            ctx.moveTo(x, pad.top);
            ctx.lineTo(x, y);
            ctx.stroke();

            // 라벨
            ctx.fillText(t.toFixed(1) + 's', x, y + 12);
        }
    }

    /** 페이즈 배경 색상 밴드 (시간 기반) */
    _drawPhaseBackground(ctx, pad, gH, gW, nGraphs, visibleData, tStart, tEnd) {
        if (visibleData.length === 0) return;
        const totalH = nGraphs * gH;
        const duration = tEnd - tStart;
        let prevPhase = visibleData[0].phase;
        let startTime = visibleData[0].time;

        for (let i = 1; i <= visibleData.length; i++) {
            const curPhase = i < visibleData.length ? visibleData[i].phase : null;
            if (curPhase !== prevPhase || i === visibleData.length) {
                const color = this.phaseColors[prevPhase];
                if (color) {
                    const x0 = pad.left + Math.max(0, ((startTime - tStart) / duration)) * gW;
                    const endT = i < visibleData.length ? visibleData[i-1].time : visibleData[visibleData.length-1].time;
                    const x1 = pad.left + Math.min(1, ((endT - tStart) / duration)) * gW;
                    ctx.fillStyle = color;
                    ctx.fillRect(x0, pad.top, Math.max(1, x1 - x0), totalH);
                }
                prevPhase = curPhase;
                if (i < visibleData.length) startTime = visibleData[i].time;
            }
        }
    }

    /** 페이즈 레이블 */
    _drawPhaseLabels(ctx, pad, gH, gW, nGraphs, visibleData, tStart, tEnd) {
        if (visibleData.length === 0) return;
        const labelY = pad.top + nGraphs * gH - 3;
        const duration = tEnd - tStart;
        const phaseLabels = { 'FOLD': 'F', 'WAIT1': 'W₁', 'EXTEND': 'E', 'WAIT2': 'W₂' };
        const phaseLabelColors = {
            'FOLD': '#FFD700', 'WAIT1': '#06d6a0', 'EXTEND': '#3a86ff', 'WAIT2': '#8338ec'
        };

        let prevPhase = visibleData[0].phase;
        let startTime = visibleData[0].time;

        for (let i = 1; i <= visibleData.length; i++) {
            const curPhase = i < visibleData.length ? visibleData[i].phase : null;
            if (curPhase !== prevPhase || i === visibleData.length) {
                const label = phaseLabels[prevPhase];
                if (label) {
                    const x0 = pad.left + ((startTime - tStart) / duration) * gW;
                    const endTime = i < visibleData.length ? visibleData[i-1].time : visibleData[visibleData.length-1].time;
                    const x1 = pad.left + ((endTime - tStart) / duration) * gW;
                    const xMid = (x0 + x1) / 2;
                    if (x1 - x0 > 12) {
                        ctx.font = 'bold 10px "Inter",sans-serif';
                        ctx.fillStyle = phaseLabelColors[prevPhase] || '#aaa';
                        ctx.textAlign = 'center';
                        ctx.fillText(label, xMid, labelY);
                    }
                }
                prevPhase = curPhase;
                if (i < visibleData.length) startTime = visibleData[i].time;
            }
        }
    }
};

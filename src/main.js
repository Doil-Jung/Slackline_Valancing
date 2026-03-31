/**
 * Slackline Balance Simulator V4 — Main Entry Point (3-DOF + LQR)
 * 
 * LQR 최적 상태 피드백 제어기를 사용한 균형 시뮬레이션
 */
var SL = SL || {};

(function () {
    'use strict';

    let model, controller, renderer;
    let running = false;
    let animFrameId = null;

    /** 초기화 */
    function init() {
        // 물리 모델
        model = new SL.Model(SL.Params);

        // LQR 제어기
        controller = new SL.LQRController();

        // 렌더러
        const simCanvas = document.getElementById('sim-canvas');
        const graphCanvas = document.getElementById('graph-canvas');
        renderer = new SL.Renderer(simCanvas, graphCanvas, SL.Params);

        // UI 바인딩
        setupSliders();
        setupButtons();

        // 리사이즈
        window.addEventListener('resize', () => renderer.resize());

        // 초기 렌더
        renderer.draw(model.state, model);

        console.log('Slackline Balance Simulator V4 (3-DOF + LQR) initialized');
    }

    /** 슬라이더 UI 설정 */
    function setupSliders() {
        const ranges = SL.Params.ranges;

        Object.keys(ranges).forEach(key => {
            const slider = document.getElementById(`slider-${key}`);
            const display = document.getElementById(`val-${key}`);
            if (!slider || !display) return;

            const range = ranges[key];
            slider.min = range.min;
            slider.max = range.max;
            slider.step = range.step;
            slider.value = SL.Params[key];
            display.textContent = Number(SL.Params[key]).toFixed(
                range.step < 1 ? (range.step < 0.1 ? 2 : 1) : 0
            );

            slider.addEventListener('input', () => {
                const val = parseFloat(slider.value);
                SL.Params[key] = val;
                display.textContent = val.toFixed(
                    range.step < 1 ? (range.step < 0.1 ? 2 : 1) : 0
                );
                renderer.params = SL.Params;
                renderer.computeTransform();
            });
        });

        // 교란 강도 슬라이더
        const psSlider = document.getElementById('slider-perturbStr');
        const psDisplay = document.getElementById('val-perturbStr');
        if (psSlider && psDisplay) {
            psSlider.addEventListener('input', () => {
                psDisplay.textContent = parseFloat(psSlider.value).toFixed(1);
            });
        }

        // 게인 스케일 슬라이더
        const gsSlider = document.getElementById('slider-gainScale');
        const gsDisplay = document.getElementById('val-gainScale');
        if (gsSlider && gsDisplay) {
            gsSlider.min = 0; gsSlider.max = 2.0; gsSlider.step = 0.05;
            gsSlider.value = 1.0;
            gsDisplay.textContent = '1.00';
            gsSlider.addEventListener('input', () => {
                const val = parseFloat(gsSlider.value);
                controller.setGainScale(val);
                gsDisplay.textContent = val.toFixed(2);
            });
        }

        // 적분 게인 슬라이더
        const kiSlider = document.getElementById('slider-lqrKi');
        const kiDisplay = document.getElementById('val-lqrKi');
        if (kiSlider && kiDisplay) {
            kiSlider.min = 0; kiSlider.max = 50; kiSlider.step = 1;
            kiSlider.value = 0;
            kiDisplay.textContent = '0';
            kiSlider.addEventListener('input', () => {
                const val = parseFloat(kiSlider.value);
                controller.setKi(val);
                kiDisplay.textContent = val.toFixed(0);
            });
        }
    }

    /** 버튼 UI 설정 */
    function setupButtons() {
        const btnStartStop = document.getElementById('btn-start-stop');
        btnStartStop.addEventListener('click', () => {
            if (running) {
                stop();
                btnStartStop.textContent = '▶ 시작';
                btnStartStop.classList.remove('active');
            } else {
                start();
                btnStartStop.textContent = '⏸ 정지';
                btnStartStop.classList.add('active');
            }
        });

        document.getElementById('btn-reset').addEventListener('click', () => {
            stop();
            SL.Params.alpha0 = parseFloat(document.getElementById('slider-alpha0')?.value || 0);
            SL.Params.theta0 = parseFloat(document.getElementById('slider-theta0')?.value || 0.15);
            model.reset(SL.Params);
            controller.reset();
            renderer.clearData();
            renderer.draw(model.state, model);
            btnStartStop.textContent = '▶ 시작';
            btnStartStop.classList.remove('active');
        });

        document.getElementById('btn-perturb-right').addEventListener('click', () => {
            const str = parseFloat(document.getElementById('slider-perturbStr')?.value || 0.5);
            model.applyPerturbation(str);
        });
        document.getElementById('btn-perturb-left').addEventListener('click', () => {
            const str = parseFloat(document.getElementById('slider-perturbStr')?.value || 0.5);
            model.applyPerturbation(-str);
        });

        const btnController = document.getElementById('btn-controller');
        btnController.addEventListener('click', () => {
            SL.Params.controllerOn = !SL.Params.controllerOn;
            if (SL.Params.controllerOn) {
                btnController.textContent = '🎛 제어기 ON';
                btnController.classList.add('active');
                controller.reset();
            } else {
                btnController.textContent = '🎛 제어기 OFF';
                btnController.classList.remove('active');
            }
        });

        if (SL.Params.controllerOn) btnController.classList.add('active');
    }

    function start() { if (running) return; running = true; loop(); }
    function stop() {
        running = false;
        if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    }

    function loop() {
        if (!running) return;
        model.step(controller);
        renderer.pushData(
            model.state.time,
            model.state.phi,
            model.state.alpha,
            model.state.theta,
            model.tau
        );
        renderer.draw(model.state, model);
        animFrameId = requestAnimationFrame(loop);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

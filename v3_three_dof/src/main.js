/**
 * Slackline Balance Simulator — Main Entry Point (3-DOF)
 * 
 * 시뮬레이션 루프, UI 바인딩, 이벤트 핸들링
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

        // PID 제어기
        controller = new SL.PIDController(
            SL.Params.Kp, SL.Params.Kd, SL.Params.Ki
        );

        // 렌더러
        const simCanvas = document.getElementById('sim-canvas');
        const graphCanvas = document.getElementById('graph-canvas');
        renderer = new SL.Renderer(simCanvas, graphCanvas, SL.Params);

        // UI 슬라이더 바인딩
        setupSliders();

        // 버튼 바인딩
        setupButtons();

        // 리사이즈 핸들링
        window.addEventListener('resize', () => {
            renderer.resize();
        });

        // 초기 렌더
        renderer.draw(model.state, model);

        console.log('Slackline Balance Simulator V3 (3-DOF) initialized');
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

                // 제어기 게인 즉시 업데이트
                if (['Kp', 'Kd', 'Ki'].includes(key)) {
                    controller.setGains(SL.Params.Kp, SL.Params.Kd, SL.Params.Ki);
                }

                // 렌더러 파라미터 갱신
                renderer.params = SL.Params;
                renderer.computeTransform();
            });
        });
    }

    /** 버튼 UI 설정 */
    function setupButtons() {
        // Start / Stop
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

        // Reset
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

        // Perturbation (우측 밀기)
        document.getElementById('btn-perturb-right').addEventListener('click', () => {
            model.applyPerturbation(2.0); // +2 rad/s 충격
        });

        // Perturbation (좌측 밀기)
        document.getElementById('btn-perturb-left').addEventListener('click', () => {
            model.applyPerturbation(-2.0);
        });

        // 제어기 ON/OFF
        const btnController = document.getElementById('btn-controller');
        btnController.addEventListener('click', () => {
            SL.Params.controllerOn = !SL.Params.controllerOn;
            if (SL.Params.controllerOn) {
                btnController.textContent = '🎛 제어기 ON';
                btnController.classList.add('active');
                controller.reset(); // 적분항 초기화
            } else {
                btnController.textContent = '🎛 제어기 OFF';
                btnController.classList.remove('active');
            }
        });

        // 초기 상태 반영
        if (SL.Params.controllerOn) {
            btnController.classList.add('active');
        }
    }

    /** 시뮬레이션 시작 */
    function start() {
        if (running) return;
        running = true;
        loop();
    }

    /** 시뮬레이션 정지 */
    function stop() {
        running = false;
        if (animFrameId) {
            cancelAnimationFrame(animFrameId);
            animFrameId = null;
        }
    }

    /** 메인 루프 */
    function loop() {
        if (!running) return;

        // 물리 스텝
        model.step(controller);

        // 그래프 데이터 기록 (α 추가)
        renderer.pushData(
            model.state.time,
            model.state.phi,
            model.state.alpha,
            model.state.theta,
            model.tau
        );

        // 렌더링
        renderer.draw(model.state, model);

        // 다음 프레임
        animFrameId = requestAnimationFrame(loop);
    }

    // DOM 로드 후 초기화
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

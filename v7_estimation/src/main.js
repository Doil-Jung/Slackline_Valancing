/**
 * Slackline Balance Simulator V6 — Main Entry Point (3D Visualization)
 * 
 * v5의 2D 물리 엔진(3-DOF + LQR + Latency)을 그대로 사용하면서
 * Three.js 3D 렌더러로 시각화
 */
var SL = SL || {};

(function () {
    'use strict';

    let model, controller, renderer3d, graphRenderer;
    let running = false;
    let animFrameId = null;

    // 3D 추가 파라미터 기본값 설정
    SL.Params.L_pole = 4.0;      // 기둥 간 거리 (m)
    SL.Params.L0 = 5.0;          // 줄 전체 길이 (m)
    SL.Params.bodyDepth = 0.25;  // 몸체 깊이 (m)
    SL.Params.showArc = true;    // 원호 궤적 표시 여부

    /** 초기화 */
    function init() {
        model = new SL.Model(SL.Params);
        controller = new SL.LQRController();
        controller.setEstimationMode(false);

        // 3D 렌더러
        const container3d = document.getElementById('container-3d');
        renderer3d = new SL.Renderer3D(container3d, SL.Params);
        renderer3d.startIdleRender();

        // 2D 그래프
        const graphCanvas = document.getElementById('graph-canvas');
        graphRenderer = new SL.GraphRenderer(graphCanvas);

        setupSliders();
        setupButtons();

        window.addEventListener('resize', () => {
            renderer3d.resize();
            graphRenderer.resize();
        });

        // 초기 렌더
        renderer3d.update(model.state, model);
        graphRenderer.draw();

        // 정보 오버레이 초기 업데이트
        updateInfoOverlay(model.state, model, controller);

        console.log('Slackline Balance Simulator V6 (3D Visualization) initialized');
    }

    /** 정보 오버레이 업데이트 */
    function updateInfoOverlay(state, model, controller) {
        const el = document.getElementById('info-overlay');
        if (!el) return;

        const thetaDeg = (state.theta * 180 / Math.PI).toFixed(1);
        const alphaDeg = (state.alpha * 180 / Math.PI).toFixed(1);
        const phiDeg = (state.phi * 180 / Math.PI).toFixed(1);
        const hipDeg = ((state.theta - state.alpha) * 180 / Math.PI).toFixed(1);
        const comErr = (model.getCoMError(state) * 100).toFixed(1);
        const energy = model.getEnergy(state);

        let html = `
            <div>t = ${state.time.toFixed(2)} s</div>
            <div>θ = ${thetaDeg}° <span style="opacity:0.5">(상체)</span></div>
            <div>α = ${alphaDeg}° <span style="opacity:0.5">(하체)</span></div>
            <div>φ = ${phiDeg}° <span style="opacity:0.5">(발)</span></div>
            <div>힙각 = ${hipDeg}°</div>
            <div>τ = ${model.tau.toFixed(0)} N·m</div>
            <div>CoM오차 = ${comErr} cm</div>
            <div>E = ${energy.total.toFixed(1)} J</div>
        `;

        if (controller && controller.getDelayInfo) {
            const d = controller.getDelayInfo();
            if (d.useEstimation) {
                html += `<div style="color:#ffbe0b;">📉 φ 파이 관측: IMU 역기구학 추정</div>`;
            } else {
                html += `<div style="color:#4cc9f0;">🎯 φ 파이 관측: 완벽한 이상적 센싱</div>`;
            }
        }

        el.innerHTML = html;
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
                renderer3d.rebuild();
                if (controller.augmented) {
                    controller.computeAugmentedGains(SL.Params, controller.totalDelayMs);
                }
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

        // === 발위치추정 모드 버튼 ===
        const btnEstIdeal = document.getElementById('btn-est-ideal');
        const btnEstImu = document.getElementById('btn-est-imu');
        const estDesc = document.getElementById('est-desc');
        
        function setEstMode(useEst) {
            controller.setEstimationMode(useEst);
            if (useEst) {
                btnEstIdeal.classList.remove('btn-primary');
                btnEstIdeal.style.background = '';
                btnEstImu.classList.add('btn-primary');
                btnEstImu.style.background = '#ffbe0b';
                btnEstImu.style.color = '#000';
                if(estDesc) estDesc.textContent = 'IMU 추정: 정적 질량중심(CoM) 역기구학 모델로 알파/세타에서 파이를 추정합니다.';
            } else {
                btnEstImu.classList.remove('btn-primary');
                btnEstImu.style.background = '';
                btnEstImu.style.color = '';
                btnEstIdeal.classList.add('btn-primary');
                btnEstIdeal.style.background = '#4cc9f0';
                btnEstIdeal.style.color = '#000';
                if(estDesc) estDesc.textContent = '완벽한 센싱: 파이(φ)를 오류나 딜레이 없이 정확하게 측정한다고 가정합니다.';
            }
        }

        if (btnEstIdeal) btnEstIdeal.addEventListener('click', () => setEstMode(false));
        if (btnEstImu) btnEstImu.addEventListener('click', () => setEstMode(true));

        // === 3D 추가 파라미터 슬라이더 ===
        const lpSlider = document.getElementById('slider-L_pole');
        const lpDisplay = document.getElementById('val-L_pole');
        if (lpSlider && lpDisplay) {
            lpSlider.addEventListener('input', () => {
                const val = parseFloat(lpSlider.value);
                SL.Params.L_pole = val;
                lpDisplay.textContent = val.toFixed(1);
                renderer3d.rebuild();
            });
        }

        const l0Slider = document.getElementById('slider-L0');
        const l0Display = document.getElementById('val-L0');
        if (l0Slider && l0Display) {
            l0Slider.addEventListener('input', () => {
                const val = parseFloat(l0Slider.value);
                SL.Params.L0 = val;
                l0Display.textContent = val.toFixed(1);
            });
        }

        const bdSlider = document.getElementById('slider-bodyDepth');
        const bdDisplay = document.getElementById('val-bodyDepth');
        if (bdSlider && bdDisplay) {
            bdSlider.addEventListener('input', () => {
                const val = parseFloat(bdSlider.value);
                SL.Params.bodyDepth = val;
                bdDisplay.textContent = val.toFixed(2);
                renderer3d.rebuild();
            });
        }

        const arcCheck = document.getElementById('check-showArc');
        if (arcCheck) {
            arcCheck.addEventListener('change', (e) => {
                SL.Params.showArc = e.target.checked;
                renderer3d.rebuild();
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
            graphRenderer.clearData();
            renderer3d.update(model.state, model);
            graphRenderer.draw();
            updateInfoOverlay(model.state, model, controller);
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
        graphRenderer.pushData(
            model.state.time,
            model.state.phi,
            model.state.alpha,
            model.state.theta,
            model.tau
        );
        renderer3d.update(model.state, model);
        graphRenderer.draw();
        updateInfoOverlay(model.state, model, controller);
        animFrameId = requestAnimationFrame(loop);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

/**
 * Slackline Balance Simulator V8 — Main Entry Point (3D + Observer)
 * 
 * V5의 2D 물리 엔진 + V8 칼만 필터 상태관측기
 * Three.js 3D 렌더러로 시각화
 */
var SL = SL || {};

(function () {
    'use strict';

    let model, controller, observer, renderer3d, graphRenderer;
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
        controller.setEstimationMode('ideal');

        // 옵저버 생성 및 초기화
        observer = new SL.StateObserver();
        observer.init(SL.Params, controller, model);
        observer.resetState({ phi: SL.Params.phi0, alpha: SL.Params.alpha0, theta: SL.Params.theta0 });
        controller.setObserver(observer);

        // 3D 렌더러
        const container3d = document.getElementById('container-3d');
        renderer3d = new SL.Renderer3D(container3d, SL.Params);

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

        console.log('Slackline Balance Simulator V8 (State Observer) initialized');
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
            html += `<div style="color:${d.estimationMode === 'observer' ? '#22d3ee' : d.estimationMode === 'kinematic' ? '#ffbe0b' : '#4cc9f0'};">${d.sensorMs}</div>`;
            
            // 옵저버 추정 오차 표시
            if (d.estimationMode === 'observer' && observer && observer.initialized) {
                const eInfo = observer.getEstimationInfo();
                const realPhi = (state.phi * 180 / Math.PI).toFixed(2);
                const estPhi = (eInfo.phiEst * 180 / Math.PI).toFixed(2);
                const err = ((state.phi - eInfo.phiEst) * 180 / Math.PI).toFixed(3);
                html += `<div style="font-size:0.85em; opacity:0.7;">φₘ: ${realPhi}° / φ̂: ${estPhi}°</div>`;
                html += `<div style="font-size:0.85em; color:#f472b6;">Δφ = ${err}°</div>`;
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

        // 지연 시간 슬라이더
        const latSlider = document.getElementById('slider-latency');
        const latDisplay = document.getElementById('val-latency');
        const augCheck = document.getElementById('check-augmented');
        if (latSlider && latDisplay) {
            latSlider.addEventListener('input', () => {
                const val = parseFloat(latSlider.value);
                latDisplay.textContent = val;
                controller.setDelay(val, 1);
            });
        }
        if (augCheck) {
            augCheck.addEventListener('change', (e) => {
                controller.setAugmented(e.target.checked, SL.Params);
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

        // === 발위치 추정 모드 버튼 (V8: 3모드) ===
        const btnEstIdeal = document.getElementById('btn-est-ideal');
        const btnEstKin = document.getElementById('btn-est-kinematic');
        const btnEstObs = document.getElementById('btn-est-observer');
        const estDesc = document.getElementById('est-desc');
        
        const estButtons = [btnEstIdeal, btnEstKin, btnEstObs];
        const estModes = ['ideal', 'kinematic', 'observer'];
        const estColors = ['#4cc9f0', '#ffbe0b', '#22d3ee'];
        const estDescs = [
            '완벽한 센싱: φ를 오류 없이 정확하게 측정한다고 가정합니다.',
            'CoM 역기구학: X_com≈0 가정하여 α/θ로 φ를 역산합니다. (구조적 한계)',
            '칼만 필터: 운동방정식 적분 + IMU 오차 보정으로 φ를 동적 추정합니다.'
        ];

        function setEstMode(idx) {
            controller.setEstimationMode(estModes[idx]);
            if (estModes[idx] === 'observer' && observer) {
                observer.resetState(model.state);
            }
            estButtons.forEach((btn, i) => {
                if (!btn) return;
                if (i === idx) {
                    btn.classList.add('btn-primary');
                    btn.style.background = estColors[i];
                    btn.style.color = '#000';
                } else {
                    btn.classList.remove('btn-primary');
                    btn.style.background = '';
                    btn.style.color = '';
                }
            });
            if (estDesc) estDesc.textContent = estDescs[idx];
        }

        estButtons.forEach((btn, i) => {
            if (btn) btn.addEventListener('click', () => setEstMode(i));
        });

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
            if (observer) {
                observer.resetState(model.state);
            }
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

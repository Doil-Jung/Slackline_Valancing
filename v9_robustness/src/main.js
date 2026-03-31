/**
 * Slackline Balance Simulator V9 — Main Entry Point (Robustness Test)
 * 
 * V8 기반 + 모델 불확실성 & 센서 노이즈 시뮬레이션
 * 
 * 핵심: 물리 엔진은 plantParams, 제어기/옵저버는 nominalParams 사용
 */
var SL = SL || {};

(function () {
    'use strict';

    let model, controller, observer, renderer3d, graphRenderer;
    let paramManager, sensorNoise;
    let running = false;
    let animFrameId = null;

    // 3D 추가 파라미터 기본값 설정
    SL.Params.L_pole = 4.0;
    SL.Params.L0 = 5.0;
    SL.Params.bodyDepth = 0.25;
    SL.Params.showArc = true;

    /** 초기화 */
    function init() {
        // 파라미터 매니저 생성 & 공칭값 스냅샷
        paramManager = new SL.ParamManager();
        paramManager.captureNominal();

        // 센서 노이즈 생성
        sensorNoise = new SL.SensorNoise();
        sensorNoise.init(SL.Params.dt);

        // 물리 모델 (플랜트 파라미터 사용 — 현재는 편차 0이므로 공칭과 동일)
        model = new SL.Model(SL.Params);

        // 제어기 (내부적으로 공칭 파라미터 기반 게인 사용)
        controller = new SL.LQRController();
        controller.setEstimationMode('ideal');
        controller.setSensorNoise(sensorNoise);

        // 옵저버 (공칭 모델 기반으로 내부 RK4 예측)
        // 옵저버는 nominalParams 기반의 별도 모델 인스턴스가 필요
        const nominalModel = new SL.Model(paramManager.nominalParams);
        observer = new SL.StateObserver();
        observer.init(paramManager.nominalParams, controller, nominalModel);
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
        updateInfoOverlay(model.state, model, controller);

        console.log('Slackline Balance Simulator V9 (Robustness Test) initialized');
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
            
            // 지연시간 표시
            if (d.totalMs > 0) {
                html += `<div style="font-size:0.85em; color:#ff6b6b;">⏱ 지연: ${d.totalMs}ms${d.augmented ? ' (보상)' : ''}</div>`;
            }

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

        // V9: 모델 편차 & 노이즈 상태
        const mismatchStr = paramManager.getMismatchSummary();
        const noiseStr = sensorNoise.getNoiseSummary();
        html += `<div style="margin-top:4px; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px;">`;
        html += `<div style="font-size:0.8em; color:${mismatchStr === '편차 없음' ? 'rgba(255,255,255,0.3)' : '#fb923c'};">🔧 ${mismatchStr}</div>`;
        html += `<div style="font-size:0.8em; color:${noiseStr === '노이즈 없음' ? 'rgba(255,255,255,0.3)' : '#a78bfa'};">📡 ${noiseStr}</div>`;
        html += `</div>`;

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
                // 공칭값 재캡처 후 편차 재적용
                paramManager.captureNominal();
                applyMismatch();
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
        if (latSlider && latDisplay) {
            latSlider.addEventListener('input', () => {
                const val = parseFloat(latSlider.value);
                latDisplay.textContent = val;
                controller.setDelay(val, 1);
            });
        }
        const augCheck = document.getElementById('check-augmented');
        if (augCheck) {
            augCheck.addEventListener('change', (e) => {
                controller.setAugmented(e.target.checked, paramManager.nominalParams);
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
            'CoM 역기구학: X_com≈0 가정하여 α/θ로 φ를 역산합니다.',
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

        // === V9: 모델 불확실성 슬라이더 ===
        ['m1', 'm2', 'L1', 'L2', 'R'].forEach(key => {
            const slider = document.getElementById(`slider-mismatch-${key}`);
            const display = document.getElementById(`val-mismatch-${key}`);
            if (!slider || !display) return;

            slider.addEventListener('input', () => {
                const pct = parseFloat(slider.value);
                display.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(0) + '%';
                display.style.color = pct === 0 ? 'rgba(255,255,255,0.5)' : (pct > 0 ? '#fb923c' : '#38bdf8');
                paramManager.setMismatch(key, pct / 100);
                applyMismatch();
            });
        });

        // === V9: 센서 노이즈 슬라이더 ===
        const angleNoiseSlider = document.getElementById('slider-angleNoise');
        const angleNoiseDisplay = document.getElementById('val-angleNoise');
        if (angleNoiseSlider && angleNoiseDisplay) {
            angleNoiseSlider.addEventListener('input', () => {
                const val = parseFloat(angleNoiseSlider.value);
                const rad = val * Math.PI / 180;
                sensorNoise.angleNoiseSigma = rad;
                angleNoiseDisplay.textContent = val.toFixed(2) + '°';
            });
        }

        const gyroNoiseSlider = document.getElementById('slider-gyroNoise');
        const gyroNoiseDisplay = document.getElementById('val-gyroNoise');
        if (gyroNoiseSlider && gyroNoiseDisplay) {
            gyroNoiseSlider.addEventListener('input', () => {
                const val = parseFloat(gyroNoiseSlider.value);
                sensorNoise.gyroNoiseSigma = val;
                gyroNoiseDisplay.textContent = val.toFixed(3);
            });
        }

        const biasDriftSlider = document.getElementById('slider-biasDrift');
        const biasDriftDisplay = document.getElementById('val-biasDrift');
        if (biasDriftSlider && biasDriftDisplay) {
            biasDriftSlider.addEventListener('input', () => {
                const val = parseFloat(biasDriftSlider.value);
                sensorNoise.gyroBiasDrift = val;
                biasDriftDisplay.textContent = val.toFixed(2);
            });
        }

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

    /** 모델 편차 적용: 플랜트 파라미터를 SL.Params에 반영 */
    function applyMismatch() {
        paramManager.applyPlantToGlobal();
        // 물리 모델은 SL.Params를 직접 참조하므로 자동 반영
        // 렌더러 리빌드
        renderer3d.rebuild();
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
            // 공칭값 재캡처 후 편차 재적용
            paramManager.captureNominal();
            applyMismatch();
            model.reset(SL.Params);
            controller.reset();
            sensorNoise.reset();
            if (observer) {
                // 옵저버를 공칭 모델로 재초기화
                const nominalModel = new SL.Model(paramManager.nominalParams);
                observer.init(paramManager.nominalParams, controller, nominalModel);
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

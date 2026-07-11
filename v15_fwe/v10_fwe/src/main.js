/**
 * Slackline Balance Simulator V10 — Main Entry Point (Adaptive Control)
 * 
 * V9 기반 + LQR R/Q 튜닝 + 온라인 시스템 식별 (RLS)
 * 
 * 핵심: R 스케일로 게인 강도 조절 + RLS로 실제 파라미터 추정
 */
var SL = SL || {};

(function () {
    'use strict';

    let model, controller, observer, renderer3d, graphRenderer;
    let paramManager, sensorNoise, sysid;
    let running = false;
    let animFrameId = null;
    let adaptedParams = null; // V10: 식별 후 적용된 파라미터 기록용
    let simSpeed = 1.0; // 재생 속도 배율
    let stateHistory = []; // 되감기용 상태 기록
    const MAX_HISTORY = 2000;

    // 3D 추가 파라미터 기본값 설정
    SL.Params.L_pole = 4.0;
    SL.Params.L0 = 5.0;
    SL.Params.bodyDepth = 0.25;
    SL.Params.showArc = true;
    SL.Params.showPoles = false;

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

        // 제어기 (FWE 제어기)
        controller = new SL.FWEController();
        controller.nominalParams = paramManager.nominalParams;
        controller.setEstimationMode('ideal');
        controller.setSensorNoise(sensorNoise);

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
        updateInfoOverlay(model.state, model, controller);
        renderer3d.setSideView(); // 기본 XZ 사이드 뷰

        console.log('Slackline Balance Simulator V15 (FWE Control) initialized');
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

        if (controller) {
            const d = controller.getDelayInfo ? controller.getDelayInfo() : {};
            const phase = controller.phase || 'IDLE';
            const deltaTargetDeg = ((controller.delta_target || 0) * 180 / Math.PI).toFixed(1);
            
            // FWE 상태머신 페이즈 강조 표시
            let phaseColor = '#ff6b6b';
            if (phase === 'IDLE') phaseColor = '#e0e0e0';
            else if (phase === 'FOLD') phaseColor = '#ff006e';
            else if (phase === 'WAIT1') phaseColor = '#06d6a0';
            else if (phase === 'EXTEND') phaseColor = '#ffbe0b';
            else if (phase === 'WAIT2') phaseColor = '#8338ec';

            html += `<div style="margin-top:4px; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px;">`;
            html += `<div style="font-size:0.95em; font-weight:bold; color:${phaseColor};">페이즈: ${phase}</div>`;
            html += `<div style="font-size:0.85em;">목표 힙각: ${deltaTargetDeg}°</div>`;
            if (controller.cycle_count !== undefined) {
                html += `<div style="font-size:0.85em; opacity:0.7;">동작 사이클: ${controller.cycle_count}회</div>`;
            }
            if (d.sensorMs) {
                html += `<div style="font-size:0.85em; color:#4cc9f0;">센싱: ${d.sensorMs}</div>`;
            }
            html += `</div>`;
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

        // === FWE 파라미터 슬라이더 바인딩 ===
        const fweSliders = [
            { id: 'slider-fwe-eps', valId: 'val-fwe-eps', prop: 'eps', scale: 1, precision: 1 },
            { id: 'slider-fwe-Tf', valId: 'val-fwe-Tf', prop: 'T_f', scale: 0.001, precision: 0 },
            { id: 'slider-fwe-Tw1', valId: 'val-fwe-Tw1', prop: 'T_w1', scale: 0.001, precision: 0 },
            { id: 'slider-fwe-betaThresh', valId: 'val-fwe-betaThresh', prop: 'beta_threshold', scale: Math.PI / 180, precision: 2 },
            { id: 'slider-fwe-cpred', valId: 'val-fwe-cpred', prop: 'c_pred', scale: 1, precision: 1 },
            { id: 'slider-fwe-cbetadot', valId: 'val-fwe-cbetadot', prop: 'c_beta_dot', scale: 1, precision: 2 },
            { id: 'slider-fwe-maxdelta', valId: 'val-fwe-maxdelta', prop: 'max_delta', scale: Math.PI / 180, precision: 0 },
            { id: 'slider-fwe-Kp', valId: 'val-fwe-Kp', prop: 'Kp_servo', scale: 1, precision: 0 },
            { id: 'slider-fwe-Kd', valId: 'val-fwe-Kd', prop: 'Kd_servo', scale: 1, precision: 0 }
        ];

        fweSliders.forEach(s => {
            const slider = document.getElementById(s.id);
            const display = document.getElementById(s.valId);
            if (!slider || !display) return;

            // 초기값 설정
            let initVal = controller[s.prop];
            if (s.prop === 'T_f' || s.prop === 'T_w1') initVal = Math.round(initVal * 1000);
            else if (s.prop === 'beta_threshold' || s.prop === 'max_delta') initVal = initVal * 180 / Math.PI;
            slider.value = initVal;
            display.textContent = Number(initVal).toFixed(s.precision);

            slider.addEventListener('input', () => {
                const val = parseFloat(slider.value);
                display.textContent = val.toFixed(s.precision);
                controller[s.prop] = val * s.scale;
            });
        });
        
        // === V15 추가 제어 옵션 체크박스 바인딩 ===
        const checkSkipWait2OnExtendZero = document.getElementById('check-fwe-skipWait2OnExtendZero');
        if (checkSkipWait2OnExtendZero) {
            checkSkipWait2OnExtendZero.addEventListener('change', (e) => {
                controller.skipWait2OnExtendZero = e.target.checked;
            });
            controller.skipWait2OnExtendZero = checkSkipWait2OnExtendZero.checked;
        }

        const checkSkipWait2Entirely = document.getElementById('check-fwe-skipWait2Entirely');
        if (checkSkipWait2Entirely) {
            checkSkipWait2Entirely.addEventListener('change', (e) => {
                controller.skipWait2Entirely = e.target.checked;
            });
            controller.skipWait2Entirely = checkSkipWait2Entirely.checked;
        }

        // === 발위치 추정 모드 버튼 (V15: 2모드) ===
        const btnEstIdeal = document.getElementById('btn-est-ideal');
        const btnEstKin = document.getElementById('btn-est-kinematic');
        const estDesc = document.getElementById('est-desc');
        
        const estButtons = [btnEstIdeal, btnEstKin];
        const estModes = ['ideal', 'kinematic'];
        const estColors = ['#4cc9f0', '#ffbe0b'];
        const estDescs = [
            '완벽한 센싱: φ를 오류 없이 정확하게 측정한다고 가정합니다.',
            'CoM 역기구학: X_com≈0 가정하여 α/θ로 φ를 역산합니다.'
        ];

        function setEstMode(idx) {
            controller.setEstimationMode(estModes[idx]);
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

        const polesCheck = document.getElementById('check-showPoles');
        if (polesCheck) {
            polesCheck.addEventListener('change', (e) => {
                SL.Params.showPoles = e.target.checked;
                renderer3d.rebuild();
            });
        }
    }

    /** 모델 편차 적용: 플랜트 파라미터를 SL.Params에 반영 */
    function applyMismatch() {
        paramManager.applyPlantToGlobal();
        if (controller) {
            controller.nominalParams = paramManager.nominalParams;
        }
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
            const chkZero = document.getElementById('check-fwe-skipWait2OnExtendZero');
            if (chkZero) controller.skipWait2OnExtendZero = chkZero.checked;
            const chkSkip = document.getElementById('check-fwe-skipWait2Entirely');
            if (chkSkip) controller.skipWait2Entirely = chkSkip.checked;
            const chkPoles = document.getElementById('check-showPoles');
            if (chkPoles) SL.Params.showPoles = chkPoles.checked;
            
            sensorNoise.reset();
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

        // 뷰 전환 버튼
        const btnViewSide = document.getElementById('btn-view-side');
        const btnView3D = document.getElementById('btn-view-3d');
        if (btnViewSide) btnViewSide.addEventListener('click', () => renderer3d.setSideView());
        if (btnView3D) btnView3D.addEventListener('click', () => renderer3d.set3DView());

        // 프레임 스텝 버튼
        const btnStep = document.getElementById('btn-step');
        if (btnStep) btnStep.addEventListener('click', () => stepOnce());
        const btnStepBack = document.getElementById('btn-step-back');
        if (btnStepBack) btnStepBack.addEventListener('click', () => stepBack());

        // 재생 속도 슬라이더
        const speedSlider = document.getElementById('slider-speed');
        const speedDisplay = document.getElementById('val-speed');
        if (speedSlider && speedDisplay) {
            speedSlider.addEventListener('input', () => {
                simSpeed = parseFloat(speedSlider.value);
                SL.Params.speedMultiplier = simSpeed;
                speedDisplay.textContent = simSpeed.toFixed(2) + 'x';
            });
        }
    }

    function start() { if (running) return; running = true; loop(); }
    function stop() {
        running = false;
        if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    }

    function loop() {
        if (!running) return;
        model.step(controller);

        // beta & delta 계산
        const p = model.params;
        const m1 = p.m1, m2 = p.m2;
        const L1 = p.L1, L2 = p.L2;
        const ell1 = L1 / 2, ell2 = L2 / 2;
        const Mt = m1 + m2;
        const h_com = (m1 * ell1 + m2 * (L1 + ell2)) / Mt;
        const p1 = m1 * ell1 + m2 * L1;
        const p2 = m2 * ell2;
        const R = p.R;
        const x_com = R * Math.sin(model.state.phi)
                     + (p1 * Math.sin(model.state.alpha) + p2 * Math.sin(model.state.theta)) / Mt;
        const beta = x_com / h_com;
        const delta = model.state.theta - model.state.alpha;

        graphRenderer.pushData(
            model.state.time,
            model.state.phi,
            model.state.alpha,
            model.state.theta,
            beta,
            delta,
            model.tau,
            controller ? controller.phase : 'IDLE'
        );
        renderer3d.update(model.state, model);
        graphRenderer.draw();
        updateInfoOverlay(model.state, model, controller);
        animFrameId = requestAnimationFrame(loop);
    }

    /** 프레임 1장 진행 (정지 상태에서 사용) */
    function stepOnce() {
        if (running) return;
        // 현재 상태를 히스토리에 저장 (되감기용)
        stateHistory.push({
            state: Object.assign({}, model.state),
            phase: controller ? controller.phase : 'IDLE',
            t_phase: controller ? controller.t_phase : 0,
            d_fold: controller ? controller.d_fold : 0,
            fold_sign: controller ? controller.fold_sign : 1,
            delta_target: controller ? controller.delta_target : 0,
            beta_prev: controller ? controller.beta_prev : null,
            phi_prev: controller ? controller.phi_prev : null,
            cycle_count: controller ? controller.cycle_count : 0,
            graphLen: graphRenderer.graphData.length
        });
        if (stateHistory.length > MAX_HISTORY) stateHistory.shift();

        const origSpeed = SL.Params.speedMultiplier;
        SL.Params.speedMultiplier = simSpeed;
        model.step(controller);
        SL.Params.speedMultiplier = origSpeed;

        const p = model.params;
        const m1 = p.m1, m2 = p.m2;
        const L1 = p.L1, L2 = p.L2;
        const ell1 = L1 / 2, ell2 = L2 / 2;
        const Mt = m1 + m2;
        const h_com = (m1 * ell1 + m2 * (L1 + ell2)) / Mt;
        const p1 = m1 * ell1 + m2 * L1;
        const p2 = m2 * ell2;
        const R = p.R;
        const x_com = R * Math.sin(model.state.phi)
                     + (p1 * Math.sin(model.state.alpha) + p2 * Math.sin(model.state.theta)) / Mt;
        const beta = x_com / h_com;
        const delta = model.state.theta - model.state.alpha;

        graphRenderer.pushData(
            model.state.time, model.state.phi, model.state.alpha, model.state.theta,
            beta, delta, model.tau, controller ? controller.phase : 'IDLE'
        );
        renderer3d.update(model.state, model);
        graphRenderer.draw();
        updateInfoOverlay(model.state, model, controller);
    }

    /** 프레임 1장 되감기 */
    function stepBack() {
        if (running || stateHistory.length === 0) return;
        const snap = stateHistory.pop();
        // 모델 상태 복원
        model.state = Object.assign({}, snap.state);
        // 컨트롤러 상태 복원
        if (controller) {
            controller.phase = snap.phase;
            controller.t_phase = snap.t_phase;
            controller.d_fold = snap.d_fold;
            controller.fold_sign = snap.fold_sign;
            controller.delta_target = snap.delta_target;
            controller.beta_prev = snap.beta_prev;
            controller.phi_prev = snap.phi_prev;
            controller.cycle_count = snap.cycle_count;
        }
        // 그래프 데이터 되감기
        while (graphRenderer.graphData.length > snap.graphLen) {
            graphRenderer.graphData.pop();
        }
        renderer3d.update(model.state, model);
        graphRenderer.draw();
        updateInfoOverlay(model.state, model, controller);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

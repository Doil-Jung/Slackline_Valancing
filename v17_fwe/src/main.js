/**
 * Slackline Balance Simulator V17 — Main Entry Point
 * 
 * 식 4(제4 메인 식) 기반 FWE 제어기 + v10 시뮬레이터 통합
 * 깔끔한 5개 파라미터: k₁, β_threshold, max_δ, Kp_servo, Kd_servo
 */
var SL = SL || {};

(function () {
    'use strict';

    let model, controller, renderer3d, graphRenderer;
    let paramManager, sensorNoise;
    let running = false;
    let animFrameId = null;
    let simSpeed = 1.0;
    let stateHistory = [];
    const MAX_HISTORY = 2000;

    // 3D 추가 파라미터
    SL.Params.L_pole = 4.0;
    SL.Params.L0 = 5.0;
    SL.Params.bodyDepth = 0.25;
    SL.Params.showArc = true;
    SL.Params.showPoles = false;

    /** 초기화 */
    function init() {
        paramManager = new SL.ParamManager();
        paramManager.captureNominal();

        sensorNoise = new SL.SensorNoise();
        sensorNoise.init(SL.Params.dt);

        model = new SL.Model(SL.Params);

        controller = new SL.FWEController();
        controller.nominalParams = paramManager.nominalParams;
        controller.setEstimationMode('ideal');
        controller.setSensorNoise(sensorNoise);

        const container3d = document.getElementById('container-3d');
        renderer3d = new SL.Renderer3D(container3d, SL.Params);
        renderer3d.startIdleRender();

        const graphCanvas = document.getElementById('graph-canvas');
        graphRenderer = new SL.GraphRenderer(graphCanvas);

        setupFWESliders();
        setupPhysicsSliders();
        setupMiscSliders();
        setupButtons();

        window.addEventListener('resize', () => {
            renderer3d.resize();
            graphRenderer.resize();
        });

        renderer3d.update(model.state, model);
        graphRenderer.draw();
        updateInfoOverlay(model.state, model, controller);
        renderer3d.setSideView();

        console.log('Slackline Balance Simulator V17 (True Equation 4 FWE) initialized');
    }

    /** 정보 오버레이 */
    function updateInfoOverlay(state, model, controller) {
        const el = document.getElementById('info-overlay');
        if (!el) return;

        const toDeg = (r) => (r * 180 / Math.PI).toFixed(2);
        const comErr = (model.getCoMError(state) * 100).toFixed(1);
        const energy = model.getEnergy(state);

        // β 계산 (전체 CoM 기울기)
        const p = model.params;
        const m1 = p.m1, m2 = p.m2, L1 = p.L1, L2 = p.L2;
        const ell1 = L1 / 2, ell2 = L2 / 2, Mt = m1 + m2;
        const h_com = (m1 * ell1 + m2 * (L1 + ell2)) / Mt;
        const pp1 = m1 * ell1 + m2 * L1;
        const pp2 = m2 * ell2;
        // β 계산 (발 기준 상대 CoM 기울기 — 힙토크는 내력이므로 phi 항 제외)
        const x_com_rel = (pp1 * Math.sin(state.alpha) + pp2 * Math.sin(state.theta)) / Mt;
        const beta = x_com_rel / h_com;
        const delta = state.theta - state.alpha;

        let html = `
            <div>t = ${state.time.toFixed(2)} s</div>
            <div style="color:#3a86ff">φ = ${toDeg(state.phi)}° <span style="opacity:0.5">(발)</span></div>
            <div style="color:#fb7185">β = ${toDeg(beta)}° <span style="opacity:0.5">(기울기)</span></div>
            <div>CoM오차 = ${comErr} cm</div>
            <div style="color:#c084fc">δ = ${toDeg(delta)}° <span style="opacity:0.5">(접힘)</span></div>
            <div style="color:#8338ec">θ = ${toDeg(state.theta)}° <span style="opacity:0.5">(상체)</span></div>
            <div style="color:#06d6a0">α = ${toDeg(state.alpha)}° <span style="opacity:0.5">(하체)</span></div>
            <div style="color:#ffbe0b">τ = ${model.tau.toFixed(2)} N·m</div>
            <div style="opacity:0.5">E = ${energy.total.toFixed(1)} J</div>
        `;

        if (controller) {
            const phase = controller.phase || 'IDLE';
            const deltaTargetDeg = ((controller.delta_target || 0) * 180 / Math.PI).toFixed(1);

            const phaseColors = {
                'IDLE': '#e0e0e0', 'FOLD': '#ff006e',
                'WAIT1': '#06d6a0', 'EXTEND': '#ffbe0b'
            };

            html += `<div style="margin-top:4px; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px;">`;
            html += `<div style="font-size:0.95em; font-weight:bold; color:${phaseColors[phase] || '#e0e0e0'};">페이즈: ${phase}</div>`;
            html += `<div style="font-size:0.85em;">목표 힙각: ${deltaTargetDeg}°</div>`;
            if (controller.cycle_count !== undefined) {
                html += `<div style="font-size:0.85em; opacity:0.7;">사이클: ${controller.cycle_count}회</div>`;
            }
            if (controller.T_f_star !== undefined) {
                html += `<div style="font-size:0.85em; color:#4cc9f0;">T_f*=${(controller.T_f_star * 1000).toFixed(0)}ms  T_w*=${(controller.T_w_star * 1000).toFixed(0)}ms</div>`;
                html += `<div style="font-size:0.8em; color:${controller.solver_converged ? '#4ade80' : '#ff6b6b'};">솔버: ${controller.solver_converged ? '수렴 ✓' : '실패 ✗'}</div>`;
            }
            html += `</div>`;
        }

        const mismatchStr = paramManager.getMismatchSummary();
        const noiseStr = sensorNoise.getNoiseSummary();
        html += `<div style="margin-top:4px; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px;">`;
        html += `<div style="font-size:0.8em; color:${mismatchStr === '편차 없음' ? 'rgba(255,255,255,0.3)' : '#fb923c'};">🔧 ${mismatchStr}</div>`;
        html += `<div style="font-size:0.8em; color:${noiseStr === '노이즈 없음' ? 'rgba(255,255,255,0.3)' : '#a78bfa'};">📡 ${noiseStr}</div>`;
        html += `</div>`;

        el.innerHTML = html;
    }

    // ================================================================
    //  FWE 슬라이더 (식 4 파라미터 5개)
    // ================================================================
    function setupFWESliders() {
        const sliders = [
            { id: 'slider-fwe-k1',         valId: 'val-fwe-k1',         prop: 'k1',             scale: 1,             precision: 0 },
            { id: 'slider-fwe-betaThresh', valId: 'val-fwe-betaThresh', prop: 'beta_threshold',  scale: Math.PI / 180, precision: 3 },
            { id: 'slider-fwe-maxdelta',   valId: 'val-fwe-maxdelta',   prop: 'max_delta',       scale: Math.PI / 180, precision: 0 },
            { id: 'slider-fwe-Tf',         valId: 'val-fwe-Tf',         prop: 'T_f_fixed',       scale: 0.001,         precision: 0 },
            { id: 'slider-fwe-Kp',         valId: 'val-fwe-Kp',         prop: 'Kp_servo',        scale: 1,             precision: 0 },
            { id: 'slider-fwe-Kd',         valId: 'val-fwe-Kd',         prop: 'Kd_servo',        scale: 1,             precision: 0 }
        ];

        sliders.forEach(s => {
            const slider = document.getElementById(s.id);
            const display = document.getElementById(s.valId);
            if (!slider || !display) return;

            // 초기값 (controller → slider 방향으로 동기화)
            let displayVal = controller[s.prop];
            if (s.prop === 'beta_threshold' || s.prop === 'max_delta') displayVal = displayVal * 180 / Math.PI;
            else if (s.prop === 'T_f_fixed') displayVal = displayVal * 1000; // s → ms
            slider.value = displayVal;
            display.textContent = Number(displayVal).toFixed(s.precision);

            slider.addEventListener('input', () => {
                const val = parseFloat(slider.value);
                display.textContent = val.toFixed(s.precision);
                controller[s.prop] = val * s.scale;
                console.log(`[V17 UI] ${s.prop} = ${controller[s.prop]}`);
            });
        });
    }

    // ================================================================
    //  물리 파라미터 슬라이더
    // ================================================================
    function setupPhysicsSliders() {
        const ranges = SL.Params.ranges;
        Object.keys(ranges).forEach(key => {
            const slider = document.getElementById(`slider-${key}`);
            const display = document.getElementById(`val-${key}`);
            if (!slider || !display) return;

            const range = ranges[key];
            const scale = range.scale || 1;  // deg→rad 등 변환 계수
            slider.min = range.min;
            slider.max = range.max;
            slider.step = range.step;
            // 내부값(rad) → 표시값(deg) 변환
            slider.value = SL.Params[key] / scale;
            const precision = range.step < 1 ? (range.step < 0.1 ? 2 : 1) : 0;
            display.textContent = Number(slider.value).toFixed(precision);

            slider.addEventListener('input', () => {
                const val = parseFloat(slider.value);
                SL.Params[key] = val * scale;  // 표시값 → 내부값 변환
                display.textContent = val.toFixed(precision);
                renderer3d.rebuild();
                paramManager.captureNominal();
                applyMismatch();
            });
        });
    }

    // ================================================================
    //  기타 슬라이더 (교란, 속도, 3D, 노이즈, 불확실성)
    // ================================================================
    function setupMiscSliders() {
        // 교란 강도
        const psSlider = document.getElementById('slider-perturbStr');
        const psDisplay = document.getElementById('val-perturbStr');
        if (psSlider && psDisplay) {
            psSlider.addEventListener('input', () => { psDisplay.textContent = parseFloat(psSlider.value).toFixed(1); });
        }

        // 재생 속도
        const speedSlider = document.getElementById('slider-speed');
        const speedDisplay = document.getElementById('val-speed');
        if (speedSlider && speedDisplay) {
            speedSlider.addEventListener('input', () => {
                simSpeed = parseFloat(speedSlider.value);
                SL.Params.speedMultiplier = simSpeed;
                speedDisplay.textContent = simSpeed.toFixed(2) + 'x';
            });
        }

        // 발위치 추정 모드
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
        estButtons.forEach((btn, i) => { if (btn) btn.addEventListener('click', () => setEstMode(i)); });

        // 모델 불확실성
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

        // 센서 노이즈
        const noiseSliders = [
            { sid: 'slider-angleNoise', did: 'val-angleNoise', prop: 'angleNoiseSigma', toRad: Math.PI / 180, fmt: (v) => v.toFixed(2) + '°' },
            { sid: 'slider-gyroNoise',  did: 'val-gyroNoise',  prop: 'gyroNoiseSigma',  toRad: 1, fmt: (v) => v.toFixed(3) },
            { sid: 'slider-biasDrift',  did: 'val-biasDrift',  prop: 'gyroBiasDrift',   toRad: 1, fmt: (v) => v.toFixed(2) }
        ];
        noiseSliders.forEach(ns => {
            const slider = document.getElementById(ns.sid);
            const display = document.getElementById(ns.did);
            if (!slider || !display) return;
            slider.addEventListener('input', () => {
                const val = parseFloat(slider.value);
                sensorNoise[ns.prop] = val * ns.toRad;
                display.textContent = ns.fmt(val);
            });
        });

        // 3D 파라미터
        const threeDSliders = [
            { sid: 'slider-L_pole',    did: 'val-L_pole',    prop: 'L_pole',    rebuild: true,  fmt: 1 },
            { sid: 'slider-L0',        did: 'val-L0',        prop: 'L0',        rebuild: false, fmt: 1 },
            { sid: 'slider-bodyDepth', did: 'val-bodyDepth', prop: 'bodyDepth', rebuild: true,  fmt: 2 }
        ];
        threeDSliders.forEach(td => {
            const slider = document.getElementById(td.sid);
            const display = document.getElementById(td.did);
            if (!slider || !display) return;
            slider.addEventListener('input', () => {
                const val = parseFloat(slider.value);
                SL.Params[td.prop] = val;
                display.textContent = val.toFixed(td.fmt);
                if (td.rebuild) renderer3d.rebuild();
            });
        });

        const arcCheck = document.getElementById('check-showArc');
        if (arcCheck) arcCheck.addEventListener('change', (e) => { SL.Params.showArc = e.target.checked; renderer3d.rebuild(); });

        const polesCheck = document.getElementById('check-showPoles');
        if (polesCheck) polesCheck.addEventListener('change', (e) => { SL.Params.showPoles = e.target.checked; renderer3d.rebuild(); });
    }

    function applyMismatch() {
        paramManager.applyPlantToGlobal();
        if (controller) controller.nominalParams = paramManager.nominalParams;
        renderer3d.rebuild();
    }

    // ================================================================
    //  버튼 UI
    // ================================================================
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
            const deg2rad = Math.PI / 180;
            SL.Params.alpha0 = parseFloat(document.getElementById('slider-alpha0')?.value || 0.1) * deg2rad;
            SL.Params.theta0 = parseFloat(document.getElementById('slider-theta0')?.value || 0.1) * deg2rad;
            paramManager.captureNominal();
            applyMismatch();
            model.reset(SL.Params);
            controller.reset();
            sensorNoise.reset();
            stateHistory = [];
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

        // 뷰 전환
        const btnViewSide = document.getElementById('btn-view-side');
        const btnView3D = document.getElementById('btn-view-3d');
        if (btnViewSide) btnViewSide.addEventListener('click', () => renderer3d.setSideView());
        if (btnView3D) btnView3D.addEventListener('click', () => renderer3d.set3DView());

        // 프레임 스텝
        const btnStep = document.getElementById('btn-step');
        if (btnStep) btnStep.addEventListener('click', () => stepOnce());
        const btnStepBack = document.getElementById('btn-step-back');
        if (btnStepBack) btnStepBack.addEventListener('click', () => stepBack());
    }

    // ================================================================
    //  시뮬레이션 루프
    // ================================================================
    function start() { if (running) return; running = true; loop(); }
    function stop() {
        running = false;
        if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    }

    function computeBetaDelta(state, model) {
        const p = model.params;
        const m1 = p.m1, m2 = p.m2, L1 = p.L1, L2 = p.L2;
        const ell1 = L1 / 2, ell2 = L2 / 2, Mt = m1 + m2;
        const h_com = (m1 * ell1 + m2 * (L1 + ell2)) / Mt;
        const p1 = m1 * ell1 + m2 * L1;
        const p2 = m2 * ell2;
        const R = p.R;
        // β: 발 기준 상대 CoM 기울기 (phi 항 제외)
        const x_com_rel = (p1 * Math.sin(state.alpha) + p2 * Math.sin(state.theta)) / Mt;
        const beta = x_com_rel / h_com;
        const delta = state.theta - state.alpha;
        return { beta, delta };
    }

    function loop() {
        if (!running) return;
        // 뒤로 간 후 다시 시작한 경우 미래 데이터 제거
        graphRenderer.trimAfter(model.state.time);
        model.step(controller);
        const { beta, delta } = computeBetaDelta(model.state, model);

        graphRenderer.pushData(
            model.state.time, model.state.phi, model.state.alpha, model.state.theta,
            beta, delta, model.tau, controller ? controller.phase : 'IDLE'
        );
        renderer3d.update(model.state, model);
        graphRenderer.draw();
        updateInfoOverlay(model.state, model, controller);
        animFrameId = requestAnimationFrame(loop);
    }

    function stepOnce() {
        if (running) return;
        stateHistory.push({
            state: Object.assign({}, model.state),
            phase: controller.phase, t_phase: controller.t_phase,
            d_fold: controller.d_fold, fold_sign: controller.fold_sign,
            delta_target: controller.delta_target,
            T_f_star: controller.T_f_star, T_w_star: controller.T_w_star,
            cycle_count: controller.cycle_count
        });
        if (stateHistory.length > MAX_HISTORY) stateHistory.shift();

        // 뒤로 간 후 다시 진행하면 미래 데이터 제거
        graphRenderer.trimAfter(model.state.time);

        const origSpeed = SL.Params.speedMultiplier;
        SL.Params.speedMultiplier = simSpeed;
        model.step(controller);
        SL.Params.speedMultiplier = origSpeed;

        const { beta, delta } = computeBetaDelta(model.state, model);
        graphRenderer.pushData(
            model.state.time, model.state.phi, model.state.alpha, model.state.theta,
            beta, delta, model.tau, controller ? controller.phase : 'IDLE'
        );
        renderer3d.update(model.state, model);
        graphRenderer.draw();
        updateInfoOverlay(model.state, model, controller);
    }

    function stepBack() {
        if (running || stateHistory.length === 0) return;
        const snap = stateHistory.pop();
        model.state = Object.assign({}, snap.state);
        if (controller) {
            controller.phase = snap.phase;
            controller.t_phase = snap.t_phase;
            controller.d_fold = snap.d_fold;
            controller.fold_sign = snap.fold_sign;
            controller.delta_target = snap.delta_target;
            controller.T_f_star = snap.T_f_star;
            controller.T_w_star = snap.T_w_star;
            controller.cycle_count = snap.cycle_count;
        }
        // 그래프 데이터는 유지하고 현재 시간 커서만 이동
        graphRenderer.currentTime = model.state.time;
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

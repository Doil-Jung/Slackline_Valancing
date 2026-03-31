/**
 * Slackline Balance Simulator V1 — Main Entry (1-Body)
 */
document.addEventListener('DOMContentLoaded', () => {
    const params = SL.Params;
    const model = new SL.Model(params);
    const controller = new SL.PIDController(params.Kp, params.Kd, params.Ki);

    const simCanvas = document.getElementById('sim-canvas');
    const graphCanvas = document.getElementById('graph-canvas');
    const renderer = new SL.Renderer(simCanvas, graphCanvas, params);

    let isRunning = false;
    let animFrameId = null;

    function setupSliders() {
        Object.keys(params.ranges).forEach(key => {
            const slider = document.getElementById(`slider-${key}`);
            const valDisplay = document.getElementById(`val-${key}`);
            if (!slider) return;

            const r = params.ranges[key];
            slider.min = r.min;
            slider.max = r.max;
            slider.step = r.step;
            slider.value = params[key];

            slider.addEventListener('input', (e) => {
                const v = parseFloat(e.target.value);
                params[key] = v;
                valDisplay.textContent = v.toFixed(r.step < 1 ? 2 : 0);

                if (key === 'Kp' || key === 'Kd' || key === 'Ki') {
                    controller.setGains(params.Kp, params.Kd, params.Ki);
                }
                if (!isRunning && (key.startsWith('theta0') || key.startsWith('phi0'))) {
                    model.reset();
                    renderer.clearData();
                    renderer.draw(model.state, model);
                }
            });
        });
    }

    function toggleSimulation() {
        isRunning = !isRunning;
        document.getElementById('btn-start-stop').innerHTML = isRunning ? '⏸ 정지' : '▶ 시작';
        document.getElementById('btn-start-stop').classList.toggle('btn-primary');
        document.getElementById('btn-start-stop').classList.toggle('btn-danger');
        if (isRunning) loop();
        else cancelAnimationFrame(animFrameId);
    }

    function resetSimulation() {
        isRunning = false;
        document.getElementById('btn-start-stop').innerHTML = '▶ 시작';
        document.getElementById('btn-start-stop').classList.add('btn-primary');
        document.getElementById('btn-start-stop').classList.remove('btn-danger');

        // 슬라이더 초기값 동기화
        params.theta0 = parseFloat(document.getElementById('slider-theta0').value);
        params.phi0 = 0;

        cancelAnimationFrame(animFrameId);
        model.reset();
        controller.reset();
        renderer.clearData();
        renderer.draw(model.state, model);
    }

    function loop() {
        if (!isRunning) return;
        const state = model.step(controller);
        renderer.pushData(state.time, state.phi, state.theta, model.tau);
        renderer.draw(state, model);
        animFrameId = requestAnimationFrame(loop);
    }

    // 이벤트 리스너
    document.getElementById('btn-start-stop').addEventListener('click', toggleSimulation);
    document.getElementById('btn-reset').addEventListener('click', resetSimulation);

    document.getElementById('btn-perturb-left').addEventListener('click', () => { model.applyPerturbation(-2.0); });
    document.getElementById('btn-perturb-right').addEventListener('click', () => { model.applyPerturbation(2.0); });

    const btnCtrl = document.getElementById('btn-controller');
    btnCtrl.addEventListener('click', () => {
        params.controllerOn = !params.controllerOn;
        if (params.controllerOn) {
            btnCtrl.classList.add('active');
            btnCtrl.innerHTML = '🎛 제어기 ON';
            controller.reset();
        } else {
            btnCtrl.classList.remove('active');
            btnCtrl.innerHTML = '🎛 제어기 OFF';
        }
    });

    window.addEventListener('resize', () => {
        renderer.resize();
        if (!isRunning) renderer.draw(model.state, model);
    });

    // 초기화
    setupSliders();
    renderer.draw(model.state, model);
});

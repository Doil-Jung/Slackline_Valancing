/**
 * Slackline Balance Simulator V6 — Three.js 3D Renderer
 * 
 * 2D 물리 엔진 결과를 3D 시각화로 매핑
 * 좌표계: x = 수평(좌우), y = 깊이(기둥 방향), z = 수직(위)
 * 
 * 기존 2D 좌표 (model): x → 3D x, y → 3D z
 * Z_OFFSET: 바닥에서 1m 위로 올림
 */
var SL = SL || {};

SL.Renderer3D = class {
    constructor(container, params) {
        this.container = container;
        this.params = params;
        this.Z_OFFSET = 1.0; // 전체 높이 오프셋 (바닥에서 띄움)

        // === Three.js 기본 설정 ===
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a1a);
        this.scene.fog = new THREE.FogExp2(0x0a0a1a, 0.06);

        // 카메라
        const aspect = container.clientWidth / container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
        this.camera.position.set(4, 3, 3.5);
        this.camera.up.set(0, 0, 1); // z = up

        // 렌더러
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.2;
        container.appendChild(this.renderer.domElement);

        // 카메라 컨트롤
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.target.set(0, 0, this.Z_OFFSET + 1.2);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 1.5;
        this.controls.maxDistance = 15;
        this.controls.update();

        // 색상 팔레트
        this.colors = {
            arc: 0x3a86ff,
            lower: 0x06d6a0,
            upper: 0x8338ec,
            hip: 0xffbe0b,
            com: 0xff006e,
            rope: 0xffcc00,
            pole: 0x888888,
            ground: 0x1a1a3e,
        };

        this._buildScene();
        this._onResize = () => this.resize();
        window.addEventListener('resize', this._onResize);
    }

    // ================================================================
    //  Scene 생성
    // ================================================================
    _buildScene() {
        // === 조명 ===
        const ambient = new THREE.AmbientLight(0x404060, 1.0);
        this.scene.add(ambient);

        const dirLight = new THREE.DirectionalLight(0xffffff, 1.5);
        dirLight.position.set(5, 3, 8);
        dirLight.castShadow = true;
        dirLight.shadow.mapSize.set(1024, 1024);
        dirLight.shadow.camera.near = 0.5;
        dirLight.shadow.camera.far = 25;
        dirLight.shadow.camera.left = -6;
        dirLight.shadow.camera.right = 6;
        dirLight.shadow.camera.top = 6;
        dirLight.shadow.camera.bottom = -6;
        this.scene.add(dirLight);

        const fillLight = new THREE.DirectionalLight(0x6688ff, 0.4);
        fillLight.position.set(-3, -2, 5);
        this.scene.add(fillLight);

        // 반구광 (부드러운 환경 조명)
        const hemiLight = new THREE.HemisphereLight(0x4466aa, 0x223344, 0.6);
        this.scene.add(hemiLight);

        // === 바닥 ===
        const groundGeo = new THREE.PlaneGeometry(20, 20);
        const groundMat = new THREE.MeshStandardMaterial({
            color: this.colors.ground,
            roughness: 0.9,
            metalness: 0.1,
        });
        this.ground = new THREE.Mesh(groundGeo, groundMat);
        this.ground.receiveShadow = true;
        this.scene.add(this.ground);

        // === 바닥 그리드 ===
        const gridHelper = new THREE.GridHelper(20, 40, 0x333366, 0x222244);
        gridHelper.rotation.x = Math.PI / 2;
        gridHelper.position.z = 0.001;
        this.scene.add(gridHelper);

        // === 원호 트랙 ===
        this._buildArcTrack();

        // === 기둥 2개 ===
        this._buildPoles();

        // === V자 줄 (직선) ===
        this._buildRope();

        // === 몸체 (하체 + 상체 직육면체) ===
        this._buildBody();

        // === 힙 관절 (원통/축) ===
        this._buildHipJoint();

        // === CoM 마커 ===
        const comGeo = new THREE.OctahedronGeometry(0.06, 0);
        const comMat = new THREE.MeshBasicMaterial({
            color: this.colors.com,
            transparent: true,
            opacity: 0.8,
        });
        this.comMesh = new THREE.Mesh(comGeo, comMat);
        this.scene.add(this.comMesh);

        // === 토크 화살표 ===
        this.torqueArrow = null;

        // === 수직 기준선 ===
        const refLineGeo = new THREE.BufferGeometry();
        const refLinePositions = new Float32Array([0, 0, 0, 0, 0, 3]);
        refLineGeo.setAttribute('position', new THREE.BufferAttribute(refLinePositions, 3));
        const refLineMat = new THREE.LineDashedMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.15,
            dashSize: 0.05,
            gapSize: 0.05,
        });
        this.refLine = new THREE.Line(refLineGeo, refLineMat);
        this.refLine.computeLineDistances();
        this.scene.add(this.refLine);
    }

    _buildArcTrack() {
        const p = this.params;
        const R = p.R;
        const phiMax = p.phiMax;
        const steps = 80;
        const zOff = this.Z_OFFSET;

        const points = [];
        for (let i = 0; i <= steps; i++) {
            const phi = -phiMax + (2 * phiMax * i) / steps;
            const x = R * Math.sin(phi);
            const z = R * (1 - Math.cos(phi)) + zOff;
            points.push(new THREE.Vector3(x, 0, z));
        }

        const visible = p.showArc !== false;

        // 주 트랙 라인
        if (this.arcLine) this.scene.remove(this.arcLine);
        const arcGeo = new THREE.BufferGeometry().setFromPoints(points);
        const arcMat = new THREE.LineBasicMaterial({
            color: this.colors.arc,
            linewidth: 2,
            transparent: true,
            opacity: 0.8,
        });
        this.arcLine = new THREE.Line(arcGeo, arcMat);
        this.arcLine.visible = visible;
        this.scene.add(this.arcLine);

        // 글로우 라인
        if (this.arcGlow) this.scene.remove(this.arcGlow);
        const glowMat = new THREE.LineBasicMaterial({
            color: this.colors.arc,
            linewidth: 4,
            transparent: true,
            opacity: 0.2,
        });
        this.arcGlow = new THREE.Line(arcGeo.clone(), glowMat);
        this.arcGlow.visible = visible;
        this.scene.add(this.arcGlow);

        // 원호 튜브 (두께 가시화)
        if (this.arcTube) this.scene.remove(this.arcTube);
        const curve = new THREE.CatmullRomCurve3(points);
        const tubeGeo = new THREE.TubeGeometry(curve, steps, 0.015, 8, false);
        const tubeMat = new THREE.MeshStandardMaterial({
            color: this.colors.arc,
            emissive: this.colors.arc,
            emissiveIntensity: 0.3,
            transparent: true,
            opacity: 0.5,
            roughness: 0.4,
        });
        this.arcTube = new THREE.Mesh(tubeGeo, tubeMat);
        this.arcTube.visible = visible;
        this.scene.add(this.arcTube);
    }

    _buildPoles() {
        const p = this.params;
        const L_pole = p.L_pole || 4;
        const R = p.R;
        const zOff = this.Z_OFFSET;
        const poleHeight = R + zOff + 0.5; // 바닥부터 줄 묶인 높이까지

        if (this.poleA) this.scene.remove(this.poleA);
        if (this.poleB) this.scene.remove(this.poleB);

        const poleGeo = new THREE.CylinderGeometry(0.04, 0.05, poleHeight, 12);
        const poleMat = new THREE.MeshStandardMaterial({
            color: this.colors.pole,
            roughness: 0.5,
            metalness: 0.6,
        });

        // CylinderGeometry 축은 기본 y축 → z축(수직)으로 회전
        this.poleA = new THREE.Mesh(poleGeo, poleMat);
        this.poleA.rotation.x = Math.PI / 2;
        this.poleA.position.set(0, L_pole / 2, poleHeight / 2);
        this.poleA.castShadow = true;
        this.scene.add(this.poleA);

        this.poleB = new THREE.Mesh(poleGeo.clone(), poleMat.clone());
        this.poleB.rotation.x = Math.PI / 2;
        this.poleB.position.set(0, -L_pole / 2, poleHeight / 2);
        this.poleB.castShadow = true;
        this.scene.add(this.poleB);

        // 기둥 상단 마커
        const topGeo = new THREE.SphereGeometry(0.06, 12, 12);
        const topMat = new THREE.MeshStandardMaterial({ color: 0xcccccc, metalness: 0.8 });
        if (this.poleTopA) this.scene.remove(this.poleTopA);
        if (this.poleTopB) this.scene.remove(this.poleTopB);
        this.poleTopA = new THREE.Mesh(topGeo, topMat);
        this.poleTopA.position.set(0, L_pole / 2, poleHeight);
        this.scene.add(this.poleTopA);
        this.poleTopB = new THREE.Mesh(topGeo.clone(), topMat.clone());
        this.poleTopB.position.set(0, -L_pole / 2, poleHeight);
        this.scene.add(this.poleTopB);
    }

    _buildRope() {
        // 사다리꼴 줄: 기둥A → 하체 아랫면 앞모서리 → 하체 아랫면 뒷모서리 → 기둥B
        // 초기 위치 (update에서 매 프레임 갱신)
        const bd = (this.params.bodyDepth || 0.25) / 2;
        const ropePoints = [
            new THREE.Vector3(0, 2, 2.5),       // 기둥A
            new THREE.Vector3(0, bd, 1),         // 하체 아랫면 앞
            new THREE.Vector3(0, -bd, 1),        // 하체 아랫면 뒤
            new THREE.Vector3(0, -2, 2.5),       // 기둥B
        ];

        if (this.ropeLine) this.scene.remove(this.ropeLine);
        const ropeGeo = new THREE.BufferGeometry().setFromPoints(ropePoints);
        const ropeMat = new THREE.LineBasicMaterial({
            color: this.colors.rope,
            linewidth: 2,
        });
        this.ropeLine = new THREE.Line(ropeGeo, ropeMat);
        this.scene.add(this.ropeLine);

        // 직선 사다리꼴 튜브 (두께 있는 줄, 3개의 직선 세그먼트)
        this._buildRopeTubes(ropePoints);
    }

    _buildRopeTubes(ropePoints) {
        // 기존 튜브 제거
        if (this.ropeTubeA) this.scene.remove(this.ropeTubeA);
        if (this.ropeTubeB) this.scene.remove(this.ropeTubeB);
        if (this.ropeTubeBottom) this.scene.remove(this.ropeTubeBottom);

        const tubeMat = new THREE.MeshStandardMaterial({
            color: this.colors.rope,
            emissive: this.colors.rope,
            emissiveIntensity: 0.2,
            roughness: 0.6,
        });

        // 세그먼트 A: 기둥A → 하체 아랫면 앞모서리
        const curveA = new THREE.LineCurve3(ropePoints[0], ropePoints[1]);
        const tubeGeoA = new THREE.TubeGeometry(curveA, 1, 0.012, 6, false);
        this.ropeTubeA = new THREE.Mesh(tubeGeoA, tubeMat);
        this.ropeTubeA.castShadow = true;
        this.scene.add(this.ropeTubeA);

        // 세그먼트 Bottom: 하체 아랫면 (아랫변)
        const curveBottom = new THREE.LineCurve3(ropePoints[1], ropePoints[2]);
        const tubeGeoBottom = new THREE.TubeGeometry(curveBottom, 1, 0.012, 6, false);
        this.ropeTubeBottom = new THREE.Mesh(tubeGeoBottom, tubeMat.clone());
        this.ropeTubeBottom.castShadow = true;
        this.scene.add(this.ropeTubeBottom);

        // 세그먼트 B: 하체 아랫면 뒷모서리 → 기둥B
        const curveB = new THREE.LineCurve3(ropePoints[2], ropePoints[3]);
        const tubeGeoB = new THREE.TubeGeometry(curveB, 1, 0.012, 6, false);
        this.ropeTubeB = new THREE.Mesh(tubeGeoB, tubeMat.clone());
        this.ropeTubeB.castShadow = true;
        this.scene.add(this.ropeTubeB);
    }

    _buildBody() {
        const p = this.params;
        const bodyDepth = p.bodyDepth || 0.25;

        // === 하체 직육면체 ===
        if (this.lowerBody) this.scene.remove(this.lowerBody);
        const lowerGeo = new THREE.BoxGeometry(p.lowerWidth, bodyDepth, p.L1);
        const lowerMat = new THREE.MeshStandardMaterial({
            color: this.colors.lower,
            transparent: true,
            opacity: 0.7,
            roughness: 0.4,
            metalness: 0.1,
        });
        this.lowerBody = new THREE.Mesh(lowerGeo, lowerMat);
        this.lowerBody.castShadow = true;
        this.scene.add(this.lowerBody);

        // 하체 윤곽선
        if (this.lowerEdges) this.scene.remove(this.lowerEdges);
        const lowerEdgeGeo = new THREE.EdgesGeometry(lowerGeo);
        const lowerEdgeMat = new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.6 });
        this.lowerEdges = new THREE.LineSegments(lowerEdgeGeo, lowerEdgeMat);
        this.lowerBody.add(this.lowerEdges);

        // === 상체 직육면체 ===
        if (this.upperBody) this.scene.remove(this.upperBody);
        const upperGeo = new THREE.BoxGeometry(p.upperWidth, bodyDepth, p.L2);
        const upperMat = new THREE.MeshStandardMaterial({
            color: this.colors.upper,
            transparent: true,
            opacity: 0.7,
            roughness: 0.4,
            metalness: 0.1,
        });
        this.upperBody = new THREE.Mesh(upperGeo, upperMat);
        this.upperBody.castShadow = true;
        this.scene.add(this.upperBody);

        // 상체 윤곽선
        if (this.upperEdges) this.scene.remove(this.upperEdges);
        const upperEdgeGeo = new THREE.EdgesGeometry(upperGeo);
        const upperEdgeMat = new THREE.LineBasicMaterial({ color: 0xa855f7, transparent: true, opacity: 0.6 });
        this.upperEdges = new THREE.LineSegments(upperEdgeGeo, upperEdgeMat);
        this.upperBody.add(this.upperEdges);
    }

    _buildHipJoint() {
        // 힙 관절: y축 방향 원통 (회전축 표현)
        const bodyDepth = this.params.bodyDepth || 0.25;

        if (this.hipMesh) this.scene.remove(this.hipMesh);

        // 원통: 축이 y방향, 길이 = bodyDepth * 1.5 (몸체보다 약간 돌출)
        const hipGeo = new THREE.CylinderGeometry(0.03, 0.03, bodyDepth * 1.5, 12);
        const hipMat = new THREE.MeshStandardMaterial({
            color: this.colors.hip,
            emissive: this.colors.hip,
            emissiveIntensity: 0.5,
            roughness: 0.3,
            metalness: 0.5,
        });
        this.hipMesh = new THREE.Mesh(hipGeo, hipMat);
        // CylinderGeometry 기본 축 = y → z-up 좌표계에서 회전축(y방향)으로 유지
        // 힙 축은 y방향으로 뻗어야 하므로 회전 불필요
        this.hipMesh.castShadow = true;
        this.scene.add(this.hipMesh);
    }

    // ================================================================
    //  줄 위치 업데이트 (사다리꼴 기하학)
    // ================================================================
    _updateRope(footX, footZ) {
        const p = this.params;
        const L_pole = p.L_pole || 4;
        const R = p.R;
        const zOff = this.Z_OFFSET;
        const bd = (p.bodyDepth || 0.25) / 2;

        // 기둥 상단 위치 (줄이 묶인 곳) = 원호 중심 높이 R + Z_OFFSET
        const poleTopZ = R + zOff;
        const poleAY = L_pole / 2;
        const poleBY = -L_pole / 2;

        // 사다리꼴: 기둥A → 하체 아랫면 앞모서리 → 하체 아랫면 뒷모서리 → 기둥B
        // 하체 아랫면은 발 위치에서 y = ±bodyDepth/2
        const ropePoints = [
            new THREE.Vector3(0, poleAY, poleTopZ),        // 기둥A 상단
            new THREE.Vector3(footX, bd, footZ),            // 하체 아랫면 앞모서리
            new THREE.Vector3(footX, -bd, footZ),           // 하체 아랫면 뒷모서리
            new THREE.Vector3(0, poleBY, poleTopZ),        // 기둥B 상단
        ];

        // 라인 업데이트
        this.ropeLine.geometry.dispose();
        this.ropeLine.geometry = new THREE.BufferGeometry().setFromPoints(ropePoints);

        // 직선 튜브 업데이트
        if (this.ropeTubeA) this.scene.remove(this.ropeTubeA);
        if (this.ropeTubeB) this.scene.remove(this.ropeTubeB);
        if (this.ropeTubeBottom) this.scene.remove(this.ropeTubeBottom);

        const tubeMat = new THREE.MeshStandardMaterial({
            color: this.colors.rope,
            emissive: this.colors.rope,
            emissiveIntensity: 0.2,
            roughness: 0.6,
        });

        // 세그먼트 A: 기둥A → 하체 아랫면 앞
        const curveA = new THREE.LineCurve3(ropePoints[0], ropePoints[1]);
        const tubeGeoA = new THREE.TubeGeometry(curveA, 1, 0.012, 6, false);
        this.ropeTubeA = new THREE.Mesh(tubeGeoA, tubeMat);
        this.ropeTubeA.castShadow = true;
        this.scene.add(this.ropeTubeA);

        // 세그먼트 Bottom: 하체 아랫면 아랫변
        const curveBottom = new THREE.LineCurve3(ropePoints[1], ropePoints[2]);
        const tubeGeoBottom = new THREE.TubeGeometry(curveBottom, 1, 0.012, 6, false);
        this.ropeTubeBottom = new THREE.Mesh(tubeGeoBottom, tubeMat.clone());
        this.ropeTubeBottom.castShadow = true;
        this.scene.add(this.ropeTubeBottom);

        // 세그먼트 B: 하체 아랫면 뒤 → 기둥B
        const curveB = new THREE.LineCurve3(ropePoints[2], ropePoints[3]);
        const tubeGeoB = new THREE.TubeGeometry(curveB, 1, 0.012, 6, false);
        this.ropeTubeB = new THREE.Mesh(tubeGeoB, tubeMat.clone());
        this.ropeTubeB.castShadow = true;
        this.scene.add(this.ropeTubeB);
    }

    // ================================================================
    //  직육면체 위치/회전 매핑 (수정: 올바른 회전 방향)
    // ================================================================
    _positionCuboid(mesh, from, to) {
        // from, to: {x, z} in 3D space (y=0)
        const midX = (from.x + to.x) / 2;
        const midZ = (from.z + to.z) / 2;

        mesh.position.set(midX, 0, midZ);

        // 방향 벡터: from → to
        const dx = to.x - from.x;
        const dz = to.z - from.z;
        // 수직(z축)으로부터의 각도
        const angle = Math.atan2(dx, dz);

        // BoxGeometry의 z축이 길이방향
        // y축 회전으로 xz 평면 기울기 매핑
        // +angle: (0,0,1) → (sin(angle), 0, cos(angle)) = from→to 방향
        mesh.rotation.set(0, 0, 0);
        mesh.rotation.y = angle;
    }

    // ================================================================
    //  프레임 업데이트
    // ================================================================
    update(state, model) {
        const p = this.params;
        const zOff = this.Z_OFFSET;

        // 2D 좌표 → 3D 좌표 매핑 (+ Z_OFFSET)
        const foot2D = model.getFootPos(state);
        const hip2D = model.getHipPos(state);
        const head2D = model.getHeadPos(state);
        const com2D = model.getTotalCoM(state);

        const foot = { x: foot2D.x, z: foot2D.y + zOff };
        const hip = { x: hip2D.x, z: hip2D.y + zOff };
        const head = { x: head2D.x, z: head2D.y + zOff };
        const com = { x: com2D.x, z: com2D.y + zOff };

        // === 힙 관절 (원통) ===
        this.hipMesh.position.set(hip.x, 0, hip.z);

        // === 하체 직육면체 ===
        this._positionCuboid(this.lowerBody, foot, hip);

        // === 상체 직육면체 ===
        this._positionCuboid(this.upperBody, hip, head);

        // === CoM ===
        this.comMesh.position.set(com.x, 0, com.z);
        this.comMesh.rotation.z += 0.03;

        // === V자 줄 ===
        this._updateRope(foot.x, foot.z);

        // === 수직 기준선 (발 위치에서 위로) ===
        const refPositions = this.refLine.geometry.attributes.position.array;
        refPositions[0] = foot.x; refPositions[1] = 0; refPositions[2] = foot.z;
        refPositions[3] = foot.x; refPositions[4] = 0; refPositions[5] = foot.z + p.L1 + p.L2 + 0.3;
        this.refLine.geometry.attributes.position.needsUpdate = true;
        this.refLine.computeLineDistances();

        // === 기둥 위치 업데이트 (파라미터 변경 시) ===
        const L_pole = p.L_pole || 4;
        const R = p.R;
        const poleHeight = R + zOff + 0.5;
        this.poleA.position.set(0, L_pole / 2, poleHeight / 2);
        this.poleB.position.set(0, -L_pole / 2, poleHeight / 2);
        this.poleTopA.position.set(0, L_pole / 2, poleHeight);
        this.poleTopB.position.set(0, -L_pole / 2, poleHeight);

        // 렌더링
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    // ================================================================
    //  리빌드 (파라미터 변경 시)
    // ================================================================
    rebuild() {
        this._buildArcTrack();
        this._buildPoles();
        this._buildBody();
        this._buildHipJoint();
    }

    resize() {
        const w = this.container.clientWidth;
        const h = this.container.clientHeight;
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    dispose() {
        window.removeEventListener('resize', this._onResize);
        this._idleRunning = false;
        this.renderer.dispose();
        if (this.renderer.domElement.parentNode) {
            this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
        }
    }

    /** 시뮬레이션 정지 상태에서도 카메라 조작이 가능하도록 독립 렌더 루프 */
    startIdleRender() {
        this._idleRunning = true;
        const idleLoop = () => {
            if (!this._idleRunning) return;
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
            requestAnimationFrame(idleLoop);
        };
        requestAnimationFrame(idleLoop);
    }
};

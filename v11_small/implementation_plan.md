# V9 확장 — LQR R/Q 튜닝 + 온라인 시스템 식별

## 1. LQR R/Q 튜닝 슬라이더

### 목적
현재 Q, R 값이 하드코딩(Q_diag=[1,50,50,0.1,5,5], R=0.001)되어 있어 게인이 매우 공격적. R값을 올리면 게인이 낮아지고 모델 오차에 강건해지되, 응답이 느려지는 트레이드오프를 사용자가 실시간으로 실험할 수 있게 함.

### 구현
- **R 스케일 슬라이더** (×0.1 ~ ×1000, 로그 스케일): R_scalar에 곱해짐
- LQR 게인 K_base를 슬라이더 변경 시 **실시간 재계산** (linearize → discretize → DARE → gain)
- 기존 [computeAugmentedGains()](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/v8_observer/src/control/lqr_v8.js#412-457)의 DARE 인프라 재활용
- 재계산된 K_base를 info 오버레이에 표시

#### [MODIFY] [lqr_v9.js](file:///c:/Users/user/OneDrive%20-%20충북과학고등학교/원드라이브%20동기화%20폴더/코딩%20작업%20폴더/slackline_valancing/v9_robustness/src/control/lqr_v9.js)
- `recomputeBaseGain(params, rScale)` 메서드 추가
- 내부에서 linearize → discretize(1ms) → DARE(6×6) → K_base 갱신
- `rScale` 프로퍼티 추가 (기본 1.0)

#### [MODIFY] [index.html](file:///c:/Users/user/OneDrive%20-%20충북과학고등학교/원드라이브%20동기화%20폴더/코딩%20작업%20폴더/slackline_valancing/v9_robustness/index.html)
- LQR 제어 섹션에 "⚖️ LQR R 스케일" 로그 슬라이더 추가

#### [MODIFY] [main.js](file:///c:/Users/user/OneDrive%20-%20충북과학고등학교/원드라이브%20동기화%20폴더/코딩%20작업%20폴더/slackline_valancing/v9_robustness/src/main.js)
- R 스케일 슬라이더 이벤트 핸들러
- 오버레이에 현재 K_base 게인 크기 표시

---

## 2. 온라인 시스템 식별 (RLS)

### 목적
모델 불확실성이 있을 때 운용 중 실제 플랜트 파라미터(m1, m2 등)를 추정하여 제어기/옵저버를 **적응적으로 업데이트**.

### 알고리즘: Recursive Least Squares (RLS)
운동방정식 M(q)q̈ = f(q,q̇,τ) 에서 가속도 관측값과 모델 예측값의 차이를 최소화하는 방향으로 파라미터를 점진적으로 추정.

**핵심 아이디어:**
- 가속도 α̈, θ̈를 수치 미분(연속 2개 α̇, θ̇ 차분)으로 측정
- 운동방정식을 파라미터에 대해 선형 회귀 형태로 변환: `y = Φ·θ_param + ε`
- RLS로 θ_param (m1, m2 등) 온라인 추정
- 추정된 파라미터로 제어기 게인 & 옵저버 모델 갱신

### 시뮬레이터에서 테스트 방법
1. 모델 불확실성 슬라이더로 편차 설정 (예: m1 +15%)
2. "🔍 시스템 식별 시작" 버튼 클릭
3. 시뮬레이션 실행 → RLS가 실시간으로 파라미터 추정
4. 추정된 값이 실제 플랜트 값에 수렴하는 과정을 오버레이에 표시
5. 추정 수렴 후 "적용" → 제어기/옵저버에 추정 파라미터 반영 → 강건성 개선

#### [NEW] [sysid.js](file:///c:/Users/user/OneDrive%20-%20충북과학고등학교/원드라이브%20동기화%20폴더/코딩%20작업%20폴더/slackline_valancing/v9_robustness/src/control/sysid.js)
- `SL.SystemIdentifier` 클래스
- [init(nominalParams)](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/v6_3d/src/main.js#22-54) — 초기 파라미터 추정치 = 공칭값
- [update(state, prevState, tau, dt)](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/v7_estimation/index.html#338-365) — RLS 1스텝 업데이트
- `getEstimatedParams()` — 현재 추정 파라미터 반환
- `getConvergenceInfo()` — 추정 수렴도 정보

#### [MODIFY] index.html
- "🔍 시스템 식별" 섹션 추가: 시작/정지 버튼, 파라미터 수렴 표시, "적용" 버튼

#### [MODIFY] main.js
- SystemIdentifier 인스턴스 관리
- 시뮬레이션 루프에서 매 스텝 `sysid.update()` 호출
- "적용" 시 옵저버/제어기 재초기화

## Verification Plan

### 사용자 수동 테스트
1. **R 스케일 테스트**: R×100에서 모델편차 15%에도 안정적인지 확인
2. **시스템 식별 테스트**: m1 +10% 편차 → 식별 실행 → m1 추정값이 실제값(33kg)에 수렴하는지 확인
3. **식별 적용 후**: 추정 파라미터 적용 후 제어 성능 개선 확인

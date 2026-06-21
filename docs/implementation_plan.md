# V3 3-DOF 물리 모델 구현

기존 2-DOF 모델(φ, θ, 하체 각도 = −φ 고정)을 올바른 3-DOF 모델(φ, α, θ)로 교체합니다.
하체 각도 α를 독립 자유도로 풀어, 발-줄 접촉이 자유 힌지인 실제 물리 상황을 구현합니다.

## Proposed Changes

### Physics Engine

#### [MODIFY] [params.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/src/physics/params.js)

- 초기조건 `alpha0` 추가 (기본값 0)
- 감쇠 `b_alpha` 추가
- 파생 상수 getter 수정: `p1 = m₁ℓ₁ + m₂L₁`, `p2 = m₂ℓ₂`, `p3 = m₂L₁ℓ₂` 등 3-DOF EOM 상수
- 기존 2-DOF용 `A`, `B`, `C_`, `Mg_phi`, `Mg_theta` getter 제거
- 슬라이더 `ranges`에 `alpha0`, `b_alpha` 추가

#### [MODIFY] [model.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/src/physics/model.js)

**핵심 변경**: `computeDerivatives()`를 3-DOF EOM으로 전면 교체

- state에 `alpha`, `alphaDot` 추가
- 3×3 Mass Matrix 계산:
  - `M₁₁ = (m₁+m₂)R²`
  - `M₁₂ = R·p₁·cos(φ+α)`, `M₁₃ = R·p₂·cos(φ+θ)` 
  - `M₂₂ = m₁ℓ₁²+I₁+m₂L₁²`, `M₂₃ = p₃·cos(α−θ)`
  - `M₃₃ = m₂ℓ₂²+I₂`
- RHS (코리올리/원심력 + 중력 + 토크):
  - `f₁ = R·p₁·sin(φ+α)·α̇² + R·p₂·sin(φ+θ)·θ̇² − (m₁+m₂)gR·sin(φ) − b_φ·φ̇`
  - `f₂ = R·p₁·sin(φ+α)·φ̇² − p₃·sin(α−θ)·θ̇² + p₁·g·sin(α) − τ − b_α·α̇`
  - `f₃ = R·p₂·sin(φ+θ)·φ̇² + p₃·sin(α−θ)·α̇² + p₂·g·sin(θ) + τ − b_θ·θ̇`
- 3×3 역행렬로 `[φ̈, α̈, θ̈]` 계산
- RK4를 6-element state `[φ, α, θ, φ̇, α̇, θ̇]`로 확장
- 위치 계산 메서드(`getHipPos`, `getLowerCoM`, `getUpperCoM` 등): α 사용하도록 수정

---

### Controller

#### [MODIFY] [pid.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%84/slackline_valancing/src/control/pid.js)

- PID `compute()` 인터페이스를 `compute(state, model, dt)`로 변경
- 제어 목표를 **CoM 수평 위치 − 발 수평 위치** 오차로 변경:
  `error = x_com − x_foot`
- `model.getTotalCoM()`, `model.getFootPos()`를 활용하여 오차 계산

---

### Rendering & UI

#### [MODIFY] [renderer.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/src/render/renderer.js)

- `drawSim()`: 하체의 기하학이 이미 `model.getFootPos()`~`model.getHipPos()`를 사용하므로 모델 수정에 따라 자동 반영됨. 별도 수정 최소화.
- `drawInfo()`: α 각도 표시 추가, CoM−발 오차 표시 추가
- `pushData()` / `drawGraphs()`: α 그래프 추가

#### [MODIFY] [main.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/src/main.js)

- PID 호출부 변경: `controller.compute(state, model, dt)`
- `pushData()`에 α 추가

#### [MODIFY] [index.html](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/index.html)

- 부제목: "3-DOF Hip Torque Model" 로 변경
- α 관련 슬라이더(초기 하체 기울기, 감쇠) 추가
- 범례에 α 항목 추가

## Verification Plan

### 브라우저 테스트 (핵심)

1. `index.html`을 브라우저에서 열어 시뮬레이션이 크래시 없이 실행되는지 확인
2. **제어기 OFF 상태**에서:
   - 초기 기울기를 주고 시작 → 역진자처럼 자연스럽게 넘어지는지 (α가 독립적으로 변하는지)
   - φ도 중력/관성에 의해 자연스럽게 변하는지
3. **제어기 ON 상태**에서:
   - 시작 시 균형을 잡으려고 하는지 (완벽하지 않아도 됨)
   - 교란 버튼으로 밀었을 때 반응하는지
4. 모든 슬라이더가 정상 동작하는지
5. 그래프에 α 데이터가 표시되는지

> [!IMPORTANT]
> 이 프로젝트는 정적 HTML/JS이므로 자동화된 단위 테스트가 없습니다.
> 검증은 브라우저에서 시뮬레이션을 직접 실행하여 수행합니다.

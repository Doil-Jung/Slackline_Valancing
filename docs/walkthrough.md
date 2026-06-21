# V3 3-DOF 구현 완료 보고

## 변경된 파일 (6개)

| 파일 | 변경 내용 |
|------|-----------|
| [params.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/src/physics/params.js) | α₀, b_α 추가, EOM 상수(p₁,p₂,p₃,M₁₁,M₂₂,M₃₃) getter 추가 |
| [model.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%94/slackline_valancing/src/physics/model.js) | **전면 재작성** — 3×3 질량행렬 + 코리올리/중력 + 3×3 역행렬 풀이 |
| [pid.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%84/slackline_valancing/src/control/pid.js) | CoM−발 수평오차 기반 PID로 변경 |
| [renderer.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%84/slackline_valancing/src/render/renderer.js) | α 그래프 추가, CoM 오차 표시, 4개 그래프(θ,α,φ,τ) |
| [main.js](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%94/%EC%BD%94%EB%94%A9%20%EC%9E%91%EC%97%85%20%ED%8F%B4%EB%8D%84/slackline_valancing/src/main.js) | 6-element state, α 데이터 전달, 새 PID 인터페이스 |
| [index.html](file:///c:/Users/user/OneDrive%20-%20%EC%B6%A9%EB%B6%81%EA%B3%BC%ED%95%99%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90/%EC%9B%90%EB%93%9C%EB%9D%BC%EC%9D%B4%EB%B8%8C%20%EB%8F%99%EA%B8%B0%ED%99%94%20%ED%8F%B4%EB%8D%84/slackline_valancing/index.html) | α₀·b_α 슬라이더 추가, 부제목·범례 업데이트 |

## 핵심 변경 — 물리 모델

```diff
- 2-DOF: q = [φ, θ], 하체 각도 = −φ (원호 법선 고정)
+ 3-DOF: q = [φ, α, θ], 하체 각도 α는 독립 자유도
```

```diff
- 제어 목표: θ → 0
+ 제어 목표: CoM_x − Foot_x → 0
```

## 검증 상태

> [!IMPORTANT]
> Python/Node.js가 설치되어 있지 않아 자동 서버 구동이 불가했습니다.
> **`index.html` 파일을 브라우저에서 직접 열어** 확인해 주세요.

### 확인 포인트
1. 시뮬레이션 시작 시 하체(녹색)가 **원호 법선과 독립적으로** 움직이는지
2. 제어기 OFF 시 역진자처럼 넘어지는지
3. 제어기 ON 시 CoM이 발 위에 유지되려 하는(혹은 시도하는)지
4. 그래프에 4개 라인(θ, α, φ, τ)이 모두 표시되는지
5. 정보 패널에 `α`, `CoM오차` 값이 보이는지

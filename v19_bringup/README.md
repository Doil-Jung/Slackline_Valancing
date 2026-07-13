# v19_bringup — 재설계 로봇·줄 시스템 실측 & 작동 테스트 (2026-07-12)

재설계(7/08 스펙: φ 엔코더 직접측정, 카본 사다리꼴 크랭크, 발목 베어링) 기준의
**실측 → 게인 계산 → LQR → FWE** 전체 파이프라인. 줄 관성(I_r, S_r)과 베어링
감쇠(c_φ)가 플랜트·FWE 상수에 **정식 파라미터로** 들어간다(7/11 이론: ε*가
−30~−44% 이동하므로 무질량 공식 사용 금지).

## 폴더 구조

```
v19_bringup/
├── params_v19.py            ★단일 진실 소스 — 실측값은 전부 여기에만 입력
├── model_v19.py             플랜트(줄 관성·감쇠 포함)·모드·ε*·w4·FWE맵 (검산 내장)
├── compute_gains_v19.py     LQR K + 칼만 L(5측정) + FWE 상수 → gains_v19.h 자동생성
├── sim_check_v19.py         ★게인 이식 전 필수 폐루프 검증 (전부 PASS여야 진행)
├── calib/
│   ├── calib_analysis.py    복합진자→I_cm / 자유진동→ω·ζ·c_φ / 모델 검증
│   └── serial_logger.py     PC측 시리얼 CSV 로거 (pip install pyserial)
└── firmware/                (Arduino IDE + OpenCR 보드 패키지)
    ├── phi_encoder.h        φ 엔코더 추상화(AS5048A 기본) — 각 스케치 폴더에 복사본
    ├── encoder_phi/         [실측1] 엔코더 점검 + ★ENC_PHI_DIR 확정
    ├── free_swing_log/      [실측2] 자유진동 φ(t) 로깅 (c_φ·ω_eff 실측)
    ├── imu_axis_check/      [실측3] IMU 단위(자이로 16.4배 버그)·부호 점검
    ├── lqr_balance_v19/     LQR 균형 (전류제어 + φ직접측정 + 칼만5)
    └── fwe_v19/             FWE 시연 (단일접기/완전사이클/반복FEW)
```

기존 코드 대비 변경: `fwe_demo_standalone`(자이로 단위버그) **대체**,
`lqr_balance_current_v2`의 φ 간접추정(C1_PHI/C2_PHI) **폐기**(엔코더 직접측정),
게인 계산은 stdout 복붙 대신 **gains_v19.h 자동생성**. 검증된 v2 처리(1Mbps 승격,
vel+pos 단일 read, 2ms 페이싱, 선형가속 보상 칼만, 게이팅)는 그대로 계승.

## 사용 순서 (전체 체크리스트)

### 0단계 — 조립 전·중 실측 (저울·자)
- [ ] 카본 빗변 1개 질량, 하단 하드웨어(바+소켓+베어링) 질량 → `params ROPE.*`
- [ ] 하체/상체 질량·길이·CoM(균형점) → `params BODY.*` (★장비 매립 후 상체!)
- [ ] 복합진자법: 하체(발축 피벗)·상체(힙 피벗) 미소진동 주기 T (10회 왕복/10)
      → `python calib/calib_analysis.py pendulum --m … --d … --T …` → I1_cm/I2_cm
- [ ] ELL_IMU(힙→IMU 칩), 조립 후 R(베어링축→발축) 직접 측정

### 1단계 — 전장 방향·단위 확정 (스케치 3개 + 기존 torque_cal)
- [ ] `encoder_phi`: 통신·노이즈 확인, **+x로 밀면 φ+ → ENC_PHI_DIR 확정**
- [ ] `imu_axis_check`: gy_dps≈90(1초 90°회전), 앞기울임 thAcc+ → **IMU_DIR**
- [ ] `../opencr_test/torque_cal`(기존): +전류→+δ(앞접기) → **MOTOR_DIR**, Kt 검증
- [ ] 확정한 3개 부호를 `lqr_balance_v19.ino`·`fwe_v19.ino` 상단 상수에 반영

### 2단계 — 줄 시스템 실측 (로봇 없이)
- [ ] `free_swing_log` + `serial_logger.py`: 크랭크만 자유진동 10~20회
- [ ] `calib_analysis.py freeswing crank.csv --I_pivot <I_r>` → **c_φ** → params
      (ω²·I_r ≟ S_r·g 로 질량표 교차검증 — 10% 이상 어긋나면 재점검)

### 3단계 — 게인 계산·검증 (PC)
- [ ] `python params_v19.py` — 미실측 항목 0인지 확인 (전부 MEASURED=True)
- [ ] `python compute_gains_v19.py` — gains_v19.h 생성 (두 펌웨어 폴더에 자동 복사)
- [ ] `python sim_check_v19.py` — **전부 PASS 확인** (τ피크, 서보속도 여유 포함)

### 4단계 — 로봇 장착 첫 검증 (제어 없이) ★조립 후 첫 실물-이론 비교
- [ ] 로봇 장착·δ=0·서보 전원 OFF 로 자유진동 → `calib_analysis.py verify robot.csv`
      실측 ω vs 모델 ω_eff(줄 관성 포함) ±5% 이내면 진행

### 5단계 — LQR 브링업 (`lqr_balance_v19`)
- [ ] 부팅 로그: 1Mbps, RATE≈500Hz, GAINS_MEASURED 경고 없음
- [ ] 'd' 진단 → 'z' → 'c' (gain 0.3, 손으로 받칠 준비) → 점진 1.0, CUR_SAFE 600→855

### 6단계 — FWE 시연 (`fwe_v19`)  ※LQR 성공 후
- [ ] 모드 1(단일접기): 살짝 기울여 |A|>trig → 접고 유지 → 유계 진동 관찰("기다림의 상한 없음")
- [ ] 모드 2(완전사이클): 접기→대기→펴기 1회 후 직립 복귀
- [ ] 모드 3(반복 FEW, ★기본 권장): 사이클 반복하며 A 수축 — δ 실행오차 10%+ 환경의 정답

## 그 밖에 필요해질 수 있는 코드 (현황 점검)

| 항목 | 상태 |
|---|---|
| 서보 전류→토크 검증 (Kt) | 기존 `opencr_test/torque_cal` 재사용 (v19 재작성 불필요) |
| MuJoCo 전물리 FWE 검증 갭 (γ·사이클·반복) | 미완 — `v14_mujoco` 확장 과제 (이론 잔여 과제 ①) |
| 데이터 로깅(제어 중 상태 기록) | 현 시리얼 10~20Hz 로그로 시작, 필요 시 `serial_logger.py` 프리픽스 추가 |
| 전시용(저상 프레임) 파라미터 세트 | 실험용 확정 후 params 사본으로 분기 |
| AMT223/MA3 엔코더 확정 시 | `phi_encoder.h` 해당 타입만 구현 추가 |

## 주의 (알려진 함정)
- 세 방향 상수(ENC_PHI_DIR/IMU_DIR/MOTOR_DIR)는 **두 제어 스케치에 동일하게**.
- `gains_v19.h` 직접 수정 금지 — 반드시 params → compute_gains 경로로.
- 실측 전 게인은 GAINS_MEASURED=0 경고가 뜬다: 로봇을 줄에 올리지 말 것.
- FWE 단일접기(모드1)가 금방 발산하는 것은 δ 실행오차 시 **이론적으로 정상**
  (대기 구간이 오차를 e^{λT}배 증폭) — 반복 FEW(모드3)가 실물의 기본값이다.

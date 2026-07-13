/*
 * free_swing_log.ino — [실측 2단계] 자유진동 φ(t) 로깅 (v19)
 * ============================================================
 * 두 가지 실측에 같은 스케치를 쓴다:
 *  (A) 크랭크 단독 자유진동  → ω 로 S_r·g/I_r 비 검증, 로그감쇠로 ★c_φ 실측
 *      (로봇 떼고, 크랭크+하단바만 달아 5~15° 들었다 놓기)
 *  (B) 로봇 장착(δ=0, 서보 전원 OFF) 자유진동 → 실측 ω_eff 를 모델 예측과 비교
 *      = 조립 후 '첫 실물-이론 검증' (연구정리 7/11 후속작업 ②)
 *
 * 사용:
 *  1) 정지 상태에서 'z' (영점)
 *  2) PC에서 로거 시작:  python calib/serial_logger.py COM3 --out crank_free.csv
 *  3) 's' → 로깅 시작(200Hz CSV: t_ms,phi_deg). 크랭크를 들었다 놓는다.
 *     10~20 진동 후 다시 's' → 정지.
 *  4) 분석:  python calib/calib_analysis.py freeswing crank_free.csv
 *
 * 출력 형식: "L,<t_ms>,<phi_deg>"  (로거가 L 행만 CSV로 저장)
 */
#include "phi_encoder.h"

bool logging = false;
uint32_t t0 = 0;
uint32_t next_us = 0;
const uint32_t PERIOD_US = 5000;   // 200 Hz — 진자 ~1Hz 에 충분, 시리얼 여유

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("=== free_swing_log (v19) ===");
  if (!encoderBegin()) Serial.println("!!! 엔코더 응답 없음");
  Serial.println("z:영점  s:로깅 시작/정지   (형식 L,t_ms,phi_deg)");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'z') { encoderZeroHere(); Serial.println("# zero set"); }
    else if (c == 's') {
      logging = !logging;
      if (logging) { t0 = millis(); next_us = micros(); Serial.println("# LOG START"); }
      else Serial.println("# LOG STOP");
    }
  }
  if (logging) {
    uint32_t now = micros();
    if ((int32_t)(now - next_us) >= 0) {
      next_us += PERIOD_US;
      Serial.print("L,");
      Serial.print(millis() - t0);
      Serial.print(",");
      Serial.println(encoderPhiRad() * 180.0f / PI, 4);
    }
  }
}

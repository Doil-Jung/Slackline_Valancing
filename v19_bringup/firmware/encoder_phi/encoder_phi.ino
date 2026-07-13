/*
 * encoder_phi.ino — [실측 1단계] φ 엔코더 단독 점검 (v19)
 * ========================================================
 * 목적: ① 엔코더가 읽히는가  ② 분해능·노이즈 확인  ③ ★부호(ENC_PHI_DIR) 확정
 *
 * 배선(AS5048A 기준): 5V/3.3V(모듈 사양대로), GND, CS=D10, MOSI=D11, MISO=D12, SCK=D13
 *
 * 절차:
 *  1) 업로드 → 시리얼 115200. raw 가 0~16383 사이에서 움직이면 통신 OK.
 *  2) 크랭크를 손으로 잡고 정지 → 'z' (영점). phi_deg 가 0 근처 ±0.05° 미만 잡음이면 OK.
 *  3) ★방향 테스트: 로봇/크랭크 하단을 스윙 정방향(+x, 관례상 오른쪽)으로 밀었을 때
 *     phi_deg 가 + 로 나오는지 확인. − 로 나오면 phi_encoder.h 의 ENC_PHI_DIR = -1.
 *     (이 값은 lqr_balance_v19 / fwe_v19 에도 똑같이 넣는다!)
 *  4) 'n' 을 눌러 1초간 노이즈 표준편차 측정 → params_v19.py ENCODER.NOISE_STD_RAD 갱신.
 */
#include "phi_encoder.h"

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("=== encoder_phi (v19) ===");
  if (!encoderBegin()) Serial.println("!!! 엔코더 응답 없음 — 배선/CS핀/ENCODER_TYPE 확인");
  else                 Serial.println("encoder OK.  z:영점  n:노이즈측정");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'z') { encoderZeroHere(); Serial.println(">>> zero set"); }
    else if (c == 'n') {
      // 1초(500샘플) 잡음 통계 — 완전 정지 상태에서!
      float sum = 0, sq = 0; int n = 500;
      for (int i = 0; i < n; i++) { float v = encoderPhiRad(); sum += v; sq += v * v; delay(2); }
      float mean = sum / n, var = sq / n - mean * mean;
      Serial.print(">>> noise std = ");
      Serial.print(sqrt(max(var, 0.0f)) * 1e6, 1);
      Serial.println(" urad  (params_v19 ENCODER.NOISE_STD_RAD 참고용)");
    }
  }
  static unsigned long lp = 0;
  if (millis() - lp >= 100) {
    lp = millis();
    uint16_t raw = 0; bool ok = encoderReadRaw(raw);
    Serial.print("raw="); Serial.print(raw);
    Serial.print("  phi="); Serial.print(encoderPhiRad() * 180.0f / PI, 3);
    Serial.print(" deg  err_cnt="); Serial.print(enc_err_cnt);
    Serial.println(ok ? "" : "  [READ FAIL]");
  }
}

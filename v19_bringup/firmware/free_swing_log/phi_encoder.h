/*
 * phi_encoder.h — 상단 피벗 φ 절대 엔코더 추상화 (v19)
 * =====================================================
 * 기본: AS5048A 14bit SPI (재설계 스펙 1순위 후보).
 * 엔코더가 다른 모델로 확정되면 이 파일의 해당 타입만 구현 교체하면
 * 모든 스케치(encoder_phi / free_swing_log / lqr_balance_v19 / fwe_v19)가 따라온다.
 *
 * ENCODER_TYPE:
 *   1 = AS5048A (SPI, 기본)      — OpenCR Arduino 커넥터 SPI(D11 MOSI/D12 MISO/D13 SCK) + CS=D10
 *   2 = PWM 출력형 (MA3-P12 등)  — pulseIn 기반. ⚠블로킹이라 '테스트 전용'. 제어 루프엔 SPI 권장.
 *   (AMT223 확정 시: SPI 프로토콜만 이 틀에 맞춰 추가)
 *
 * 사용:
 *   encoderBegin();            // setup()
 *   encoderZeroHere();         // 정지 상태에서 영점 ('z')
 *   float phi = encoderPhiRad();  // 부호는 ENC_PHI_DIR 로 (브링업에서 확정)
 *
 * 부호 규약: 로봇(발)이 +x(스윙 정방향)로 갈 때 φ > 0 이어야 한다.
 *            encoder_phi.ino 의 방향 테스트로 ENC_PHI_DIR 확정.
 */
#pragma once
#include <Arduino.h>

#ifndef ENCODER_TYPE
#define ENCODER_TYPE 1
#endif
#ifndef ENC_CS_PIN
#define ENC_CS_PIN 10
#endif
#ifndef ENC_PWM_PIN
#define ENC_PWM_PIN 2
#endif

static int   ENC_PHI_DIR    = +1;      // ★브링업에서 확정
static float enc_zero_rad   = 0.0f;
static uint32_t enc_err_cnt = 0;

#if ENCODER_TYPE == 1
// ============================ AS5048A (SPI) ============================
#include <SPI.h>
static SPISettings ENC_SPI_SET(1000000, MSBFIRST, SPI_MODE1);

static inline uint16_t as5048a_parity(uint16_t v) {   // even parity of low 15 bits
  v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1;
  return v & 1;
}
static inline uint16_t as5048a_xfer(uint16_t frame) {
  SPI.beginTransaction(ENC_SPI_SET);
  digitalWrite(ENC_CS_PIN, LOW);
  delayMicroseconds(1);
  uint16_t r = SPI.transfer16(frame);
  digitalWrite(ENC_CS_PIN, HIGH);
  SPI.endTransaction();
  delayMicroseconds(1);
  return r;
}

inline bool encoderBegin() {
  pinMode(ENC_CS_PIN, OUTPUT);
  digitalWrite(ENC_CS_PIN, HIGH);
  SPI.begin();
  uint16_t raw;
  extern bool encoderReadRaw(uint16_t &raw);
  return encoderReadRaw(raw);
}

// 각도 레지스터 0x3FFF 읽기 (READ 플래그 0x4000 + 짝수패리티 bit15)
inline bool encoderReadRaw(uint16_t &raw) {
  uint16_t cmd = 0x3FFF | 0x4000;
  cmd |= as5048a_parity(cmd) << 15;
  as5048a_xfer(cmd);                 // 요청 프레임
  uint16_t resp = as5048a_xfer(cmd); // 응답 프레임
  if (resp & 0x4000) {               // 에러 플래그 → 에러 레지스터 클리어
    uint16_t clr = 0x0001 | 0x4000;
    clr |= as5048a_parity(clr) << 15;
    as5048a_xfer(clr); as5048a_xfer(clr);
    enc_err_cnt++;
    return false;
  }
  raw = resp & 0x3FFF;               // 14bit
  return true;
}

#elif ENCODER_TYPE == 2
// ====================== PWM (MA3 등) — 테스트 전용 ======================
inline bool encoderBegin() { pinMode(ENC_PWM_PIN, INPUT); return true; }
inline bool encoderReadRaw(uint16_t &raw) {
  // MA3-P12: 주기 ~4098us, 듀티 = 각도. pulseIn 은 최대 ~4ms 블로킹.
  unsigned long hi = pulseIn(ENC_PWM_PIN, HIGH, 6000);
  if (hi == 0) { enc_err_cnt++; return false; }
  unsigned long lo = pulseIn(ENC_PWM_PIN, LOW, 6000);
  if (lo == 0) { enc_err_cnt++; return false; }
  raw = (uint16_t)(16383.0f * (float)hi / (float)(hi + lo));
  return true;
}
#endif

// ============================ 공통 API ============================
inline float encoderAngleAbsRad() {         // 0 ~ 2π (영점 미적용)
  uint16_t raw;
  static float last = 0.0f;
  if (!encoderReadRaw(raw)) return last;    // 실패 시 직전값 유지
  last = raw * (2.0f * PI / 16384.0f);
  return last;
}

inline void encoderZeroHere() { enc_zero_rad = encoderAngleAbsRad(); }

inline float encoderPhiRad() {              // 영점 기준 ±π 랩, 부호 적용
  float d = encoderAngleAbsRad() - enc_zero_rad;
  while (d >  PI) d -= 2.0f * PI;
  while (d < -PI) d += 2.0f * PI;
  return ENC_PHI_DIR * d;
}

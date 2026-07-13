/*
 * imu_axis_check.ino — [실측 3단계] IMU 단위·부호 점검 (v19)
 * ============================================================
 * 7/02 에 확정된 자이로 단위 버그(imu.gyroData[] = raw LSB, deg/s 아님!)의
 * 재발 방지용 상시 점검 스케치. 새 보드/새 라이브러리에서 반드시 1회 실행.
 *
 * 확인 3가지:
 *  [1] 단위: 로봇(보드)을 약 1초에 90° 돌렸을 때 gy_dps ≈ 90 (±30) 인가?
 *      raw 열이 dps 열의 ~16.4배로 나오는 것이 정상(0.061 dps/LSB).
 *      만약 gy_dps 가 수백~수천이면 라이브러리 단위 문제 — 제어 코드 금지.
 *  [2] 부호: 로봇 상체를 '앞으로'(δ 앞접기와 같은 쪽) 기울이면 thAcc 가 + 인가?
 *      − 면 제어 스케치의 IMU_DIR = -1.
 *  [3] 정지 바이어스: 정지 시 gy_dps 표류가 ±0.5 dps 안인가 (제어 코드가 부팅 시 제거).
 */
#include <IMU.h>

cIMU imu;
float mount_off = 0;

void setup() {
  Serial.begin(115200);
  delay(300);
  imu.begin(500);
  Serial.println("=== imu_axis_check (v19) ===");
  Serial.println("z:자세 영점 | [1]1초에 90° 회전→gy_dps~90 [2]앞기울임→thAcc+ [3]정지 바이어스");
  delay(1000);
}

void loop() {
  imu.update();
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'z') {
      float ax = imu.accData[0], az = imu.accData[2];
      mount_off = -90.0f * DEG_TO_RAD - atan2(ax, az);
      Serial.println(">>> mount zero set");
    }
  }
  static unsigned long lp = 0;
  if (millis() - lp >= 50) {
    lp = millis();
    float ax = imu.accData[0], az = imu.accData[2];
    float thAcc = (-90.0f * DEG_TO_RAD - atan2(ax, az)) - mount_off;
    Serial.print("thAcc=");  Serial.print(thAcc * RAD_TO_DEG, 1);
    Serial.print("  gy_dps=");   Serial.print(imu.gy, 1);            // ★제어에 쓰는 값
    Serial.print("  gyRaw=");    Serial.print(imu.gyroData[1]);      // raw LSB (참고)
    Serial.print("  raw/dps=");  Serial.print(imu.gy != 0 ? imu.gyroData[1] / imu.gy : 0, 1);
    Serial.println("  (16.4 근처면 정상)");
  }
}

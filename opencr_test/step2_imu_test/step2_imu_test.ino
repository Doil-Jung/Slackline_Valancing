/*
 * ============================================
 * Step 2: 내장 IMU 테스트
 * ============================================
 * 
 * 목적: IMU(ICM-20648) 값 읽기, Roll/Pitch 확인
 * 전원: USB만으로 충분
 * 모터: 연결 불필요
 * 
 * 확인 사항:
 *   - Roll, Pitch 값이 보드 기울기에 따라 변하는지
 *   - 보드를 수평으로 놓으면 0도 근처인지
 *   - 기울이면 각도가 변하는지
 */

#include <IMU.h>

cIMU imu;

// 이동평균 필터용
#define FILTER_SIZE 10
float roll_buf[FILTER_SIZE] = {0};
float pitch_buf[FILTER_SIZE] = {0};
int filter_idx = 0;

unsigned long prev_print_time = 0;
const unsigned long PRINT_INTERVAL = 50;  // 50ms = 20Hz 출력

void setup() {
  Serial.begin(115200);
  
  // IMU 초기화
  imu.begin();
  
  Serial.println("========================================");
  Serial.println("  IMU Test - Step 2");
  Serial.println("  OpenCR 내장 IMU (ICM-20648)");
  Serial.println("========================================");
  Serial.println();
  Serial.println("보드를 기울여 보세요!");
  Serial.println("Roll = 좌우 기울기, Pitch = 앞뒤 기울기");
  Serial.println();
  Serial.println("출력 형식: Roll(도), Pitch(도), GyroX, GyroY, GyroZ");
  Serial.println("----------------------------------------");
  
  delay(2000);  // IMU 안정화 대기
}

void loop() {
  // IMU 업데이트 (매 루프마다 호출 필요)
  imu.update();
  
  // 이동평균 필터 적용
  roll_buf[filter_idx] = imu.rpy[0];
  pitch_buf[filter_idx] = imu.rpy[1];
  filter_idx = (filter_idx + 1) % FILTER_SIZE;
  
  float roll_avg = 0, pitch_avg = 0;
  for (int i = 0; i < FILTER_SIZE; i++) {
    roll_avg += roll_buf[i];
    pitch_avg += pitch_buf[i];
  }
  roll_avg /= FILTER_SIZE;
  pitch_avg /= FILTER_SIZE;
  
  // 주기적 출력 (시리얼 플로터에서도 볼 수 있도록)
  unsigned long now = millis();
  if (now - prev_print_time >= PRINT_INTERVAL) {
    prev_print_time = now;
    
    // 시리얼 모니터용 출력
    Serial.print("Roll: ");
    Serial.print(roll_avg, 1);
    Serial.print("°\tPitch: ");
    Serial.print(pitch_avg, 1);
    Serial.print("°\tGyroX: ");
    Serial.print(imu.gyroData[0], 1);
    Serial.print("\tGyroY: ");
    Serial.print(imu.gyroData[1], 1);
    Serial.print("\tGyroZ: ");
    Serial.print(imu.gyroData[2], 1);
    Serial.println();
    
    // LED로 기울기 방향 표시
    // Roll > 10도: LED1 점등 (왼쪽)
    // Roll < -10도: LED4 점등 (오른쪽)
    digitalWrite(22, roll_avg > 10.0 ? HIGH : LOW);   // LED1
    digitalWrite(25, roll_avg < -10.0 ? HIGH : LOW);  // LED4
    // Pitch > 10도: LED2 점등 (앞쪽)
    // Pitch < -10도: LED3 점등 (뒤쪽)
    digitalWrite(23, pitch_avg > 10.0 ? HIGH : LOW);  // LED2
    digitalWrite(24, pitch_avg < -10.0 ? HIGH : LOW); // LED3
  }
}

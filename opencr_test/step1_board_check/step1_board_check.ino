/*
 * ============================================
 * Step 1: OpenCR 보드 기본 동작 확인
 * ============================================
 * 
 * 목적: 보드 연결, 업로드, 시리얼 통신 확인
 * 전원: USB만으로 충분 (배터리/SMPS 불필요)
 * 모터: 연결 불필요
 * 
 * 확인 사항:
 *   - USER LED 깜박임
 *   - 시리얼 모니터 출력
 *   - 보드 정상 동작
 */

#define USER_LED_1  22   // OpenCR USER LED 1
#define USER_LED_2  23   // OpenCR USER LED 2
#define USER_LED_3  24   // OpenCR USER LED 3
#define USER_LED_4  25   // OpenCR USER LED 4

#define USER_BUTTON 7    // OpenCR USER Button

int count = 0;

void setup() {
  Serial.begin(115200);
  
  // LED 핀 설정
  pinMode(USER_LED_1, OUTPUT);
  pinMode(USER_LED_2, OUTPUT);
  pinMode(USER_LED_3, OUTPUT);
  pinMode(USER_LED_4, OUTPUT);
  pinMode(USER_BUTTON, INPUT_PULLDOWN);
  
  // 시작 메시지
  Serial.println("========================================");
  Serial.println("  OpenCR Board Check - Step 1");
  Serial.println("  Slackline Balancing Robot Project");
  Serial.println("========================================");
  Serial.println();
  Serial.println("LED가 순서대로 깜박이면 정상입니다!");
  Serial.println("USER 버튼을 누르면 메시지가 출력됩니다.");
  Serial.println();
}

void loop() {
  // LED 순차 점등 (나이트라이더 효과)
  for (int i = 0; i < 4; i++) {
    digitalWrite(USER_LED_1 + i, HIGH);
    delay(100);
    digitalWrite(USER_LED_1 + i, LOW);
  }
  for (int i = 2; i >= 1; i--) {
    digitalWrite(USER_LED_1 + i, HIGH);
    delay(100);
    digitalWrite(USER_LED_1 + i, LOW);
  }
  
  // 카운터 출력
  count++;
  Serial.print("[Step1] 보드 정상 동작 중... count = ");
  Serial.println(count);
  
  // 버튼 체크
  if (digitalRead(USER_BUTTON) == HIGH) {
    Serial.println(">>> USER 버튼 눌림! 보드 입력 정상!");
    
    // 모든 LED 점등
    digitalWrite(USER_LED_1, HIGH);
    digitalWrite(USER_LED_2, HIGH);
    digitalWrite(USER_LED_3, HIGH);
    digitalWrite(USER_LED_4, HIGH);
    delay(500);
    digitalWrite(USER_LED_1, LOW);
    digitalWrite(USER_LED_2, LOW);
    digitalWrite(USER_LED_3, LOW);
    digitalWrite(USER_LED_4, LOW);
  }
}

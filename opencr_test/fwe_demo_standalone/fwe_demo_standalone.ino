/*
 * ============================================================
 *  FWE 데모 — STANDALONE (USB 없이 줄 위 시연용)
 *  OpenCR 1.0 + Dynamixel XM430-W210-R
 * ============================================================
 *  fwe_demo.ino의 시연 특화판. 차이점:
 *   1) 부팅 시 토크 OFF로 시작 → 손으로 곧게 편 뒤 0점 잡음(잠금 문제 해결).
 *   2) OpenCR 온보드 버튼/LED로 USB 없이 조작:
 *        SW1 = 0점+시작(ARM).  SW2 = 수동 앞접기 한 사이클.
 *        LED1 = 깜빡(자세 잡아라) / 켜짐(ARM됨).  LED2 = 사이클 중.
 *   3) ARM 후 IMU 자동트리거 기본 ON → 기울이면 그쪽으로 FWE 사이클.
 *        (타이밍 맞출 필요 없음: 로봇이 기울임에 반응.)
 *   USB를 꽂으면 시리얼 명령도 그대로 동작(튜닝용).
 *
 *  ★시연 순서(USB 없이):
 *    배터리 ON → LED1 깜빡 → 로봇을 줄에 얹고 곧게 잡기 → SW1 누름(0점+ARM)
 *    → 한쪽으로 기울이면 그쪽으로 접었다 폄. (SW2로 수동 발동도 가능)
 */

#include <Dynamixel2Arduino.h>
#include <IMU.h>

#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
#define DXL_BAUD     1000000   // ★v2가 모터를 1Mbps로 승격시켜 놓았으므로 맞춤 (구 57600)
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;
cIMU imu;

const float DEG2RAD = PI/180.0f, RAD2DEG = 180.0f/PI;

// === 온보드 버튼/LED (OpenCR) ===
//  버튼: 눌리면 HIGH.  LED: 보드에 따라 active-LOW면 LED_ON=LOW.
//  LED가 반대로 동작하면 아래 두 줄 LOW/HIGH를 맞바꿀 것.
#define LED_ON   LOW
#define LED_OFF  HIGH
const int SW1 = BDPIN_PUSH_SW_1;
const int SW2 = BDPIN_PUSH_SW_2;
const int LED1 = BDPIN_LED_USER_1;
const int LED2 = BDPIN_LED_USER_2;

// === FWE 파라미터 (시연용 기본값) ===
float fold_deg   = 35.0f;
int   wait_ms    = 300;
int   fold_ms    = 150;
int   expand_ms  = 150;
int   profile_vel= 0;        // 0=최대속도
const float FOLD_MAX = 50.0f;

bool  armed = false;
bool  auto_trig = true;      // ARM 후 기울이면 발동
int   trig_mode = 2;         // ★2=상보필터(실측상 최선) 1=자이로(빠른회전서 반동샘플링 위험) 0=가속도계각도
float trig_deg  = 5.0f;      // ★각도 모드(0,2) 임계 [deg]. 낮출수록 빨리 발동(너무 낮으면 흔들림에 오발동)
float rate_thresh = 25.0f;   // 자이로 모드(1) 임계 [deg/s]
int   TRIG_SIGN = +1;        // 자이로 부호↔접기방향. 손으로 앞 기울일때 뒤로 접으면 -1로
// 상보필터(모드2): 단기 자이로 + 장기 가속도계
float cf_angle = 0.0f; unsigned long cf_prev = 0;
float CF_A = 0.98f;          // 자이로 신뢰도(높일수록 가속도 오염에 강하나 드리프트↑)
int   CF_GYRO_SIGN = +1;     // CF 자이로↔가속도각 부호일치용. CF가 발산(가속도각과 멀어짐)하면 -1
unsigned long last_cycle_ms = 0;
int COOLDOWN_MS = 600;       // ★사이클 후 재발동 금지 [ms]. 낮출수록 연속반응 빠름
bool  loop_mode = false; int loop_dir = +1;

int32_t enc0 = 0;
float imu_mount = 0.0f;

void ledWrite(int pin, bool on){ digitalWrite(pin, on?LED_ON:LED_OFF); }

void gotoDelta(float delta_deg){
  delta_deg = constrain(delta_deg, -FOLD_MAX-5, FOLD_MAX+5);
  int32_t raw = enc0 + (int32_t)(delta_deg*DEG2RAD*4096.0f/(2.0f*PI));
  dxl.setGoalPosition(DXL_ID, raw);
}
float getDeltaDeg(){
  int32_t raw = dxl.getPresentPosition(DXL_ID);
  return (float)(raw-enc0)*(2.0f*PI/4096.0f)*RAD2DEG;
}
float getThetaDeg(){
  float ax=imu.accData[0], az=imu.accData[2];
  return (-90.0f*DEG2RAD - atan2(ax,az) - imu_mount)*RAD2DEG;
}
// 상보필터 각도 업데이트(매 루프): 단기 자이로 적분 + 장기 가속도각 보정
void updateCF(){
  unsigned long now=millis();
  float dt=(cf_prev==0)?0.005f:(now-cf_prev)*0.001f; cf_prev=now;
  if(dt<=0||dt>0.1f) dt=0.005f;
  float gy = CF_GYRO_SIGN * imu.gyroData[1];   // deg/s
  float acc = getThetaDeg();                    // deg (가속도각)
  cf_angle = CF_A*(cf_angle + gy*dt) + (1.0f-CF_A)*acc;
}
void setProfile(int v){
  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, v);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, DXL_ID, 0);
}

// 토크 켜고 현재(곧게 편) 자세를 δ=0으로
void armZero(){
  dxl.torqueOff(DXL_ID); delay(30);
  dxl.setOperatingMode(DXL_ID, OP_EXTENDED_POSITION); delay(30);
  enc0 = dxl.getPresentPosition(DXL_ID);
  setProfile(profile_vel);
  dxl.writeControlTableItem(GOAL_POSITION, DXL_ID, enc0);
  dxl.torqueOn(DXL_ID);
  imu_mount = -90.0f*DEG2RAD - atan2(imu.accData[0], imu.accData[2]);
  cf_angle = 0.0f; cf_prev = 0;   // 직립 기준에서 CF 초기화
  armed = true; last_cycle_ms = millis();
  ledWrite(LED1, true);
  Serial.println(">>> ARMED (delta=0 set, torque on). 기울이면 FWE 발동.");
}

void foldCycle(int dir){
  float d = constrain(fold_deg,5.0f,FOLD_MAX);
  ledWrite(LED2, true);
  setProfile(profile_vel);
  gotoDelta(dir*d); delay(fold_ms);   // 접기
  delay(wait_ms);                     // 기다리기
  gotoDelta(0);     delay(expand_ms); // 펴기
  ledWrite(LED2, false);
  last_cycle_ms = millis();
  Serial.print(">>> FWE cycle "); Serial.println(dir>0?"FWD":"BACK");
}

// 버튼 엣지 검출(간단 디바운스)
bool pressed(int pin, bool &prev){
  bool now = (digitalRead(pin)==HIGH);
  bool edge = (now && !prev);
  prev = now;
  if(edge) delay(20); // 디바운스
  return edge;
}

void setup(){
  Serial.begin(115200);
  pinMode(SW1, INPUT); pinMode(SW2, INPUT);
  pinMode(LED1, OUTPUT); pinMode(LED2, OUTPUT);
  ledWrite(LED1,false); ledWrite(LED2,false);

  imu.begin();
  dxl.begin(DXL_BAUD); dxl.setPortProtocolVersion(2.0);
  bool found=false;
  for(int i=0;i<5;i++){ if(dxl.ping(DXL_ID)){found=true;break;} delay(300);}
  if(found){ dxl.torqueOff(DXL_ID); }   // ★시작은 토크 OFF: 손으로 곧게 펼 수 있게
  for(int i=0;i<150;i++){ imu.update(); delay(3);}  // IMU 안정화

  Serial.println("=== FWE DEMO standalone ===");
  Serial.println("로봇 곧게 잡고 SW1 → ARM. 기울이면 접음. SW2=수동.");
  if(!found) Serial.println("!!! Motor not found (계속 진행은 하나 모터 무반응)");
}

unsigned long blink_t=0; bool blink_on=false; bool p1=false,p2=false;
bool mon=false; unsigned long mon_t=0;

void loop(){
  imu.update();
  updateCF();   // 상보필터 각도 항상 갱신

  // --- 온보드 버튼 ---
  if(pressed(SW1,p1)){ armZero(); }
  if(pressed(SW2,p2)){ if(armed) foldCycle(+1); }

  // --- 시리얼(USB 연결 시 튜닝용) ---
  if(Serial.available()){
    char c=Serial.read();
    if      (c=='z') armZero();
    else if (c=='f'){ if(armed) foldCycle(+1);}
    else if (c=='b'){ if(armed) foldCycle(-1);}
    else if (c=='a'){ auto_trig=!auto_trig; Serial.print("auto="); Serial.println(auto_trig);}
    else if (c=='g'){ trig_mode=(trig_mode+1)%3; Serial.print("trig="); Serial.println(trig_mode==1?"GYRO":trig_mode==2?"CF":"accel");}
    else if (c=='m'){ mon=!mon; Serial.print("mon="); Serial.println(mon);}   // 연속 모니터 토글
    else if (c=='j'){ trig_deg=constrain(trig_deg-1,2,30); Serial.print("trig_deg="); Serial.println(trig_deg,0);}  // 더 민감(빨리)
    else if (c=='k'){ trig_deg=constrain(trig_deg+1,2,30); Serial.print("trig_deg="); Serial.println(trig_deg,0);}  // 덜 민감
    else if (c=='c'){ COOLDOWN_MS=constrain(COOLDOWN_MS-100,200,3000); Serial.print("cooldown="); Serial.println(COOLDOWN_MS);} // 연속반응 빠르게
    else if (c=='v'){ COOLDOWN_MS=constrain(COOLDOWN_MS+100,200,3000); Serial.print("cooldown="); Serial.println(COOLDOWN_MS);}
    else if (c=='l'){ loop_mode=!loop_mode; Serial.print("loop="); Serial.println(loop_mode);}
    else if (c=='+'||c=='='){ fold_deg=constrain(fold_deg+5,5,FOLD_MAX); Serial.print("fold="); Serial.println(fold_deg,0);}
    else if (c=='-'||c=='_'){ fold_deg=constrain(fold_deg-5,5,FOLD_MAX); Serial.print("fold="); Serial.println(fold_deg,0);}
    else if (c=='w'){ wait_ms=constrain(wait_ms+100,0,3000); Serial.print("wait="); Serial.println(wait_ms);}
    else if (c=='s'){ wait_ms=constrain(wait_ms-100,0,3000); Serial.print("wait="); Serial.println(wait_ms);}
    else if (c=='['){ profile_vel=constrain(profile_vel+20,0,500); setProfile(profile_vel);}
    else if (c==']'){ profile_vel=constrain(profile_vel-20,0,500); setProfile(profile_vel);}
    else if (c=='x'){ loop_mode=false; auto_trig=false; if(armed) gotoDelta(0); Serial.println("STOP");}
  }

  // 연속 모니터(빠른회전 캡처용): gyro / acc각 / cf각 ~100Hz
  if(mon && millis()-mon_t>=10){
    mon_t=millis();
    Serial.print(millis()); Serial.print(",");
    Serial.print((float)imu.gyroData[1],1); Serial.print(",");   // 자이로 각속도 deg/s (★float캐스트: 음수 부호 정상출력)
    Serial.print(getThetaDeg(),1); Serial.print(",");     // 가속도각 deg
    Serial.println(cf_angle,1);                            // CF각 deg
  }

  if(!armed){
    // 자세 잡아라: LED1 깜빡
    if(millis()-blink_t>300){ blink_t=millis(); blink_on=!blink_on; ledWrite(LED1,blink_on);}
    delay(2); return;
  }

  // --- ARM 후 ---
  // IMU 자동트리거
  if(auto_trig && (millis()-last_cycle_ms>COOLDOWN_MS)){
    if(trig_mode==1){
      // ★자이로(각속도) — 선형가속 면역. 줄 위에서 기울기 시작하면 그쪽으로.
      float w = TRIG_SIGN * imu.gyroData[1];   // deg/s
      if(fabs(w) > rate_thresh) foldCycle(w>0?+1:-1);
    } else if(trig_mode==2){
      // 상보필터 각도 — 절대각(단기 자이로 우세). 지속 운동시 가속도각으로 끌릴 수 있음.
      if(fabs(cf_angle) > trig_deg) foldCycle(cf_angle>0?+1:-1);
    } else {
      // 가속도계 각도 — 손으로 들고 준정적일 때만 신뢰(줄 위에선 반대로 나옴)
      float th=getThetaDeg();
      if(fabs(th)>trig_deg) foldCycle(th>0?+1:-1);
    }
  }
  // 리듬 반복(선택)
  if(loop_mode && (millis()-last_cycle_ms > (unsigned)(fold_ms+wait_ms+expand_ms+600))){
    foldCycle(loop_dir); loop_dir=-loop_dir;
  }
  delay(2);
}

/*
  2026 전람회 "줄타기 밸런싱 로봇" - V11 시뮬레이션용 간소화 모델
  
  원본 대비 변경사항:
  - 전선(베지어 곡선 배선) 모두 제거
  - 나사/볼트 시각화 모두 제거  
  - 모터 나사 구멍 관통 시각화 제거
  - 큰 부품만 유지: 상체 프레임, 하체 프레임, 모터, U브라켓, 
    ESP32, URT1, LiPo배터리, 보조배터리, MPU6050 x2
*/

$fn = 50; 
tilt_angle = 30 * sin($t * 360); 

// ==========================================
// ⚙️ 하드웨어 치수 파라미터 
// ==========================================
upper_length = 200; lower_length = 200; col_y_target = 60.0; 

// [모터 및 브라켓]
motor_x = 24.7; motor_y = 35.0; motor_z_up = 33.7; motor_z_down = 11.5; motor_z = 45.2;
horn_d = 20.0; horn_h = 3.0;
b_inner_y = 42.0; b_w = 24.0; b_arm_l = 30.0; b_thick = 2.0; 
rope_d = 2.0; 

// ⭐ [전자부품 실측 치수]
esp_w = 50.0; esp_h = 65.0; esp_d = 12.0;
urt_w = 37.0; urt_h = 56.0; urt_d = 12.0;
bat_w = 30.0; bat_h = 61.0; bat_d = 15.0;  // 7.4V LiPo (모터 구동 전용)
pb_d  = 22.0; pb_h  = 92.0;                // 5V 보조 배터리 (ESP32 전용)
mpu_w = 16.0; mpu_h = 21.0; mpu_d = 3.0;

plate_t = 4.0; 
up_x_fixed = motor_x + (plate_t * 2); 
low_x_fixed = b_w;                    

view_mode = 0; 
b_bot_z = -b_arm_l - b_thick;
rope_z = b_bot_z - lower_length + 2.0; 

// ==========================================
// 👁️ 렌더링 및 애니메이션
// ==========================================
if (view_mode == 0) {
    // 줄 (슬랙라인)
    color("DarkOrange") translate([0, 0, rope_z]) rotate([90, 0, 0]) 
        cylinder(d=rope_d, h=400, center=true);
    
    // 하체 어셈블리 (줄에 매달림 + 틸트)
    translate([0, 0, rope_z]) rotate([0, tilt_angle, 0]) translate([0, 0, -rope_z]) {
        color("DimGray", 0.4) lower_body();
        color("Silver") u_bracket_dummy();
        lower_electronics(); 
        
        // 상체 어셈블리 (힙 관절 회전)
        U_rot = -tilt_angle * 1.5;
        rotate([0, U_rot, 0]) { 
            color("LightSkyBlue", 0.4) upper_body(); 
            motor_dummy(); 
            upper_electronics(); 
        }
    }
} else if (view_mode == 1) upper_body();
  else if (view_mode == 2) lower_body();

// ==========================================
// 🧱 상체 프레임
// ==========================================
module upper_body() {
    difference() {
        union() {
            z_center = (motor_z_up - motor_z_down)/2;
            // 좌우 측판
            translate([motor_x/2 + plate_t/2, 0, z_center]) 
                cube([plate_t, motor_y + 4, motor_z], center=true);
            translate([-motor_x/2 - plate_t/2, 0, z_center]) 
                cube([plate_t, motor_y + 4, motor_z], center=true);
            // 상판
            translate([0, 0, motor_z_up + plate_t/2]) 
                cube([up_x_fixed, motor_y + 4, plate_t], center=true);
            // 테이퍼 구간
            hull() {
                translate([0, 0, motor_z_up + plate_t]) 
                    cube([up_x_fixed, motor_y + 4, 0.1], center=true);
                translate([0, 0, motor_z_up + 20]) 
                    cube([up_x_fixed, col_y_target, 0.1], center=true);
            }
            // 칼럼 본체
            h_len = upper_length - (motor_z_up + 20);
            translate([0, 0, motor_z_up + 20 + h_len/2]) 
                cube([up_x_fixed, col_y_target, h_len], center=true);
        }
        
        // 모터축 관통 구멍
        translate([0, 0, motor_z_up + plate_t/2]) 
            cylinder(d=16, h=plate_t + 5, center=true);
        
        // 내부 중공
        h_len2 = upper_length - (motor_z_up + 10);
        translate([0, 0, motor_z_up + 10 + h_len2/2]) 
            cube([up_x_fixed - 8, col_y_target - 8, h_len2], center=true);
            
        // 환기/경량화 슬릿
        for(z = [motor_z_up + 40 : 40 : upper_length - 20]) {
            translate([0, -col_y_target/2, z]) 
                cube([up_x_fixed + 10, 4, 3], center=true);
            translate([0,  col_y_target/2, z]) 
                cube([up_x_fixed + 10, 4, 3], center=true);
        }
    }
}

// ==========================================
// 🧱 하체 프레임
// ==========================================
module lower_body() {
    translate([0, 0, b_bot_z])
    difference() {
        union() {
            // 상판 (브라켓 체결부)
            translate([0, 0, -2]) 
                cube([low_x_fixed, b_inner_y + b_thick*2, 4], center=true);
            // 테이퍼 구간
            hull() {
                translate([0, 0, -4]) 
                    cube([low_x_fixed, b_inner_y + b_thick*2, 0.1], center=true);
                translate([0, 0, -24]) 
                    cube([low_x_fixed, col_y_target, 0.1], center=true);
            }
            // 칼럼 본체
            h_len = lower_length - 24;
            translate([0, 0, -24 - h_len/2]) 
                cube([low_x_fixed, col_y_target, h_len], center=true);
        }
        
        // 내부 중공
        h_len2 = lower_length - 24 - 20;
        translate([0, 0, -24 - h_len2/2]) 
            cube([low_x_fixed - 8, col_y_target - 8, h_len2], center=true);
        
        // 바닥 V형 노치 (줄 접촉)
        translate([0, 0, -lower_length - 1.5]) rotate([0, 45, 0]) 
            cube([6, col_y_target + 10, 6], center=true);
    }
}

// ==========================================
// 🔩 모터 본체 (나사 제거)
// ==========================================
module motor_dummy() {
    z_center = (motor_z_up - motor_z_down)/2;
    // 모터 본체
    color("DarkSlateGray") translate([0, 0, z_center]) 
        cube([motor_x, motor_y, motor_z], center=true);
    // 서보 혼 (양쪽)
    color("Gold") {
        translate([0, motor_y/2 + 1.5, 0]) rotate([90, 0, 0]) 
            cylinder(d=horn_d, h=horn_h, center=true);
        translate([0, -motor_y/2 - 1.5, 0]) rotate([90, 0, 0]) 
            cylinder(d=horn_d, h=horn_h, center=true);
    }
}

// ==========================================
// 🔩 U자 브라켓 (볼트 제거)
// ==========================================
module u_bracket_dummy() {
    color("Silver") {
        // 바닥판
        translate([0, 0, b_bot_z + b_thick/2]) 
            cube([b_w, b_inner_y + 2*b_thick, b_thick], center=true);
        // 좌측 팔
        translate([0, b_inner_y/2 + b_thick/2, -b_arm_l/2]) 
            cube([b_w, b_thick, b_arm_l], center=true);
        // 우측 팔
        translate([0, -b_inner_y/2 - b_thick/2, -b_arm_l/2]) 
            cube([b_w, b_thick, b_arm_l], center=true);
    }
}

// ==========================================
// 📦 상체 전자부품 (전선 없이 부품만)
// ==========================================
module upper_electronics() {
    // +X면: ESP32 (상단)
    color("ForestGreen") translate([up_x_fixed/2 + esp_d/2 + 0.1, 0, 160]) 
        cube([esp_d, esp_w, esp_h], center=true);
    
    // +X면: URT1 (하단)
    color("Goldenrod") translate([up_x_fixed/2 + urt_d/2 + 0.1, 0, 90]) 
        cube([urt_d, urt_w, urt_h], center=true);
    
    // -X면 (등판): 원통형 5V 보조배터리 (상단)
    color("White") translate([-up_x_fixed/2 - pb_d/2 - 0.1, 0, 155]) 
        cylinder(d=pb_d, h=pb_h, center=true);
    
    // -X면 (등판): 7.4V LiPo 배터리 (하단)
    color("RoyalBlue") translate([-up_x_fixed/2 - bat_d/2 - 0.1, 0, 65]) 
        cube([bat_d, bat_w, bat_h], center=true);
    
    // +Y면: 상체 MPU6050
    color("Crimson") translate([0, col_y_target/2 + mpu_d/2 + 0.1, 60]) 
        cube([mpu_w, mpu_d, mpu_h], center=true);
}

// ==========================================
// 📦 하체 전자부품
// ==========================================
module lower_electronics() {
    // +Y면: 하체 MPU6050
    color("Crimson") translate([0, col_y_target/2 + mpu_d/2 + 0.1, b_bot_z - 40]) 
        cube([mpu_w, mpu_d, mpu_h], center=true);
}

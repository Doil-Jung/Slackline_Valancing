/*
  2026 전람회 "줄타기 밸런싱 로봇" - PyBullet 시뮬레이션 전용 모델
  - 목적: 물리 엔진(URDF) 업로드를 위한 강체(Rigid Body) Mesh 생성
  - export_mode = 1 : 상체 덩어리 (프레임+모터바디+보드x2+배터리x2+센서)
  - export_mode = 2 : 하체 덩어리 (프레임+금속브라켓+모터혼+센서)
  - 조치사항: 전선 및 가상의 나사 핀 제거, 부품 통짜 Union 병합
*/

$fn = 30; // 시뮬레이터 연산 부하를 줄이기 위해 곡면 해상도를 30으로 최적화

// ==========================================
// 🚀 STL Export 설정 (매우 중요)
// ==========================================
// ⭐ 1로 설정하고 F6 누른 뒤 STL 저장 -> upper_link.stl
// ⭐ 2로 설정하고 F6 누른 뒤 STL 저장 -> lower_link.stl
// (0으로 설정하면 조립된 모습을 미리보기 할 수 있습니다.)
export_mode = 1; 

// ==========================================
// ⚙️ 하드웨어 치수 파라미터 
// ==========================================
upper_length = 200; lower_length = 200; col_y_target = 60.0; 

// [모터 및 브라켓]
motor_x = 24.7; motor_y = 35.0; motor_z_up = 33.7; motor_z_down = 11.5; motor_z = 45.2;
m_hole_y = 14.0; m_hole_z_positions = [6.0, 26.0]; 
horn_d = 20.0; horn_h = 3.0;

b_inner_y = 42.0; b_w = 24.0; b_arm_l = 30.0; b_thick = 2.0; 
b_hole_x = 14.0; b_hole_y = 20.0; 
rope_d = 2.0; 

// [전자부품 실측]
esp_w = 50.0; esp_h = 65.0; esp_d = 12.0;
urt_w = 37.0; urt_h = 56.0; urt_d = 12.0;
bat_w = 30.0; bat_h = 61.0; bat_d = 15.0;
pb_d  = 22.0; pb_h  = 92.0;
mpu_w = 16.0; mpu_h = 21.0; mpu_d = 3.0;

plate_t = 4.0; 
up_x_fixed = motor_x + (plate_t * 2); 
low_x_fixed = b_w;                    
b_bot_z = -b_arm_l - b_thick;

// ==========================================
// 👁️ 메인 병합 로직
// ==========================================
if (export_mode == 1) {
    // [상체 강체 그룹]
    union() {
        upper_body_mesh();         
        motor_body_mesh();         
        upper_electronics_mesh();  
    }
} else if (export_mode == 2) {
    // [하체 강체 그룹]
    union() {
        lower_body_mesh();         
        bracket_mesh();            
        horn_mesh();               
        lower_electronics_mesh();  
    }
} else if (export_mode == 0) {
    // [어셈블리 미리보기]
    color("LightSkyBlue") { upper_body_mesh(); motor_body_mesh(); upper_electronics_mesh(); }
    color("DimGray") { lower_body_mesh(); bracket_mesh(); horn_mesh(); lower_electronics_mesh(); }
}

// ==========================================
// 🧱 세부 파트 모델링 (물리 충돌 특성 유지를 위해 뼈대 구멍은 보존)
// ==========================================
module upper_body_mesh() {
    difference() {
        union() {
            z_center = (motor_z_up - motor_z_down)/2;
            translate([motor_x/2 + plate_t/2, 0, z_center]) cube([plate_t, motor_y + 4, motor_z], center=true);
            translate([-motor_x/2 - plate_t/2, 0, z_center]) cube([plate_t, motor_y + 4, motor_z], center=true);
            translate([0, 0, motor_z_up + plate_t/2]) cube([up_x_fixed, motor_y + 4, plate_t], center=true);
            hull() {
                translate([0, 0, motor_z_up + plate_t]) cube([up_x_fixed, motor_y + 4, 0.1], center=true);
                translate([0, 0, motor_z_up + 20]) cube([up_x_fixed, col_y_target, 0.1], center=true);
            }
            h_len = upper_length - (motor_z_up + 20);
            translate([0, 0, motor_z_up + 20 + h_len/2]) cube([up_x_fixed, col_y_target, h_len], center=true);
        }
        for(x = [up_x_fixed/2, -up_x_fixed/2]) for(y = [-m_hole_y/2, m_hole_y/2]) for(z = m_hole_z_positions)
            hull() {
                translate([x, y, z - 2.5]) rotate([0, 90, 0]) cylinder(d=2.6, h=10, center=true);
                translate([x, y, z + 2.5]) rotate([0, 90, 0]) cylinder(d=2.6, h=10, center=true);
            }
        translate([0, 0, motor_z_up + plate_t/2]) cylinder(d=16, h=plate_t + 5, center=true);
        h_len2 = upper_length - (motor_z_up + 10);
        translate([0, 0, motor_z_up + 10 + h_len2/2]) cube([up_x_fixed - 8, col_y_target - 8, h_len2], center=true);
            
        for(z = [motor_z_up + 40 : 40 : upper_length - 20]) {
            translate([0, -col_y_target/2, z]) cube([up_x_fixed + 10, 4, 3], center=true);
            translate([0,  col_y_target/2, z]) cube([up_x_fixed + 10, 4, 3], center=true);
        }
        translate([0, col_y_target/2, 60]) {
            translate([0, 0, 7.5]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
            translate([0, 0, -7.5]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
        }
    }
}

module motor_body_mesh() {
    z_center = (motor_z_up - motor_z_down)/2;
    translate([0, 0, z_center]) cube([motor_x, motor_y, motor_z], center=true);
}

module upper_electronics_mesh() {
    translate([up_x_fixed/2 + esp_d/2 + 0.1, 0, 160]) cube([esp_d, esp_w, esp_h], center=true);
    translate([up_x_fixed/2 + urt_d/2 + 0.1, 0, 90]) cube([urt_d, urt_w, urt_h], center=true);
    translate([-up_x_fixed/2 - pb_d/2 - 0.1, 0, 155]) cylinder(d=pb_d, h=pb_h, center=true);
    translate([-up_x_fixed/2 - bat_d/2 - 0.1, 0, 65]) cube([bat_d, bat_w, bat_h], center=true);
    translate([0, col_y_target/2 + mpu_d/2 + 0.1, 60]) cube([mpu_w, mpu_d, mpu_h], center=true);
}

module lower_body_mesh() {
    translate([0, 0, b_bot_z])
    difference() {
        union() {
            translate([0, 0, -2]) cube([low_x_fixed, b_inner_y + b_thick*2, 4], center=true);
            hull() {
                translate([0, 0, -4]) cube([low_x_fixed, b_inner_y + b_thick*2, 0.1], center=true);
                translate([0, 0, -24]) cube([low_x_fixed, col_y_target, 0.1], center=true);
            }
            h_len = lower_length - 24;
            translate([0, 0, -24 - h_len/2]) cube([low_x_fixed, col_y_target, h_len], center=true);
        }
        for(x = [-b_hole_x/2, b_hole_x/2]) for(y = [-b_hole_y/2, b_hole_y/2])
            translate([x, y, -10]) cylinder(d=3.2, h=30, center=true);
            
        h_len2 = lower_length - 24 - 4; // 바닥 V홈 파임용 4mm 두께만 남기고 모두 파냄!
translate([0, 0, -24 - h_len2/2]) cube([low_x_fixed - 8, col_y_target - 8, h_len2], center=true);
        
        translate([0, 0, -lower_length - 1.5]) rotate([0, 45, 0]) cube([6, col_y_target + 10, 6], center=true);
        
        translate([0, col_y_target/2, -40]) {
            translate([0, 0, 7.5]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
            translate([0, 0, -7.5]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
        }
    }
}

module bracket_mesh() {
    difference() {
        translate([0, 0, b_bot_z + b_thick/2]) cube([b_w, b_inner_y + 2*b_thick, b_thick], center=true);
        for(x = [-b_hole_x/2, b_hole_x/2]) for(y = [-b_hole_y/2, b_hole_y/2])
            translate([x, y, b_bot_z + b_thick/2]) cylinder(d=3.5, h=10, center=true);
    }
    translate([0, b_inner_y/2 + b_thick/2, -b_arm_l/2]) cube([b_w, b_thick, b_arm_l], center=true);
    translate([0, -b_inner_y/2 - b_thick/2, -b_arm_l/2]) cube([b_w, b_thick, b_arm_l], center=true);
}

module horn_mesh() {
    translate([0, motor_y/2 + 1.5, 0]) rotate([90, 0, 0]) cylinder(d=horn_d, h=horn_h, center=true);
    translate([0, -motor_y/2 - 1.5, 0]) rotate([90, 0, 0]) cylinder(d=horn_d, h=horn_h, center=true);
}

module lower_electronics_mesh() {
    translate([0, col_y_target/2 + mpu_d/2 + 0.1, b_bot_z - 40]) cube([mpu_w, mpu_d, mpu_h], center=true);
}
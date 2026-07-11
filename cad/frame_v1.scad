// =====================================================================
//  슬랙라인 균형 로봇 — 프레임 v1  (2026-07-08)
//  2020 알루미늄 프로파일 직육면체, 상단 60cm(Y방향) 레일 2개 제거
// ---------------------------------------------------------------------
//  좌표 규약 (단위: mm)
//    X = 깊이(앞뒤)      = 400  (40 cm)
//    Y = 피벗 이격 방향  = 600  (60 cm)
//    Z = 높이           = 1200 (120 cm)
//  모든 회전은 Y축. 상단에 남는 두 개의 X방향 레일 중앙이 피벗 자리
//  (두 피벗은 Y로 ~60cm 이격, 회전축은 Y와 평행).
// =====================================================================

// ---------------- 파라미터 ----------------
P  = 20;      // 2020 프로파일 단면 한 변 [mm]
X  = 400;     // 깊이
Y  = 600;     // 피벗 이격 방향
Z  = 1200;    // 높이
$fn = 32;

show_removed_ghost = true;  // 제거한 상단 60cm 2개를 반투명으로 표시
show_pivot_axis    = true;  // 피벗 축(Y) 시각화

// 파생 길이 (기둥 사이에 들어가는 가로재 길이)
xrail_len = X - 2*P;   // X방향 레일 = 360
yrail_len = Y - 2*P;   // Y방향 레일 = 560 (상단 이 2개를 제거)

// ---------------- 프로파일 모듈 (단순화: 20x20 solid) ----------------
module post()  color("silver") cube([P, P, Z]);           // 수직 기둥 (Z)
module xrail() color("silver") cube([xrail_len, P, P]);    // X방향 가로재
module yrail() color("silver") cube([P, yrail_len, P]);    // Y방향 가로재

// =====================================================================
//  조립
// =====================================================================

// ---- 1) 수직 기둥 4개 (모서리, 전체 높이) ----
translate([0,   0,   0]) post();
translate([X-P, 0,   0]) post();
translate([0,   Y-P, 0]) post();
translate([X-P, Y-P, 0]) post();

// ---- 2) X방향 가로재 (기둥 사이) ----
//  하단 2개
translate([P, 0,   0  ]) xrail();
translate([P, Y-P, 0  ]) xrail();
//  상단 2개 (★유지 — 이 두 레일의 중앙이 피벗 자리)
translate([P, 0,   Z-P]) xrail();
translate([P, Y-P, Z-P]) xrail();

// ---- 3) Y방향 가로재 (기둥 사이) ----
//  하단 2개
translate([0,   P, 0  ]) yrail();
translate([X-P, P, 0  ]) yrail();
//  상단 2개 → ★제거 (요청: 상단 60cm 프로파일 2개 제거)
//  translate([0,   P, Z-P]) yrail();
//  translate([X-P, P, Z-P]) yrail();

// ---- (옵션) 제거한 상단 Y레일을 반투명 유령으로 표시 ----
if (show_removed_ghost) {
    color([1,0,0,0.20]) translate([0,   P, Z-P]) yrail();
    color([1,0,0,0.20]) translate([X-P, P, Z-P]) yrail();
}

// ---- (옵션) 피벗 축(Y방향) 시각화 ----
//  상단 두 X레일 중앙(X/2)을 잇는 Y축 = 줄이 매달려 회전하는 축
if (show_pivot_axis) {
    pz = Z - P/2;
    color([1,0.4,0,0.4])
        translate([X/2, P/2, pz]) rotate([-90,0,0]) cylinder(h = Y - P, r = 2);
    // (피벗 위치에는 아래 베어링/엔코더 마운트가 놓임)
}

echo(str("Frame  X=", X/10, "cm  Y=", Y/10, "cm  Z=", Z/10, "cm"));
echo(str("남은 상단 레일 = X방향 2개 (각 ", xrail_len/10, "cm), 피벗 이격 = Y ~", (Y-P)/10, "cm"));
echo("상단 Y방향 60cm 레일 2개 제거됨");

// =====================================================================
//  피벗 마운트 — 앞쪽(y≈P)=베어링, 뒤쪽(y≈Y-P)=엔코더
//  로컬 좌표: 원점=회전축, 축=+Y(안쪽), 플레이트 뒷면=y=0(레일 안쪽 면에 밀착)
//  레일 안쪽 면 슬롯(z=축 중심선)에 M5 T너트로 체결
// =====================================================================
axis_z   = Z - P/2;        // 피벗 축 높이 = 상단 레일 중심 (1190)
plate_t  = 6;              // 마운트 플레이트 두께
plate_w  = 52;             // 플레이트 X폭
plate_h  = 44;             // 플레이트 Z높이
bolt_d   = 5.5;            // M5 통과홀
bolt_dx  = 20;            // 레일 볼트 X간격 (슬롯 중심선 z=0)

// ... (이하 생략하지 않고 전체 다 작성함)
brg_od = 22; brg_id = 8; brg_w = 7;
boss_od  = brg_od + 8;     // 30
boss_len = 16;             // 축방향 보스 길이

enc_body_d     = 24;       // 엔코더 몸체 안착경 (예시)
enc_body_depth = 6;
enc_shaft_d    = 7;        // 샤프트/자석 관통
enc_bolt_bc    = 16;       // 엔코더 볼트원 지름 (예시)

module bearing_mount() {
    difference() {
        union() {
            translate([-plate_w/2, 0, -plate_h/2]) cube([plate_w, plate_t, plate_h]);
            translate([0, plate_t, 0]) rotate([-90,0,0]) cylinder(h=boss_len, d=boss_od);
        }
        translate([0, plate_t+boss_len-brg_w, 0]) rotate([-90,0,0]) cylinder(h=brg_w+0.2, d=brg_od);
        translate([0,-1,0]) rotate([-90,0,0]) cylinder(h=plate_t+boss_len+2, d=brg_id+1);
        for (dx=[-bolt_dx,bolt_dx])
            translate([dx,-0.1,0]) rotate([-90,0,0]) cylinder(h=plate_t+0.2, d=bolt_d);
    }
}

module encoder_mount() {
    difference() {
        union() {
            translate([-plate_w/2, 0, -plate_h/2]) cube([plate_w, plate_t, plate_h]);
            translate([0, plate_t, 0]) rotate([-90,0,0]) cylinder(h=boss_len, d=boss_od);
        }
        translate([0, plate_t+boss_len-enc_body_depth, 0]) rotate([-90,0,0]) cylinder(h=enc_body_depth+0.2, d=enc_body_d);
        translate([0,-1,0]) rotate([-90,0,0]) cylinder(h=plate_t+boss_len+2, d=enc_shaft_d);
        for (a=[0,180])
            rotate([0,a,0]) translate([enc_bolt_bc/2, plate_t+boss_len-enc_body_depth-2, 0])
                rotate([-90,0,0]) cylinder(h=enc_body_depth+4, d=2.6);
        for (dx=[-bolt_dx,bolt_dx])
            translate([dx,-0.1,0]) rotate([-90,0,0]) cylinder(h=plate_t+0.2, d=bolt_d);
    }
}

color("orange") translate([X/2, P,   axis_z]) bearing_mount();
color("green")  translate([X/2, Y-P, axis_z]) mirror([0,1,0]) encoder_mount();

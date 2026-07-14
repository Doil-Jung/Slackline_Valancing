// =====================================================================
// Slackline balancing robot frame and rope system v4
// 2026-07-13, from v3 (2026-07-08, robot added 2026-07-11)
// v4 goal: OpenCR + 3S battery symmetric internal mounting in the 80 mm
//          upper-body box (was: external asymmetric standoffs + open tray)
//
// Design intent
//   - Keep the v2 pivot axis convention:
//       X = swing direction / depth
//       Y = pivot-axis direction / frame width
//       Z = vertical
//   - The slackline trapezoid swings about one Y-parallel top axis.
//   - The pivot mounts sit ON TOP of the top X rails (2026-07-11); the
//     axis is mount_rise above the frame top, which lifts the whole rope
//     and robot for bottom Y-brace and floor clearance.
//   - Phi is measured at the upper frame pivot by a frame-fixed shaft encoder.
//   - The encoder is NOT a structural bearing:
//       each upper pivot mount carries two 608 bearings,
//       a short shaft extension drives the encoder through a printed Oldham
//       solid sleeve coupler.
//   - Lower ankle bearings belong to the robot lower body, not to the rope.
//     The 8 mm lower carbon bar passes through two bearings in the robot body.
//   - Encoder wiring uses service loops:
//       top loop: frame-fixed encoder -> rotating upper carbon tube
//       bottom loop: rotating lower bar -> robot lower body
//
// Robot (2026-07-11)
//   - Lower body L1 = 250 (ankle axis -> hip axis),
//     upper body L2 = 350 (hip axis -> head top). The head rises above the
//     pivot axis on purpose; the alpha stop keeps it off the floor.
//   - Printed plate/box skeleton, lower body 80 / upper body 90 wide
//     with tapered waists at
//     the bracket interfaces. Ankle = printed carrier with two 608s on the
//     lower carbon bar, axially retained by two plain lock collars. No
//     rotation stop: with the raised pivot even a fully inverted fall
//     clears the floor and the Y braces, so alpha is unconstrained.
//   - Hip = XM430-W210-R with ROBOTIS FR12 frame hardware (2026-07-11):
//     an FR12-S101 style C bracket over the motor top ties the motor BODY
//     to the upper body; an FR12-H101 hinge C bracket grips the horn and
//     the HN12-I101 idler and bolts to the lower spine top cap. The mass
//     split stays near the 30:70 shape-optimization target.
//   - v4 (2026-07-13): OpenCR and the 3S lipo live INSIDE the upper-body
//     box (widened to 90; the head flares to 98 so the measured 84 mm
//     pack plus a lead bay for its Y-exiting wires fit between the Y
//     walls), symmetric about the swing plane. The board stands
//     vertically on a central bulkhead (plane normal to X, components
//     toward X-), the battery lies flat on a shelf at the very top (mass
//     stays high). The whole X- face of the B piece is a screwed-on
//     service door: board screws, USB/switches and the battery slide-in
//     are all reached from there. The B/A splice sleeve became two X-side
//     tongues so the 75 mm wide board slides up through the open bottom.
//     Only the AN25 analog wire crosses the ankle and hip joints through
//     small service loops.
//   - Dimensions marked with a star (*) must be checked against the real
//     XM430 drawing / OpenCR board before printing.
// =====================================================================

fast_preview = is_undef(print_export) ? true : !print_export;
$fn = fast_preview ? 24 : 64;

// -------------------- Display controls --------------------
phi_demo          = 15;     // rope swing angle about the Y axis [deg]
alpha_demo       = 180;    // robot lower body absolute lean preview [deg]
theta_demo       = 180;    // robot upper body absolute lean preview [deg]
rigid_rod_demo   = false;  // rope+robot swing as ONE rigid rod (alpha=theta=phi+180)
show_removed_top = false;  // ghost of the old top Y rails (off since 2026-07-11)
show_pivot_axis  = false;  // orange virtual pivot axis line
show_cables      = !fast_preview;
show_robot       = true;   // full balancing robot on the lower bar
show_bearing_cutaway = false; // cut windows in printed bearing housings so the pockets stay visible
show_lower_socket_cutaway = false; // use lower_socket_part.scad for socket cutaway inspection

// -------------------- Frame parameters --------------------
profile = 20;              // 2020 aluminum profile side [mm]
frame_x = 400;             // depth / swing direction [mm]
frame_y = 600;             // width / pivot-axis direction [mm]
frame_z = 1220;            // height [mm]
brace_z = 120;             // stiffening Y braces: bottom face height [mm]

xrail_len = frame_x - 2 * profile;
yrail_len = frame_y - 2 * profile;

axis_x    = frame_x / 2;
// The pivot mounts SIT ON TOP of the top X rails (changed 2026-07-11,
// previously they hung below). This raises the whole rope + robot by
// profile + 2*rise, buying bottom clearance for the Y braces and floor.
mount_rise = 25;           // top pivot axis above the top profile top face [mm]
axis_z    = frame_z + mount_rise;

// Upper bearing centers. These keep the v2 axis position.
top_mount_front_y = profile / 2;
top_mount_rear_y  = frame_y - profile / 2;

plate_t   = 6;
boss_len  = 16;
brg_w     = 7;
brg_od    = 22;
brg_id    = 8;
brg_outer_race_id = 19;
brg_inner_race_od = 12;
shaft_d   = 8;             // nominal shaft = 608 inner ring bore [mm]
shaft_print_d = 7.8;       // PRINTED shaft diameter, front AND rear
                           // sockets (2026-07-13: easier slip into the
                           // 8.0 inner rings; the 7.85 print needed
                           // grinding, so go under, not over)
shaft_clearance_d = 8.4;

top_boss_len = 24;         // upper pivot housing length for two 608 bearings [mm]
top_bearing_offset = top_boss_len/2 - brg_w/2; // bearings press in from both open ends
top_center_bore_len = top_boss_len - 2 * brg_w; // center land bore between the two 608 bearings
top_center_bore_d = brg_outer_race_id; // keeps the printed land touching only the outer races
inner_ring_contact_d = 11.4; // contact faces stay smaller than the 608 inner-ring OD [mm]
inner_ring_contact_len = 1.2;

lock_collar_t = 6;
lock_collar_od = 15;
lock_collar_shaft_clearance = 2.0;
lock_bolt_d = 3.2;         // M3 clearance for the lock collar clamp bolt
lock_slit_w = 1.2;
lock_lug_w = 4.0;          // one top lug width across X
lock_lug_t = 6.0;          // lug thickness along the shaft axis
lock_lug_h = 7.8;
lock_lug_bite = 3.0;       // how far the lugs sink into the collar OD
lock_lug_inset = 1.0;      // moves lugs inward so they blend into the collar
lock_bolt_z = lock_collar_od/2 + lock_bolt_d/2 + 0.2;

bearing_center_to_rail = plate_t + boss_len - brg_w / 2;
front_pivot_y = top_mount_front_y + bearing_center_to_rail;
rear_pivot_y  = top_mount_rear_y  - bearing_center_to_rail;

// -------------------- Rope system parameters --------------------
carbon_od  = 8;
carbon_id  = 6;
diag_len   = 530;          // diagonal carbon tube corner-to-corner length [mm]
lower_socket_outer_d = 18;
lower_socket_insert_len = 27;
lower_bar_exposed_len = 100; // lower bar length visible between the two socket mouths
lower_w    = lower_bar_exposed_len + 2 * lower_socket_insert_len;
lower_pipe_insert_len = 10;
lower_pipe_stop_s = lower_socket_insert_len - lower_pipe_insert_len;
lower_socket_bore_d = 8.2; // measured carbon pipe is about 7.9 mm OD
lower_wire_bore_d = 5;
lower_wire_extra_len = 6;
lower_wire_elbow_s = 7;
lower_wire_elbow_overlap = 2;
lower_wire_elbow_steps = fast_preview ? 6 : 12;
lower_side_wire_from_mouth = 15;
lower_side_wire_s = lower_socket_insert_len - lower_side_wire_from_mouth;
lower_side_wire_bore_d = 5;
lower_side_wire_hole_len = lower_socket_outer_d + 8;

top_socket_inset = 30;     // clearance between bearing center and tube entry [mm]
top_front_y = front_pivot_y + top_socket_inset;
top_rear_y  = rear_pivot_y  - top_socket_inset;
upper_w     = top_rear_y - top_front_y;
top_socket_outer_d = 16;
top_socket_cup_len = 22;
top_pipe_insert_len = 10;
top_pipe_stop_s = top_socket_cup_len - top_pipe_insert_len;
top_socket_bore_d = lower_socket_bore_d;
top_socket_overlap = 5;
top_socket_miter_extend = top_socket_outer_d;
top_socket_miter_overlap = 0.15; // tiny overlap across the miter plane for a solid union
top_encoder_wire_bore_d = 6;
top_encoder_wire_bore_extra = 2;
top_encoder_wire_exit_s = top_pipe_stop_s - 4;
top_encoder_wire_side_hole_d = 5;
top_encoder_wire_side_hole_lift_z = 1;
top_encoder_wire_side_hole_len = top_socket_outer_d + 8;

dy_sag = (upper_w - lower_w) / 2;
sag    = sqrt(pow(diag_len, 2) - pow(dy_sag, 2));

bottom_front_y = frame_y / 2 - lower_w / 2;
bottom_rear_y  = frame_y / 2 + lower_w / 2;

// -------------------- Encoder and coupling parameters --------------------
rear_shaft_out = 48;       // structural shaft beyond rear bearing [mm]
                           // v4 2026-07-13: the printed shaft BROKE, so
                           // the reprint gets the PROPER +24 that the
                           // 7/12 coupler-engagement miss called for
                           // (was 24 + inside-mounted-encoder workaround)
// v4 2026-07-13 (coupler REPRINT): encoder-side bore shortened 12 -> 9
// so the coupler end stays 0.6 mm clear of the bracket plate - the
// bracket keeps its printed 13 mm hole, no drilling. 23 = 12 + 2 + 9.
coupler_len = 23;
coupler_od = 14;
coupler_struct_bore_d = 8.1; // 8 mm structural shaft, close fit after sanding
coupler_encoder_bore_d = 6.1; // AN25 6 mm shaft, close fit after sanding
coupler_struct_bore_len = 12;
coupler_encoder_bore_len = 9;
coupler_clamp_bolt_d = 3.2; // M3 clearance, through-bolt and nut
coupler_slit_w = 1.0;
coupler_relief_slit_w = 1.0;
coupler_center_web_t = coupler_relief_slit_w;
coupler_lug_w = 4.0;
coupler_lug_t = 6.0;
coupler_lug_h = 7.8;
coupler_lug_bite = 3.0;
coupler_lug_inset = 1.0;

// Dream Solution AN25-Analog angle sensor, outline dimensions from datasheet.
enc_shaft_d    = 6;
enc_shaft_len  = 14.6;
enc_body_d     = 25;
enc_body_len   = 17;
enc_face_t     = 2.0;
enc_mount_bcd  = 20;
enc_mount_angles = [45, 135, 225, 315];
enc_bolt_d     = 3.2;      // M3 clearance / printed pilot hole
enc_center_clearance_d = 13.0;
enc_connector_w = 14;
enc_connector_len = 13.2;
enc_connector_h = 6.3;

enc_bracket_t = 5;
enc_driver_access_d = 12;       // screwdriver access through the lower face for M5 rail bolts
enc_rail_bolt_x = 20;
enc_driver_access_edge_meat = 3;
enc_bracket_w = max(
    enc_body_d + 22,
    enc_mount_bcd + 22,
    2 * (enc_rail_bolt_x + enc_driver_access_d/2 + enc_driver_access_edge_meat)
);
enc_bracket_h = 2 * mount_rise; // bottom face sits on the top profile top face

rear_outer_bearing_y = rear_pivot_y + top_bearing_offset + brg_w / 2;
rear_shaft_tip_y     = rear_outer_bearing_y + rear_shaft_out;

// v4 2026-07-13: shaft reprint with rear_shaft_out=48 REVERTS the 7/12
// inside-mount + washer workaround. The AN25 goes back on the plate
// OUTER face, front-tap mounted (4 x M3 from the plate inner side, no
// washers - the side-exit pins sit outside, clear of everything).
// Chain: structural shaft bottoms its 12 mm bore, encoder engages 9 mm
// (the coupler is reprinted with a shortened encoder bore), coupler end
// stays 0.6 mm clear of the plate -> the bracket is UNTOUCHED, as printed.
enc_plate_hole_d = enc_center_clearance_d;   // printed 13, kept
enc_plate_out_y = rear_outer_bearing_y + 24 + 26
                + enc_shaft_len;   // 24 / 26 = the rear_shaft_out /
                                   // coupler_len the bracket was PRINTED
                                   // with; its position stays frozen
enc_plate_in_y  = enc_plate_out_y - enc_bracket_t;
enc_face_y      = enc_plate_out_y;                   // encoder FRONT face
enc_rear_y      = enc_face_y + enc_body_len;         // encoder rear face
enc_tip_y       = enc_face_y - enc_shaft_len;
coupler_end_y   = enc_tip_y + coupler_encoder_bore_len;
coupler_start_y = coupler_end_y - coupler_len;

// -------------------- Robot parameters --------------------
// Robot local frame: origin at the ankle (lower bar) axis, +Z up to the head.
// Mass targets (shape optimization 2026-06-15): lower:upper ~ 30:70,
// equipment high on the upper body, ankle kept light.
robot_l1 = 250;            // ankle axis -> hip axis [mm]
robot_l2 = 350;            // hip axis -> head top, battery included [mm]
hip_z    = robot_l1;

// XM430-W210-R outline. The motor BODY is attached to the upper body
// (mass split) through an FR12-S101 style C bracket over the motor top;
// the horn and the HN12-I101 idler ("fake horn") are gripped by an
// FR12-H101 hinge C bracket that bolts to the lower spine top cap.
// (*) all bracket dimensions: verify on the ROBOTIS drawings / real parts.
xm_w = 28.5;               // case width  (X) [mm]
xm_len = 46.5;             // case length (Z) [mm]
xm_dep = 34.0;             // case depth  (Y) [mm]
xm_axis_from_end = 14.25;  // horn axis to the near case end [mm] (*)
xm_horn_od = 21.5;         // exposed horn / idler diameter (*)
xm_horn_lift = 3.5;        // horn face above the case surface (*)

// FR12-H101 hinge frame (lower-body side, purchased part, modeled approx).
h101_t = 2.0;              // sheet thickness (*)
h101_axis_to_web = 24;     // horn axis to the web INNER face (*)
h101_arm_top = 14;         // arm extent above the horn axis (*)
h101_arm_w = 28;           // arm width along X (*)
h101_web_x = 28;           // web width along X (*)
h101_web_holes_dx = 16;    // web mounting hole grid, M2.5 (*)
h101_web_holes_dy = 8;     // (*)
h101_bolt_d = 2.8;         // M2.5 clearance

// FR12-S101 style body bracket (upper-body side, purchased part, approx).
s101_t = 2.0;              // sheet thickness (*)
s101_arm_h = 18;           // side arms down the motor X faces (*)
s101_top_holes_dx = 16;    // top plate hole grid, M2.5 (*)
s101_top_holes_dy = 8;     // (*)

// Printed ankle bearing carrier. The two 608s press in from both end faces,
// the lower carbon bar runs through their inner rings (thin tape/retainer
// glue on the 7.9 mm carbon bar takes up the last of the slop).
// Bar assembly order: front socket | lock collar | carrier(+bearings) |
// lock collar | rear socket - slide everything on before the second glue-up.
// No rotation stop: with the raised pivot the robot clears the floor even
// fully inverted, so alpha is free and two plain lock collars suffice.
// Wide bearing spacing: roll (X-axis) stiffness of the robot on the bar
// comes from this spacing, matched to the 80 mm wide foot section.
// Bar budget: carrier 80 + two lock collars 14.4 = 94.4 of the 100 mm
// exposed bar, ~2.8 mm slack to each socket mouth (tight but fine).
rb_spacing = 73;                          // 608 pair spacing on the bar:
                                          // bearing OUTER faces span
                                          // 73 + 7 = 80 = the body width
carrier_len = rb_spacing + brg_w;         // pockets open to the end faces
carrier_boss_d = 40;
carrier_flange_z0 = carrier_boss_d / 2;
carrier_flange_t = 6;
carrier_flange_x = 52;
carrier_flange_y = 80;     // flange flush with the 80 mm foot (= boss
                           // length, since rb_spacing widened to 73)
carrier_bolt_dx = 20;
carrier_bolt_dy = 25;
carrier_bolt_d = 3.4;                     // M3 clearance to the spine flange

// Lower spine: printed box column, carrier flange -> H101 web cap.
// The body is 80 mm wide from the FOOT all the way up (roll-stability
// request 2026-07-11); only the waist tapers down to the narrow
// FR12-H101 web cap.
lsp_x = 34;
lsp_y_top = 41;            // narrow width at the H101 web cap
lsp_y_wide = 80;           // robot body width, foot included
lsp_wall = 3;
lsp_z0 = carrier_flange_z0 + carrier_flange_t;
lsp_cap_t = 4;
lsp_z1 = hip_z - h101_axis_to_web - h101_t;  // cap top face = web outer face
lsp_flare2 = 170;          // waist taper 80 -> 41 over flare2..flare3
lsp_flare3 = 210;
lsp_win_w = 46;  lsp_win_h = 32;  lsp_win_pitch = 46;

// Upper body: box spine, printed as two pieces A/B. The A piece starts
// ABOVE the motor: its bottom cap bolts onto the FR12-S101 top plate
// (kept at bracket width), then the body tapers out to 80 mm wide.
usp_x = 38;  usp_wall_x = 4.0;
usp_y = 40;  usp_wall_y = 2.5;     // hip-side width, matches the S101 plate
usp_y_wide = 90;                   // upper body width (v4: 90 so the 84
                                   // battery fits fully inside; the FOOT
                                   // stays 80 - bar roll stance)
usp_y_head = 98;                   // battery-bay head, flared wider for
                                   // the LEAD BAY (see battery notes)
head_flare_z0 = 542;               // head flare 90 -> 98
head_flare_z1 = 552;
usp_cap_t = 4;
usp_z0 = hip_z + xm_len - xm_axis_from_end + s101_t;  // S101 top plate face
usp_flare0 = usp_z0 + 16;          // taper start
usp_flare1 = usp_flare0 + 50;      // taper end, 80 mm wide from here up
usp_split_z = 445;                 // A/B splice height (bed-size driven;
                                   // v4: lowered 460 -> 445 so the 105 mm
                                   // OpenCR fits above the splice tongues.
                                   // B piece is 190 tall incl. tongues -
                                   // fits the Bambu X1C (256 Z), upright)
usp_sleeve_len = 35;
usp_z1 = hip_z + robot_l2;         // head top, local z
usp_win_w = 50;  usp_win_h = 38;  usp_win_pitch = 55;
splice_bolt_d = 3.4;

// -------- v4 internal equipment bay (upper body B piece) --------
// OpenCR 1.0 (105 x 75, ~60 g) stands VERTICALLY INSIDE the body on a
// central bulkhead: board plane normal to X (the swing plane), so its mass
// sits on the swing centerline. Components face X- toward the service
// door; the 4 board screws drive in from the door side.
// (*) hole spacing + tallest component: measure the real board.
ocr_l = 105;  ocr_w = 75;  ocr_t = 1.6;
ocr_hole_dz = 96;          // (*)
ocr_hole_dy = 66;          // (*)
ocr_comp_h = 13;           // tallest component on the -X face (*)
ocr_door_gap = 1.5;        // component tips to the door inner face
bulk_t = 3;                // central bulkhead plate thickness
// Board BACK face x, derived so the component tips just clear the wall:
ocr_back_x = -usp_x/2 + usp_wall_x + ocr_door_gap + ocr_comp_h + ocr_t;

// Snap-rivet board mount (ROBOTIS PCB-support style, MEASURED (*)):
// barrel dia 3.8, snapped tip 4.3, barrel (grip) length 2.7. The rivets
// push in from the door side through board (1.6) + web (1.0) = 2.6 grip,
// slightly snug. Each mount = a 1.0 web disc on a hollow stand-off tube
// on the bulkhead: the tube void takes the snapped-open legs, and a
// 3.5 mm push-out hole through the bulkhead PLUS a 4 mm access hole
// through the X+ wall lets a pin poke them free from outside.
riv_hole_d = 4.0;          // web hole for the 3.8 barrel
riv_web_t = 1.0;
riv_leg_room = 3.5;        // void depth behind the web for the legs
riv_tube_od = 9;
riv_tube_id = 6;
bulk_x0 = ocr_back_x + riv_web_t + riv_leg_room;  // bulkhead front face
                           // (rear channel bulk_x0+bulk_t .. +15 free)
// (90 body) interior is 85 wide vs the 75 board: no wall grooves
// needed, the four rivet mounts alone locate the board.

// 3S lipo INSIDE the flared head on a shelf (mass on the very top).
// MEASURED pack 84 x 34 x 25, fully internal. BOTH leads (charge + T-plug
// output) leave the Y+ narrow end face pointing along Y, so the pack
// butts on the Y- wall (positive stop) and the head is flared to 98 to
// leave an 8.5 mm LEAD BAY in front of the Y+ end: the leads bend there.
// The output lead turns down inside the bay to the board; the short
// ~20 mm charge lead hangs out through the charge window in the Y+ head
// wall - charging in place, nothing to open. Y+ retention: foam pad in
// the lead bay + the strap (travel is capped at the bay depth anyway (*)).
// The pack slides in along X through the door's battery-bay opening.
bat_x = 25;  bat_y = 84;  bat_z = 34;    // 25 across X, 84 along Y (*)
bat_y0 = -(usp_y_head/2 - usp_wall_y) + 0.5;  // butted on the Y- head wall
bat_y_mid = bat_y0 + bat_y/2;                 // pack center y (= -4)
chg_win_x = 20;            // (*) charge window; recenter on the real lead
chg_win_z = 16;
bat_top_gap = 2;
shelf_t = 4;
bat_z1 = usp_z1 - usp_cap_t - bat_top_gap;
bat_z0 = bat_z1 - bat_z;
shelf_z0 = bat_z0 - shelf_t;
strap_slot_w = 3;
strap_slot_l = 16;

// Board as high as possible under the shelf.
ocr_top_gap = 3;
ocr_center_z = shelf_z0 - ocr_top_gap - ocr_l/2;

// X- service door: one T-shaped plate over the whole bay. Top edge
// tucks into a rebate under the top cap (no top screws - the cap edge is
// too thin to thread); then 3 x 2 M3 self-taps drive into wall bosses
// (~9 mm engagement). No nuts anywhere on the door. Board screws / USB /
// switches / wiring are reached through the opening.
door_t = 3;
door_z0 = usp_split_z + 5;
door_z1 = bat_z1;
door_open_y = 66;          // opening width between the Y-wall lips
door_lip_z = 4;            // plate overlap beyond the opening, top/bottom
door_w = 88;               // plate width over the board zone
door_w_top = 97;           // plate width over the flared battery bay
door_screw_y = 36;         // screw line inside the 7 mm side lands
door_bolt_d = 2.7;         // M3 self-tap pilot in the wall

// -------------------- Colors --------------------
c_profile = [0.74, 0.76, 0.78];
c_ghost   = [1.00, 0.20, 0.15, 0.18];
c_mount   = [0.95, 0.47, 0.12];
c_socket  = [0.76, 0.05, 0.08];
c_carbon  = [0.08, 0.08, 0.08];
c_bearing_outer = [0.18, 0.18, 0.18];
c_bearing_inner = [0.75, 0.75, 0.72];
c_encoder = [0.05, 0.18, 0.35];
c_coupler = [0.98, 0.72, 0.15];
c_robot   = [0.10, 0.35, 0.85, 0.45];
c_rlow    = [0.25, 0.50, 0.90];
c_rup     = [0.30, 0.72, 0.45];
c_cable   = [0.02, 0.02, 0.02];
c_cable2  = [0.05, 0.18, 0.95];

// -------------------- Vector helpers --------------------
function vlen(v) = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
function unit(v) = v / vlen(v);
function add3(a,b) = [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
function rot_y(p,a) = [
    p[0] * cos(a) + p[2] * sin(a),
    p[1],
   -p[0] * sin(a) + p[2] * cos(a)
];

// -------------------- Primitive helpers --------------------
module cyl_y(h, d, center=true) {
    rotate([-90, 0, 0]) cylinder(h=h, d=d, center=center);
}

module cyl_x(h, d, center=true) {
    rotate([0, 90, 0]) cylinder(h=h, d=d, center=center);
}

module rod_between(p1, p2, d=4) {
    hull() {
        translate(p1) sphere(d=d);
        translate(p2) sphere(d=d);
    }
}

module capsule_between(p1, p2, d=4) {
    hull() {
        translate(p1) sphere(d=d);
        translate(p2) sphere(d=d);
    }
}

module cylinder_between(p1, p2, d=4) {
    v = p2 - p1;
    l = vlen(v);
    translate(p1)
        rotate(a=acos(v[2] / l), v=[-v[1], v[0], 0])
            cylinder(h=l, d=d, center=false);
}

module yz_halfspace(p, n, positive=true, size=1000) {
    nn = unit(n);
    a = atan2(nn[2], nn[1]);
    translate(p)
        rotate([a, 0, 0])
            translate([0, (positive ? 1 : -1) * size/2, 0])
                cube([size, size, size], center=true);
}

module mitered_cylinder_between(p1, p2, d, plane_p, plane_n, positive=true, overlap=0) {
    n = unit(plane_n);
    cut_p = positive ? plane_p - n * overlap : plane_p + n * overlap;
    intersection() {
        cylinder_between(p1, p2, d=d);
        yz_halfspace(cut_p, n, positive=positive);
    }
}

function bezier2(p0, p1, p2, t) =
    p0 * pow(1 - t, 2) + p1 * (2 * (1 - t) * t) + p2 * pow(t, 2);

module curved_wire_elbow(p_corner, u1, u2, radius_s, d=5, steps=12) {
    p0 = p_corner + u1 * radius_s;
    pc = p_corner;
    p2 = p_corner + u2 * radius_s;

    for (i = [0 : steps - 1]) {
        hull() {
            translate(bezier2(p0, pc, p2, i / steps))
                sphere(d=d);
            translate(bezier2(p0, pc, p2, (i + 1) / steps))
                sphere(d=d);
        }
    }
}

module tube_path(points, d=2, col=c_cable) {
    color(col)
    for (i = [0 : len(points) - 2])
        rod_between(points[i], points[i+1], d);
}

module arc_cable_y(center, radius, yoff, a0, a1, d=2, col=c_cable, n=24) {
    pts = [
        for (i = [0:n])
        [
            center[0] + radius * cos(a0 + (a1 - a0) * i / n),
            center[1] + yoff,
            center[2] + radius * sin(a0 + (a1 - a0) * i / n)
        ]
    ];
    tube_path(pts, d=d, col=col);
}

// -------------------- Frame profiles --------------------
module post() {
    color(c_profile) cube([profile, profile, frame_z]);
}

module xrail() {
    color(c_profile) cube([xrail_len, profile, profile]);
}

module yrail() {
    color(c_profile) cube([profile, yrail_len, profile]);
}

module frame_assembly() {
    // Vertical posts.
    translate([0, 0, 0]) post();
    translate([frame_x-profile, 0, 0]) post();
    translate([0, frame_y-profile, 0]) post();
    translate([frame_x-profile, frame_y-profile, 0]) post();

    // X direction rails. Top front/rear rails remain.
    translate([profile, 0, 0]) xrail();
    translate([profile, frame_y-profile, 0]) xrail();
    translate([profile, 0, frame_z-profile]) xrail();
    translate([profile, frame_y-profile, frame_z-profile]) xrail();

    // Y direction rails. Only bottom rails remain; upper Y rails are removed.
    translate([0, profile, 0]) yrail();
    translate([frame_x-profile, profile, 0]) yrail();

    // Stiffening Y braces above the floor rectangle (added 2026-07-11).
    // They span the full frame width, so unlike the corner posts they DO
    // cross the robot's Y range - the inverted-hang swing envelope is
    // checked in the console echo below.
    translate([0, profile, brace_z]) yrail();
    translate([frame_x-profile, profile, brace_z]) yrail();

    if (show_removed_top) {
        color(c_ghost) translate([0, profile, frame_z-profile]) yrail();
        color(c_ghost) translate([frame_x-profile, profile, frame_z-profile]) yrail();
    }
}

// -------------------- Bearings and fixed upper mounts --------------------
module bearing_608_y() {
    // 608 bearing visual model: outer and inner races only.
    color(c_bearing_outer)
    difference() {
        cyl_y(brg_w, brg_od);
        cyl_y(brg_w + 0.2, brg_outer_race_id);
    }

    color(c_bearing_inner)
    difference() {
        cyl_y(brg_w + 0.1, brg_inner_race_od);
        cyl_y(brg_w + 0.3, brg_id);
    }
}

module upper_bearing_pair_y() {
    // Two independent 608 bearings in one printed upper pivot housing.
    // They are inserted from the two open ends; the center land bore is at the
    // outer-race inner diameter so the printed land preserves bearing spacing.
    for (yoff = [-top_bearing_offset, top_bearing_offset])
        translate([0, yoff, 0])
            bearing_608_y();
}

module top_pivot_bearing_mount(cutaway=show_bearing_cutaway) {
    plate_w = 58;
    plate_h = 52;
    flange_y = -bearing_center_to_rail - profile/2;
    web_y    = -bearing_center_to_rail + profile/2 - plate_t/2;

    color(c_mount)
    difference() {
        union() {
            // Flange bolted onto the TOP FACE of the remaining top X rail.
            translate([-plate_w/2, flange_y, -mount_rise])
                cube([plate_w, profile, 6]);

            // Vertical web rising from the rail top face to the bearing boss.
            translate([-plate_w/2, web_y, -mount_rise])
                cube([plate_w, plate_t, plate_h/2 + mount_rise]);

            // Bearing housing. One printed mount carries two separate 608 bearings.
            cyl_y(top_boss_len, brg_od + 10);

            // Small ribs make the printed part less floppy.
            for (sx = [-1, 1]) {
                hull() {
                    translate([sx * (plate_w/2 - 7), web_y + plate_t/2, -mount_rise + 5])
                        sphere(d=6);
                    translate([sx * 14, 0, 0])
                        sphere(d=10);
                }
            }
        }

        // Two 608 bearing pockets, open from the two ends for real assembly.
        for (yoff = [-top_bearing_offset, top_bearing_offset])
            translate([0, yoff, 0])
                cyl_y(brg_w + 0.35, brg_od + 0.35);

        // The opening between the two bearings matches the outer race ID.
        // The remaining annular land contacts the outer races and preserves
        // the spacing between the two pressed-in bearings.
        cyl_y(top_center_bore_len + 0.35, top_center_bore_d);

        // Continuous clearance for the rotating 8 mm shaft.
        cyl_y(top_boss_len + 2, shaft_clearance_d);

        // Preview/render cutaway. This avoids relying on transparent draw order
        // and keeps both bearings visible as separate parts.
        if (cutaway)
            translate([-(brg_od + 18)/2, -top_boss_len/2 - 1, 1])
                cube([brg_od + 18, top_boss_len + 2, brg_od + 12]);

        // M5 rail bolts down through the base flange into the rail top slot.
        for (xoff = [-20, 20])
            translate([xoff, -bearing_center_to_rail, -mount_rise - 3])
                cylinder(h=12, d=5.5);
    }
}

module rear_pivot_encoder_mount() {
    // Rear side is modeled as one clean printed part:
    // a side-open rectangular frame with the 608 bearing housing on one end
    // and the encoder mounting face on the other end.
    y_bearing_face = rear_pivot_y + top_boss_len/2;
    y0 = y_bearing_face - enc_bracket_t;
    y1 = enc_plate_out_y;   // rear plate outer face, as originally printed
    len_y = y1 - y0;
    w = enc_bracket_w;
    h = enc_bracket_h;
    t = enc_bracket_t;

    color(c_mount)
    difference() {
        union() {
            // Clean side-view square frame. The sides stay open for assembly;
            // no old bracket geometry is reused here.
            translate([axis_x - w/2, y0, axis_z + h/2 - t])
                cube([w, len_y, t]);
            translate([axis_x - w/2, y0, axis_z - h/2])
                cube([w, len_y, t]);

            // Bearing-side vertical plate, merged into the bearing housing.
            translate([axis_x - w/2, y0, axis_z - h/2])
                cube([w, t, h]);

            // Encoder face: the rear surface is the encoder reference plane.
            translate([axis_x - w/2, y1 - t, axis_z - h/2])
                cube([w, t, h]);

            // Integrated 608 bearing housing on the rope side.
            translate([axis_x, rear_pivot_y, axis_z])
                cyl_y(top_boss_len, brg_od + 10);
        }

        // Two 608 pockets press in from the open ends; the center bore leaves
        // an annular land that locates only the outer races.
        for (yoff = [-top_bearing_offset, top_bearing_offset])
            translate([axis_x, rear_pivot_y + yoff, axis_z])
                cyl_y(brg_w + 0.35, brg_od + 0.35);

        translate([axis_x, rear_pivot_y, axis_z])
            cyl_y(top_center_bore_len + 0.35, top_center_bore_d);

        translate([axis_x, rear_pivot_y, axis_z])
            cyl_y(top_boss_len + 2, shaft_clearance_d);

        // AN25 shaft clearance and the 4-M3 front mounting pattern.
        translate([axis_x, y1 - t/2, axis_z])
            cyl_y(t + 0.4, enc_plate_hole_d);

        for (a = enc_mount_angles) {
            translate([
                axis_x + enc_mount_bcd/2 * cos(a),
                y1 - t/2,
                axis_z + enc_mount_bcd/2 * sin(a)
            ])
            cyl_y(t + 0.4, enc_bolt_d);
        }

        // M5 rail bolts down through the BOTTOM bar into the rear rail top
        // slot (the bracket now sits on top of the profile).
        for (xoff = [-enc_rail_bolt_x, enc_rail_bolt_x])
            translate([axis_x + xoff, frame_y - profile/2, axis_z - h/2 - 0.2])
                cylinder(h=t + 0.4, d=5.5);

        // Matching access holes through the TOP bar so a driver can reach
        // the M5 rail bolts from above during assembly.
        for (xoff = [-enc_rail_bolt_x, enc_rail_bolt_x])
            translate([axis_x + xoff, frame_y - profile/2, axis_z + h/2 - t - 0.2])
                cylinder(h=t + 0.4, d=enc_driver_access_d);

    }
}

module upper_fixed_hardware() {
    // Front structural bearing bracket.
    translate([axis_x, front_pivot_y, axis_z])
        top_pivot_bearing_mount();

    // Rear bearing bracket and encoder support are one printed piece.
    rear_pivot_encoder_mount();

    // Bearings placed into the pockets: two 608 bearings per upper mount.
    translate([axis_x, front_pivot_y, axis_z]) upper_bearing_pair_y();
    translate([axis_x, rear_pivot_y, axis_z])  upper_bearing_pair_y();

    // Fixed AN25 analog encoder on the plate OUTER face, front-tap
    // mounted with 4 x M3 driven from the plate inner side (v4: the
    // longer shaft reverts the 7/12 inside-mount + washer workaround).
    translate([axis_x, enc_face_y, axis_z])
        encoder_AN25_analog();

    // Solid coupler between the structural shaft extension and encoder shaft.
    solid_coupler_y(coupler_start_y, coupler_end_y);
}

// -------------------- AN25 analog encoder and printed coupling --------------------
module encoder_AN25_analog() {
    // Local origin = encoder front face center. Shaft points toward -Y.
    color(c_bearing_inner)
        translate([0, -enc_shaft_len/2, 0])
        cyl_y(enc_shaft_len, enc_shaft_d);

    color([0.82, 0.82, 0.78])
    difference() {
        cyl_y(enc_face_t, enc_body_d);
        cyl_y(enc_face_t + 0.2, enc_center_clearance_d);
        for (a = enc_mount_angles) {
            translate([enc_mount_bcd/2 * cos(a), 0, enc_mount_bcd/2 * sin(a)])
                cyl_y(enc_face_t + 0.3, enc_bolt_d);
        }
    }

    color(c_encoder)
        translate([0, enc_body_len/2, 0])
        cyl_y(enc_body_len, enc_body_d);

    // Pins exit along the body SIDE (real AN25, unlike the first model);
    // their solder joints are what the 2 mm washers make room for.
    color("white")
        translate([0, enc_body_len - 5, enc_body_d/2 + 2.5])
        cube([10, 9, 5], center=true);
}

module coupler_clamp_lugs_y(yoff) {
    lug_x = coupler_slit_w/2 + coupler_lug_w/2 - coupler_lug_inset;
    lug_z = coupler_od/2 - coupler_lug_bite + coupler_lug_h/2;

    for (sx = [-1, 1])
        translate([sx * lug_x, yoff, lug_z])
            cube([coupler_lug_w, coupler_lug_t, coupler_lug_h], center=true);
}

module coupler_clamp_cuts_y(bolt_y, slit_y, slit_len, bore_d) {
    lug_z = coupler_od/2 - coupler_lug_bite + coupler_lug_h/2;
    bolt_z = coupler_od/2 + coupler_clamp_bolt_d/2 + 0.2;
    slit_bottom_z = 0;
    slit_top_z = lug_z + coupler_lug_h/2 + 0.3;
    slit_z = (slit_bottom_z + slit_top_z) / 2;
    slit_h = slit_top_z - slit_bottom_z;

    translate([0, bolt_y, bolt_z])
        cyl_x(2 * coupler_lug_w + coupler_slit_w + 2, coupler_clamp_bolt_d);

    translate([0, slit_y, slit_z])
        cube([coupler_slit_w, slit_len, slit_h], center=true);
}

module coupler_relief_slit_y(yoff, bore_d) {
    slit_bottom_z = 0;
    slit_top_z = coupler_od/2 + 0.3;
    slit_z = (slit_bottom_z + slit_top_z) / 2;
    slit_h = slit_top_z - slit_bottom_z;

    translate([0, yoff, slit_z])
        cube([coupler_od + 2, coupler_relief_slit_w, slit_h], center=true);
}

module coupler_center_top_clearance_y(center_len) {
    // The lower half of the center web is enough as an insertion stop.
    // Remove the upper half so no semicircular boss remains between clamps.
    translate([-(coupler_od + 2)/2, -(center_len + 0.2)/2, 0])
        cube([coupler_od + 2, center_len + 0.2, coupler_od + coupler_lug_h + 2]);
}

module solid_coupler_local(total_len=coupler_len, cutaway=false) {
    struct_len = min(coupler_struct_bore_len, total_len/2 - 0.4);
    encoder_len = min(coupler_encoder_bore_len, total_len/2 - 0.4);
    center_stop_len = total_len - struct_len - encoder_len;
    struct_slit_len = struct_len + center_stop_len - coupler_center_web_t;
    encoder_slit_len = encoder_len + center_stop_len - coupler_center_web_t;
    struct_slit_y = -total_len/2 + struct_len/2;
    encoder_slit_y = total_len/2 - encoder_len/2;
    struct_relief_y = -center_stop_len/2 - coupler_relief_slit_w/2;
    encoder_relief_y = center_stop_len/2 + coupler_relief_slit_w/2;
    struct_clamp_y = (-total_len/2 + struct_relief_y - coupler_relief_slit_w/2) / 2;
    encoder_clamp_y = (total_len/2 + encoder_relief_y + coupler_relief_slit_w/2) / 2;

    color(c_coupler)
    difference() {
        union() {
            cyl_y(total_len, coupler_od);
            coupler_clamp_lugs_y(struct_clamp_y);
            coupler_clamp_lugs_y(encoder_clamp_y);
        }

        // Blind bore for the 8 mm structural shaft side.
        translate([0, -total_len/2 + struct_len/2 - 0.05, 0])
            cyl_y(struct_len + 0.2, coupler_struct_bore_d);

        // Blind bore for the 6 mm encoder shaft side. The uncut center web
        // works as an insertion stop and keeps the coupler torsionally rigid.
        translate([0, total_len/2 - encoder_len/2 + 0.05, 0])
            cyl_y(encoder_len + 0.2, coupler_encoder_bore_d);

        // Through-bolt clamp cuts. No printed thread is required.
        coupler_clamp_cuts_y(struct_clamp_y, struct_slit_y, struct_slit_len, coupler_struct_bore_d);
        coupler_clamp_cuts_y(encoder_clamp_y, encoder_slit_y, encoder_slit_len, coupler_encoder_bore_d);

        // X-direction relief slits just outside the center stopper shoulder.
        // These decouple each clamp end from the rigid center stop so tightening
        // the through-bolts can actually close the bore.
        coupler_relief_slit_y(struct_relief_y, coupler_struct_bore_d);
        coupler_relief_slit_y(encoder_relief_y, coupler_encoder_bore_d);

        // Leave only the lower half of the center stopper.
        coupler_center_top_clearance_y(center_stop_len);

        if (cutaway)
            translate([-coupler_od/2 - 0.5, -total_len/2 - 0.5, 0])
                cube([coupler_od + 1, total_len + 1, coupler_od/2 + 1]);
    }
}

module solid_coupler_y(y0, y1) {
    len = y1 - y0;
    yc  = (y0 + y1) / 2;

    translate([axis_x, yc, axis_z])
        solid_coupler_local(len);
}

module solid_coupler_part(cutaway=false) {
    solid_coupler_local(coupler_len, cutaway=cutaway);
}

// -------------------- Rotating carbon trapezoid and sockets --------------------
module carbon_tube_between(p1, p2, od=carbon_od) {
    color(c_carbon)
    rod_between(p1, p2, od);
}

module lock_collar_at_y(face_y, dir=1) {
    // Shaft lock collar. Only the small yellow nose touches the 608 inner ring;
    // the larger clamp body sits outside the bearing face.
    body_y = face_y + dir * (inner_ring_contact_len + lock_collar_t/2);
    total_t = inner_ring_contact_len + lock_collar_t;
    total_y = face_y + dir * total_t/2;
    lug_x = lock_slit_w/2 + lock_lug_w/2 - lock_lug_inset;
    lug_z = lock_collar_od/2 - lock_lug_bite + lock_lug_h/2;
    slit_bottom_z = (shaft_d + 0.25)/2 - 0.6;
    slit_top_z = lug_z + lock_lug_h/2 + 0.3;
    slit_z = (slit_bottom_z + slit_top_z) / 2;
    slit_h = slit_top_z - slit_bottom_z;
    color([1.0, 0.74, 0.10])
    difference() {
        union() {
            translate([0, face_y + dir * inner_ring_contact_len/2, 0])
                cyl_y(inner_ring_contact_len, inner_ring_contact_d);

            translate([0, body_y, 0])
                cyl_y(lock_collar_t, lock_collar_od);

            // Omega-style clamp lugs placed beside the top slit.
            // The bolt head and nut sit on the lug outer faces, not on a
            // shaved section of the round collar body.
            for (sx = [-1, 1])
                translate([sx * lug_x, body_y, lug_z])
                    cube([lock_lug_w, lock_lug_t, lock_lug_h], center=true);
        }

        translate([0, face_y + dir * (inner_ring_contact_len + lock_collar_t)/2, 0])
            cyl_y(inner_ring_contact_len + lock_collar_t + 0.4, shaft_d + 0.25);

        // Clamp bolt hole. It sits above the shaft bore and pulls the split
        // collar closed instead of passing through the shaft.
        translate([0, body_y, lock_bolt_z])
            cyl_x(2 * lock_lug_w + lock_slit_w + 2, lock_bolt_d);

        // Split cut from the OD into the shaft bore so tightening the bolt
        // actually clamps the collar onto the printed hub shaft.
        translate([0, total_y, slit_z])
            cube([lock_slit_w, total_t + 0.8, slit_h], center=true);
    }
}

module socket_inner_hub_face_at_y(face_y, inside_dir=1) {
    // This is part of the printed rope socket, not a loose spacer.
    // Its diameter is small enough to touch only the bearing inner ring.
    color(c_socket)
    translate([0, face_y + inside_dir * inner_ring_contact_len/2, 0])
        cyl_y(inner_ring_contact_len, inner_ring_contact_d);
}

module upper_socket(is_front=true, include_lock_collar=true) {
    pivot_y = is_front ? front_pivot_y : rear_pivot_y;
    tube_y  = is_front ? top_front_y : top_rear_y;
    dir     = is_front ? -1 : 1;  // direction to the outside of the frame
    inside_dir = -dir;
    outer_y = pivot_y + dir * (top_boss_len/2 + 5);
    inner_y = pivot_y - dir * (top_boss_len/2 + 6);
    outside_face_y = pivot_y + dir * top_boss_len/2;
    front_shaft_tip_y = outside_face_y + dir * (inner_ring_contact_len + lock_collar_t + lock_collar_shaft_clearance);
    shaft_tip_y = is_front ? front_shaft_tip_y : rear_shaft_tip_y;
    inside_face_y = pivot_y + inside_dir * top_boss_len/2;
    socket_start_y = inner_y - inside_dir * top_socket_overlap;

    color(c_socket)
    difference() {
        union() {
            // Shaft passing through both bearings in the mount, printed
            // slightly under 8 mm for a sand-to-fit slip in the inner rings.
            // Keep this as a flat-ended cylinder, not a sphere-ended hull.
            cylinder_between([0, inner_y, 0], [0, shaft_tip_y, 0], d=shaft_print_d);

            // Integrated socket hub face: the inside-side stop for the bearing
            // inner ring, assembled as one piece with the rope socket.
            socket_inner_hub_face_at_y(inside_face_y, inside_dir);

            y_bottom = is_front ? bottom_front_y : bottom_rear_y;
            p_top = [0, tube_y, 0];
            p_bot = [0, y_bottom, -sag];
            u = unit(p_bot - p_top);
            bridge_dir = [0, -inside_dir, 0];
            miter_n = bridge_dir - u;

            // Printed bridge from bearing axis to carbon tube entry.
            mitered_cylinder_between(
                [0, socket_start_y, 0],
                p_top - bridge_dir * top_socket_miter_extend,
                d=top_socket_outer_d,
                plane_p=p_top,
                plane_n=miter_n,
                positive=true,
                overlap=top_socket_miter_overlap
            );

            // Cylindrical carbon tube cup, directed down along the diagonal tube.
            mitered_cylinder_between(
                p_top - u * top_socket_miter_extend,
                p_top + u * top_socket_cup_len,
                d=top_socket_outer_d,
                plane_p=p_top,
                plane_n=miter_n,
                positive=false,
                overlap=top_socket_miter_overlap
            );
        }

        // Blind upper carbon-pipe socket. The pipe enters only
        // top_pipe_insert_len from the mouth and stops on a flat shoulder.
        y_bottom = is_front ? bottom_front_y : bottom_rear_y;
        p_top = [0, tube_y, 0];
        p_bot = [0, y_bottom, -sag];
        u = unit(p_bot - p_top);
        cylinder_between(
            p_top + u * top_pipe_stop_s,
            p_top + u * (top_socket_cup_len + 4),
            d=top_socket_bore_d
        );

        // Encoder-side wire passage through the diagonal tube stop. The
        // carbon tube bore already opens from the mouth to the stop; this
        // axial bore carries the wire past the shoulder and the side hole
        // gives it a short external exit.
        if (!is_front) {
            cylinder_between(
                p_top - u * top_encoder_wire_bore_extra,
                p_top + u * (top_pipe_stop_s + top_encoder_wire_bore_extra),
                d=top_encoder_wire_bore_d
            );

            translate(p_top + u * top_encoder_wire_exit_s + [0, 0, top_encoder_wire_side_hole_lift_z])
                rotate([0, 90, 0])
                    cylinder(
                        h=top_encoder_wire_side_hole_len,
                        d=top_encoder_wire_side_hole_d,
                        center=true
                    );
        }
    }

    // Retainer collar outside the bearing. This slides onto the shaft after
    // insertion and its small nose touches only the bearing inner ring.
    if (include_lock_collar)
        lock_collar_at_y(outside_face_y, dir);
}

module upper_socket_part(side="front") {
    // Standalone upper socket, using the same geometry as the full assembly.
    if (side == "rear")
        translate([0, -rear_pivot_y, 0])
            upper_socket(false, include_lock_collar=false);
    else
        translate([0, -front_pivot_y, 0])
            upper_socket(true, include_lock_collar=false);
}

module lock_collar_part(dir=1) {
    // Standalone lock collar, with the inner-ring contact face at Y=0.
    lock_collar_at_y(0, dir);
}

module encoder_bracket_part() {
    // Standalone rear encoder/bearing bracket. Origin is the rear bearing axis.
    translate([-axis_x, -rear_pivot_y, -axis_z])
        rear_pivot_encoder_mount();
}

module bearing_mount_part(cutaway=false) {
    // Standalone front bearing mount. Origin is the bearing axis.
    top_pivot_bearing_mount(cutaway=cutaway);
}

module lower_corner_socket(p_corner, p_top, p_other, cutaway=show_lower_socket_cutaway) {
    u_diag = unit(p_top - p_corner);
    u_bar  = unit(p_other - p_corner);

    color(c_socket)
    difference() {
        union() {
            // Diagonal carbon tube insertion cup.
            cylinder_between(
                p_corner,
                p_corner + u_diag * lower_socket_insert_len,
                d=lower_socket_outer_d
            );

            // Lower carbon bar insertion cup.
            cylinder_between(
                p_corner,
                p_corner + u_bar * lower_socket_insert_len,
                d=lower_socket_outer_d
            );

            // Slightly fuller corner node where the two cups meet.
            translate(p_corner) sphere(d=lower_socket_outer_d + 2);
        }

        // Blind carbon-pipe sockets. The pipe enters only lower_pipe_insert_len
        // from each mouth, then stops against a long internal shoulder.
        cylinder_between(
            p_corner + u_diag * lower_pipe_stop_s,
            p_corner + u_diag * (lower_socket_insert_len + 4),
            d=lower_socket_bore_d
        );
        cylinder_between(
            p_corner + u_bar * lower_pipe_stop_s,
            p_corner + u_bar * (lower_socket_insert_len + 4),
            d=lower_socket_bore_d
        );

        // Coaxial wire passages run from each mouth toward the corner, then
        // hand off to a smooth elbow so a pushed wire does not catch on a
        // sharp crossing.
        cylinder_between(
            p_corner + u_diag * (lower_wire_elbow_s - lower_wire_elbow_overlap),
            p_corner + u_diag * (lower_socket_insert_len + lower_wire_extra_len),
            d=lower_wire_bore_d
        );
        cylinder_between(
            p_corner + u_bar * (lower_wire_elbow_s - lower_wire_elbow_overlap),
            p_corner + u_bar * (lower_socket_insert_len + lower_wire_extra_len),
            d=lower_wire_bore_d
        );

        curved_wire_elbow(
            p_corner,
            u_diag,
            u_bar,
            radius_s=lower_wire_elbow_s,
            d=lower_wire_bore_d,
            steps=lower_wire_elbow_steps
        );

        // Side wire exits through the cylinder wall, about 15 mm in from each
        // pipe mouth. They intersect the coaxial wire passages above.
        translate(p_corner + u_diag * lower_side_wire_s)
            rotate([0, 90, 0])
                cylinder(h=lower_side_wire_hole_len, d=lower_side_wire_bore_d, center=true);
        translate(p_corner + u_bar * lower_side_wire_s)
            rotate([0, 90, 0])
                cylinder(h=lower_side_wire_hole_len, d=lower_side_wire_bore_d, center=true);

        // Inspection cutaway: remove the viewer-side half of the socket.
        if (cutaway) {
            translate([
                p_corner[0],
                p_corner[1] - lower_socket_insert_len - 4,
                p_corner[2] - lower_socket_insert_len - 8
            ])
            cube([
                lower_socket_outer_d,
                2 * lower_socket_insert_len + 8,
                2 * lower_socket_insert_len + 16
            ]);
        }
    }
}

module lower_socket_part(side="front", cutaway=false) {
    // Standalone lower socket. Dimensions are derived from the same rope
    // geometry used in the full assembly so parameter edits stay linked.
    p_corner = [0, 0, 0];
    p_top = side == "rear" ? [0, dy_sag, sag] : [0, -dy_sag, sag];
    p_other = side == "rear" ? [0, -lower_w, 0] : [0, lower_w, 0];

    lower_corner_socket(p_corner, p_top, p_other, cutaway=cutaway);
}

// -------------------- Balancing robot --------------------
function arc_pts(r, a0, a1, n=16) =
    [for (i = [0:n]) [r*cos(a0 + (a1 - a0)*i/n), r*sin(a0 + (a1 - a0)*i/n)]];

module sector_ring_y(r0, r1, a0, a1, depth) {
    // Ring sector extruded along +Y from y=0.
    // Polygon angle 90 deg maps to the local -Z direction.
    rotate([-90, 0, 0])
        linear_extrude(height=depth)
            polygon(concat(arc_pts(r1, a0, a1), arc_pts(r0, a1, a0)));
}

module box_col(sx, sy, z0, z1, wx, wy,
               win_w=0, win_h=0, win_pitch=0, win_z0=0, win_z1=0,
               x_wins=true, y_wins=true) {
    // Hollow rectangular column with optional lightening windows.
    difference() {
        translate([-sx/2, -sy/2, z0]) cube([sx, sy, z1 - z0]);
        translate([-sx/2 + wx, -sy/2 + wy, z0 - 1])
            cube([sx - 2*wx, sy - 2*wy, z1 - z0 + 2]);
        if (win_pitch > 0)
            for (zc = [win_z0 + win_pitch/2 : win_pitch : win_z1]) {
                if (x_wins)
                    translate([0, 0, zc]) cube([sx + 2, win_w, win_h], center=true);
                if (y_wins)
                    translate([0, 0, zc]) cube([win_w, sy + 2, win_h], center=true);
            }
    }
}

module taper_box(sx, y0, y1, z0, z1, wx, wy) {
    // Hollow rectangular tube whose Y width tapers y0 -> y1 over z0..z1.
    difference() {
        hull() {
            translate([-sx/2, -y0/2, z0])        cube([sx, y0, 0.01]);
            translate([-sx/2, -y1/2, z1 - 0.01]) cube([sx, y1, 0.01]);
        }
        hull() {
            translate([-sx/2 + wx, -y0/2 + wy, z0 - 0.5])
                cube([sx - 2*wx, y0 - 2*wy, 0.01]);
            translate([-sx/2 + wx, -y1/2 + wy, z1 + 0.5])
                cube([sx - 2*wx, y1 - 2*wy, 0.01]);
        }
    }
}

module robot_ankle_carrier_part(cutaway=false) {
    // Origin: bar axis center. Two 608 pockets open from both end faces;
    // the center land touches only the outer races (same rule as the top
    // pivot mounts). The flange on top takes the lower spine on 4 x M3.
    color(c_rlow)
    difference() {
        union() {
            cyl_y(carrier_len, carrier_boss_d);

            translate([-carrier_flange_x/2, -carrier_flange_y/2, carrier_flange_z0])
                cube([carrier_flange_x, carrier_flange_y, carrier_flange_t]);

            for (sx = [-1, 1])
                hull() {
                    translate([sx*(carrier_flange_x/2 - 5), 0, carrier_flange_z0 + 2])
                        sphere(d=6);
                    translate([sx*12, 0, 6])
                        sphere(d=12);
                }
        }

        for (yoff = [-rb_spacing/2, rb_spacing/2])
            translate([0, yoff, 0]) cyl_y(brg_w + 0.35, brg_od + 0.35);

        cyl_y(rb_spacing - brg_w + 0.35, top_center_bore_d);
        cyl_y(carrier_len + 2, shaft_clearance_d);

        for (sx = [-1, 1]) for (sy = [-1, 1])
            translate([sx*carrier_bolt_dx, sy*carrier_bolt_dy, carrier_flange_z0 - 1])
                cylinder(h=carrier_flange_t + 2, d=carrier_bolt_d);

        if (cutaway)
            translate([-carrier_boss_d/2 - 1, -carrier_len/2 - 1, 0])
                cube([carrier_boss_d + 2, carrier_len + 2, carrier_boss_d]);
    }
}

module robot_lower_spine_part() {
    // Origin: robot ankle frame. Bottom flange bolts onto the carrier and
    // the top cap carries the FR12-H101 hinge web; between them the body
    // flares out to lsp_y_wide and back (80 mm wide waistline).
    color(c_rlow)
    difference() {
        union() {
            translate([-carrier_flange_x/2, -lsp_y_wide/2, lsp_z0])
                cube([carrier_flange_x, lsp_y_wide, carrier_flange_t]);

            // 80 mm wide from the foot up to the waist.
            box_col(lsp_x, lsp_y_wide, lsp_z0, lsp_flare2,
                    lsp_wall, lsp_wall,
                    lsp_win_w, lsp_win_h, lsp_win_pitch,
                    lsp_z0 + 40, lsp_flare2 - 20, x_wins=true, y_wins=false);

            // Waist taper down to the FR12-H101 web width.
            taper_box(lsp_x, lsp_y_wide, lsp_y_top, lsp_flare2, lsp_flare3,
                      lsp_wall, lsp_wall);
            box_col(lsp_x, lsp_y_top, lsp_flare3, lsp_z1, lsp_wall, lsp_wall);

            // Top cap = FR12-H101 web seat.
            translate([-lsp_x/2, -lsp_y_top/2, lsp_z1 - lsp_cap_t])
                cube([lsp_x, lsp_y_top, lsp_cap_t]);
        }

        for (sx = [-1, 1]) for (sy = [-1, 1])
            translate([sx*carrier_bolt_dx, sy*carrier_bolt_dy, lsp_z0 - 1])
                cylinder(h=carrier_flange_t + 2, d=carrier_bolt_d);

        // v4: NO pre-drilled H101 web bolt holes - the real bracket
        // hole grid is uncertain, drill the cap through the web on
        // assembly (M2.5, nuts inside the box).
    }
}

module robot_upper_body_a_part() {
    // Lower piece of the upper spine, from just above the motor up to the
    // splice. Its bottom cap bolts down onto the FR12-S101 top plate.
    color(c_rup)
    difference() {
        union() {
            // Narrow hip section at S101 width, tapering out to 80 mm.
            box_col(usp_x, usp_y, usp_z0, usp_flare0, usp_wall_x, usp_wall_y);
            taper_box(usp_x, usp_y, usp_y_wide, usp_flare0, usp_flare1,
                      usp_wall_x, usp_wall_y);
            box_col(usp_x, usp_y_wide, usp_flare1, usp_split_z,
                    usp_wall_x, usp_wall_y,
                    usp_win_w, usp_win_h, usp_win_pitch,
                    usp_flare1 + 10, usp_split_z - 45,
                    x_wins=true, y_wins=false);

            // Bottom cap = FR12-S101 top plate seat.
            translate([-usp_x/2, -usp_y/2, usp_z0])
                cube([usp_x, usp_y, usp_cap_t]);
        }

        // v4: NO pre-drilled S101 bolt holes - the real bracket hole
        // grid is uncertain, drill the cap through the bracket on
        // assembly (M2.5, nuts inside the box).

        for (zz = [usp_split_z - 12, usp_split_z - 26])
            translate([0, 0, zz]) cyl_x(usp_x + 2, splice_bolt_d);
    }
}

module robot_upper_body_b_part() {
    // Upper half of the upper spine = v4 equipment bay:
    //   - splice TONGUES below (X sides only - a full sleeve box would
    //     block the 75 mm wide OpenCR sliding up through the open bottom),
    //   - central bulkhead carrying the OpenCR vertically on the swing
    //     centerline, components facing the X- service door,
    //   - battery shelf at the very top in the flared (98) head; the
    //     84 mm pack butts on the Y- wall, its leads bend in the 8.5 mm
    //     Y+ lead bay, the charge lead hangs out the charge window; the
    //     pack slides in along X through the door's battery-bay opening,
    //   - X- wall is one big door opening (robot_upper_body_door_part).
    // Assembly: board up through the bottom along the Y-wall grooves,
    // 4 snap rivets pushed in from the door side, wires, battery in
    // along X onto the shelf, strap, door on, then splice onto part A
    // (2 cross bolts).
    // Print UPRIGHT, tongues down (closed hollow box: printed lying down
    // the inner ceiling is an unreachable full-length bridge). Only the
    // battery shelf bridges ~30 mm across X - allow bridging or supports.
    sleeve_x = usp_x - 2*usp_wall_x - 0.6;
    sleeve_y = usp_y_wide - 2*usp_wall_y - 0.6;

    color(c_rup)
    difference() {
        union() {
            box_col(usp_x, usp_y_wide, usp_split_z, head_flare_z0,
                    usp_wall_x, usp_wall_y);

            // Head flare 90 -> 98: room for the battery lead bay.
            taper_box(usp_x, usp_y_wide, usp_y_head,
                      head_flare_z0, head_flare_z1, usp_wall_x, usp_wall_y);
            box_col(usp_x, usp_y_head, head_flare_z1, usp_z1,
                    usp_wall_x, usp_wall_y);

            // Top cap closes the head.
            translate([-usp_x/2, -usp_y_head/2, usp_z1 - usp_cap_t])
                cube([usp_x, usp_y_head, usp_cap_t]);

            // Splice tongues sliding down into part A (X sides only).
            for (sx = [-1, 1])
                translate([sx > 0 ? sleeve_x/2 - 4 : -sleeve_x/2,
                           -sleeve_y/2, usp_split_z - usp_sleeve_len])
                    cube([4, sleeve_y, usp_sleeve_len + 2]);

            // Central bulkhead, welded into both Y walls; carries the
            // board bosses and props the battery shelf from below.
            translate([bulk_x0, -usp_y_wide/2 + 1,
                       ocr_center_z - ocr_l/2 - 2])
                cube([bulk_t, usp_y_wide - 2,
                      shelf_z0 - (ocr_center_z - ocr_l/2 - 2)]);

            // Snap-rivet board mounts on the bulkhead: 1.0 web disc on
            // a hollow stand-off tube (legs snap open inside the tube).
            for (sy = [-1, 1]) for (sz = [-1, 1])
                translate([ocr_back_x, sy*ocr_hole_dy/2,
                           ocr_center_z + sz*ocr_hole_dz/2])
                    rotate([0, 90, 0]) {
                        cylinder(h=riv_web_t, d=riv_tube_od);
                        difference() {
                            cylinder(h=riv_web_t + riv_leg_room + 0.1,
                                     d=riv_tube_od);
                            translate([0, 0, -0.1])
                                cylinder(h=riv_web_t + riv_leg_room + 0.3,
                                         d=riv_tube_id);
                        }
                    }

            // Battery shelf, front edge flush with the door opening.
            translate([-usp_x/2 + usp_wall_x, -usp_y_head/2 + 1, shelf_z0])
                cube([usp_x - usp_wall_x - 3, usp_y_head - 2, shelf_t]);

            // Door-screw bosses behind the X- wall (self-tap engagement
            // 4 mm wall + 5 mm boss; clear of the board plane at x>-0.5).
            for (sy = [-1, 1])
                for (zz = [door_z0 + 4, (door_z0 + shelf_z0)/2 - 2,
                           shelf_z0 - 8])
                    translate([-usp_x/2 + usp_wall_x - 1, sy*door_screw_y, zz])
                        rotate([0, 90, 0]) cylinder(h=6, d=8);
        }

        // Service door opening through the X- wall: 66 wide over the
        // board zone, FULL interior width over the battery bay (the pack
        // slides in through here along X onto the shelf).
        translate([-usp_x/2 - 1, -door_open_y/2, door_z0])
            cube([usp_wall_x + 2, door_open_y, shelf_z0 - door_z0]);
        translate([-usp_x/2 - 1, -(usp_y_head/2 - usp_wall_y) - 0.01,
                   shelf_z0])
            cube([usp_wall_x + 2, usp_y_head - 2*usp_wall_y + 0.02,
                  door_z1 - shelf_z0]);

        // Output-plug pass: the T plug (15 x 10 (*)) drops from the
        // lead bay through the shelf down to the board space.
        translate([-8, usp_y_head/2 - usp_wall_y - 11, shelf_z0 - 1])
            cube([16, 11, shelf_t + 2]);

        // Charge window in the Y+ HEAD wall, facing the lead bay: the
        // short charge lead hangs out here and charges in place.
        // (*) recenter on the real lead position; mirror everything to Y-
        // if the pack is mounted the other way around.
        translate([-chg_win_x/2, usp_y_head/2 - usp_wall_y - 1,
                   (bat_z0 + bat_z1)/2 - chg_win_z/2])
            cube([chg_win_x, usp_wall_y + 2, chg_win_z]);

        // Rivet holes through the webs, and push-out holes through the
        // bulkhead behind them (poke the legs to release the board).
        for (sy = [-1, 1]) for (sz = [-1, 1]) {
            translate([ocr_back_x - 1, sy*ocr_hole_dy/2,
                       ocr_center_z + sz*ocr_hole_dz/2])
                rotate([0, 90, 0])
                    cylinder(h=riv_web_t + 2, d=riv_hole_d);
            translate([bulk_x0 - 0.1, sy*ocr_hole_dy/2,
                       ocr_center_z + sz*ocr_hole_dz/2])
                rotate([0, 90, 0])
                    cylinder(h=bulk_t + 0.2, d=3.5);
            // Pin access through the X+ wall, coaxial with the push hole.
            translate([usp_x/2 - usp_wall_x - 0.5, sy*ocr_hole_dy/2,
                       ocr_center_z + sz*ocr_hole_dz/2])
                rotate([0, 90, 0])
                    cylinder(h=usp_wall_x + 1.5, d=4);
        }

        // Door screw pilots: 3 pairs into the X- wall side lands, each
        // backed by an internal boss -> ~9 mm of self-tap engagement (a
        // bare 4 mm wall is too shallow). NO screws at the top: the cap
        // edge is too thin for a pilot, the door top edge tucks into the
        // cap rebate instead.
        for (sy = [-1, 1])
            for (zz = [door_z0 + 4, (door_z0 + shelf_z0)/2 - 2,
                       shelf_z0 - 8])
                translate([-usp_x/2 - 1, sy*door_screw_y, zz])
                    rotate([0, 90, 0])
                        cylinder(h=usp_wall_x + 6.5, d=door_bolt_d);

        // Rebate in the top cap X- edge: the door's top lip tucks in
        // here (tilt the door top in first, swing flat, then screw).
        translate([-usp_x/2 - 0.1, -44, usp_z1 - usp_cap_t - 0.4])
            cube([2.9, 88, 2.4]);

        // Battery strap slots through the shelf (strap wraps the pack
        // across X, buckle on the door side).
        for (sx = [-1, 1])
            translate([sx*13, 0, shelf_z0 + shelf_t/2])
                cube([strap_slot_w, strap_slot_l, shelf_t + 2], center=true);

        for (zz = [usp_split_z - 12, usp_split_z - 26])
            translate([0, 0, zz]) cyl_x(usp_x + 2, splice_bolt_d);
    }
}

module robot_upper_body_door_part() {
    // X- service door: flat plate over the whole equipment bay.
    // Reached from here: 4 OpenCR screws, USB / switches, battery slide-in.
    // (*) Cut USB / power-switch openings after measuring the real board;
    //     only two generic vents are modeled.
    color(c_rlow)   // contrast color: reads as the removable part
    difference() {
        // T-shaped plate: 88 wide over the board zone, 97 over the
        // flared battery bay, flush with the head top. Fit: tuck the top
        // lip into the cap rebate, swing flat, drive the 6 side screws.
        union() {
            translate([-usp_x/2 - door_t, -door_w/2, door_z0 - door_lip_z])
                cube([door_t, door_w, shelf_z0 - (door_z0 - door_lip_z)]);
            translate([-usp_x/2 - door_t, -door_w_top/2, shelf_z0])
                cube([door_t, door_w_top, usp_z1 - shelf_z0]);

            // Top lip: tucks into the cap rebate (tilt-and-tuck), so the
            // door top needs no screws at all.
            translate([-usp_x/2, -43.5, usp_z1 - usp_cap_t - 0.1])
                cube([2.3, 87, 1.8]);
        }

        // M3 clearance holes matching the three wall pilot pairs.
        for (sy = [-1, 1])
            for (zz = [door_z0 + 4, (door_z0 + shelf_z0)/2 - 2,
                       shelf_z0 - 8])
                translate([-usp_x/2 - door_t - 1, sy*door_screw_y, zz])
                    rotate([0, 90, 0]) cylinder(h=door_t + 2, d=3.4);

        // Generic vents over the board zone.
        for (zc = [ocr_center_z - 28, ocr_center_z + 28])
            translate([-usp_x/2 - door_t - 1, -18, zc - 11])
                cube([door_t + 2, 36, 22]);
    }
}

module xm430_vis() {
    // Origin: horn axis, horn toward +Y, case length along Z.
    color([0.15, 0.15, 0.17])
        translate([-xm_w/2, -xm_dep/2, -xm_axis_from_end])
            cube([xm_w, xm_dep, xm_len]);

    // Front horn (+Y) and HN12-I101 idler "fake horn" (-Y).
    color([0.30, 0.30, 0.33])
        translate([0, xm_dep/2 + xm_horn_lift/2, 0])
            cyl_y(xm_horn_lift, xm_horn_od);
    color([0.45, 0.45, 0.48])
        translate([0, -xm_dep/2 - xm_horn_lift/2, 0])
            cyl_y(xm_horn_lift, xm_horn_od);
}

module fr12_h101_vis() {
    // FR12-H101 hinge C bracket, approx (*). Origin: horn axis.
    // Two arms grip the horn (+Y) and the idler (-Y); the web below the
    // axis bolts onto the lower spine top cap. Lower-body-fixed part.
    arm_y_in = xm_dep/2 + xm_horn_lift;
    web_z_in = -h101_axis_to_web;

    color([0.62, 0.64, 0.68]) {
        for (sy = [-1, 1])
            translate([0, sy*(arm_y_in + h101_t/2), 0])
            difference() {
                hull() {
                    cyl_y(h101_t, xm_horn_od + 7);
                    translate([-h101_arm_w/2, -h101_t/2, web_z_in - 0.1])
                        cube([h101_arm_w, h101_t, 4]);
                }
                cyl_y(h101_t + 0.2, 9);  // center clearance over horn boss
            }

        // Web spanning between the two arms, under the motor tail.
        translate([-h101_web_x/2, -(arm_y_in + h101_t), web_z_in - h101_t])
            cube([h101_web_x, 2*(arm_y_in + h101_t), h101_t]);
    }
}

module fr12_s101_vis() {
    // FR12-S101 style body C bracket, approx (*). Origin: horn axis.
    // Side plates on the motor X faces, top plate over the motor tail end,
    // bolted up into the upper spine bottom cap. Upper-body-fixed part.
    top_z_in = xm_len - xm_axis_from_end;

    color([0.62, 0.64, 0.68]) {
        translate([-(xm_w/2 + s101_t), -xm_dep/2, top_z_in])
            cube([xm_w + 2*s101_t, xm_dep, s101_t]);

        for (sx = [-1, 1])
            translate([sx > 0 ? xm_w/2 : -xm_w/2 - s101_t, -xm_dep/2,
                       top_z_in - s101_arm_h])
                cube([s101_t, xm_dep, s101_arm_h]);
    }
}

module opencr_vis() {
    // Origin: board center, board plane normal +X. The on-board IMU is the
    // theta sensor - its height above the hip is echoed for the firmware.
    color([0.05, 0.42, 0.18])
        cube([ocr_t, ocr_w, ocr_l], center=true);
    color([0.10, 0.10, 0.10])
        translate([3, 0, 8]) cube([5, 14, 14], center=true);
    color([0.88, 0.88, 0.88])
        translate([4, ocr_w/2 - 12, -ocr_l/2 + 9]) cube([7, 16, 12], center=true);
}

module lipo_vis() {
    color([0.20, 0.30, 0.90])
        cube([bat_x, bat_y, bat_z], center=true);
}

module robot_assembly(p_ankle) {
    // Rigid-rod demo: the robot stays aligned with the rope (hanging), so
    // the whole rope+robot rotates about the pivot like one pendulum rod.
    rel1 = rigid_rod_demo ? 180 : alpha_demo - phi_demo;
    rel2 = rigid_rod_demo ? 0   : theta_demo - alpha_demo;

    translate(p_ankle) {
        // Bar-fixed hardware: bearings and two plain lock collars. No
        // rotation stop - the inverted robot clears floor and braces.
        for (yoff = [-rb_spacing/2, rb_spacing/2])
            translate([0, yoff, 0]) bearing_608_y();

        lock_collar_at_y(-carrier_len/2, -1);
        lock_collar_at_y(carrier_len/2, 1);

        // Robot lower body.
        rotate([0, rel1, 0]) {
            robot_ankle_carrier_part();
            robot_lower_spine_part();

            // FR12-H101 hinge: fixed to the lower body via the web, driven
            // by the horn - it IS the lower side of the hip joint.
            translate([0, 0, hip_z]) fr12_h101_vis();

            // Upper body about the hip axis.
            translate([0, 0, hip_z]) rotate([0, rel2, 0]) translate([0, 0, -hip_z]) {
                robot_upper_body_a_part();
                robot_upper_body_b_part();

                translate([0, 0, hip_z]) xm430_vis();
                translate([0, 0, hip_z]) fr12_s101_vis();

                // Board inside on the bulkhead, components toward X-.
                translate([ocr_back_x - ocr_t/2, 0, ocr_center_z])
                    rotate([0, 0, 180]) opencr_vis();

                // Battery on the shelf, butted on the Y- head wall.
                translate([0, bat_y_mid, (bat_z0 + bat_z1)/2]) lipo_vis();

                robot_upper_body_door_part();
            }
        }

        // AN25 analog wire: bar side exit -> up the lower spine -> small hip
        // loop -> in through an A-piece X- window -> up inside to the
        // OpenCR analog input (door side).
        if (show_cables) {
            tube_path([
                [0, 30, 6],
                [0, lsp_y_wide/2 + 4, 40],
                [0, lsp_y_wide/2 + 4, hip_z - 60],
                [-14, -6, hip_z + 6],
                [-usp_x/2 - 4, -10, usp_flare1 + 30],
                [0, -8, usp_split_z - 10],
                [-5, -8, ocr_center_z - ocr_l/2 + 8]
            ], d=2, col=c_cable2);
        }
    }
}

module lower_service_loop(p_ankle) {
    // A clock-spring-like loop around the ankle axis.
    // This is intentionally outside the bearing centerline to reduce cable torque.
    if (show_cables) {
        color(c_cable)
        union() {
            // Cable exiting the lower bar near the rear side.
            tube_path([
                p_ankle + [0, 28, 0],
                p_ankle + [8, 35, -10],
                p_ankle + [22, 35, -18]
            ], d=2, col=c_cable);

            arc_cable_y(p_ankle, radius=30, yoff=35, a0=210, a1=-120, d=2, col=c_cable, n=36);

            // Cable entering the robot body after the loop.
            tube_path([
                p_ankle + [-15, 35, -26],
                p_ankle + [-8, 18, -45],
                p_ankle + [0, 0, -72]
            ], d=2, col=c_cable);
        }
    }
}

module slackline_trapezoid() {
    p_top_front = [0, top_front_y, 0];
    p_top_rear  = [0, top_rear_y, 0];
    p_bot_front = [0, bottom_front_y, -sag];
    p_bot_rear  = [0, bottom_rear_y, -sag];
    p_ankle     = [0, frame_y/2, -sag];
    p_bot_front_diag_stop = p_bot_front + unit(p_top_front - p_bot_front) * lower_pipe_stop_s;
    p_bot_rear_diag_stop  = p_bot_rear  + unit(p_top_rear  - p_bot_rear)  * lower_pipe_stop_s;
    p_bot_front_bar_stop  = p_bot_front + unit(p_bot_rear  - p_bot_front) * lower_pipe_stop_s;
    p_bot_rear_bar_stop   = p_bot_rear  + unit(p_bot_front - p_bot_rear)  * lower_pipe_stop_s;
    p_top_front_diag_stop = p_top_front + unit(p_bot_front - p_top_front) * top_pipe_stop_s;
    p_top_rear_diag_stop  = p_top_rear  + unit(p_bot_rear  - p_top_rear)  * top_pipe_stop_s;

    // Top sockets and shaft stubs.
    upper_socket(true);
    upper_socket(false);

    // Carbon trapezoid: no physical upper bar.
    carbon_tube_between(p_top_front_diag_stop, p_bot_front_diag_stop);
    carbon_tube_between(p_top_rear_diag_stop,  p_bot_rear_diag_stop);
    carbon_tube_between(p_bot_front_bar_stop, p_bot_rear_bar_stop);

    // Lower corner sockets.
    lower_corner_socket(p_bot_front, p_top_front, p_bot_rear);
    lower_corner_socket(p_bot_rear,  p_top_rear,  p_bot_front);

    // Cable drawn on/inside the rear diagonal and lower bar.
    if (show_cables) {
        tube_path([
            p_top_rear + [1.8, 0, -4],
            p_bot_rear + [1.8, 0, 4],
            p_ankle    + [1.8, 24, 3]
        ], d=1.8, col=c_cable2);
    }

    // Balancing robot assembled on the lower bar, plus the lower service loop.
    if (show_robot)
        robot_assembly(p_ankle);

    lower_service_loop(p_ankle);
}

// -------------------- Top cable service loop --------------------
module top_encoder_service_loop() {
    if (show_cables) {
        // The moving entry point is a point slightly down the rear diagonal tube.
        local_entry = [0, top_rear_y, -34];
        entry = add3([axis_x, 0, axis_z], rot_y(local_entry, phi_demo));

        fixed_connector = [axis_x, enc_face_y + enc_body_len - 5,
                           axis_z + enc_body_d/2 + 6];
        fixed_relief    = [axis_x - 38, frame_y + 28, axis_z + 30];
        loop_low        = [axis_x - 54, frame_y + 20, axis_z - 12];
        rotating_relief = entry + [-12, 0, 8];

        tube_path([
            fixed_connector,
            fixed_relief,
            loop_low,
            rotating_relief,
            entry
        ], d=2, col=c_cable);

        // Small strain-relief clamps, one fixed and one rotating.
        color([0.02, 0.02, 0.02])
            translate(fixed_relief) cube([14, 7, 7], center=true);

        color([0.02, 0.02, 0.02])
            translate(rotating_relief) cube([12, 6, 6], center=true);
    }
}

// -------------------- Render selection --------------------
module full_assembly() {
    frame_assembly();
    upper_fixed_hardware();

    if (show_pivot_axis) {
        color([1.0, 0.45, 0.0, 0.55])
        translate([axis_x, profile/2, axis_z])
            cyl_y(frame_y - profile, 2.2, center=false);
    }

    translate([axis_x, 0, axis_z])
        rotate([0, phi_demo, 0])
            slackline_trapezoid();

    top_encoder_service_loop();
}

if (is_undef(part_mode) || part_mode == "assembly") {
    full_assembly();

    // -------------------- Console summary --------------------
    echo("============================================================");
    echo("frame_v4.scad: internal OpenCR + battery equipment bay");
    echo(str("Frame X/Y/Z [mm] = ", frame_x, " / ", frame_y, " / ", frame_z));
    echo(str("Pivot axis: X=", axis_x, "  Z=", axis_z, "  Y from ", front_pivot_y, " to ", rear_pivot_y));
    echo(str("Upper tube entry width [mm] = ", upper_w));
    echo(str("Lower bar exposed length [mm] = ", lower_bar_exposed_len));
    echo(str("Lower socket corner-to-corner width [mm] = ", lower_w));
    echo(str("Diagonal carbon tube [mm] = ", diag_len));
    echo(str("Computed sag R preview [mm] = ", sag, " (", sag/10, " cm)"));
    echo(str("Preview phi [deg] = ", phi_demo, "  alpha [deg] = ", alpha_demo));
    echo("Upper side: two 608 bearings per mount + solid shaft coupler + fixed shaft encoder");
    echo(str("Encoder chain [y]: shaft tip ", rear_shaft_tip_y,
             " (print dia ", shaft_print_d, ") | coupler ", coupler_start_y,
             "..", coupler_end_y, " | struct engage ",
             rear_shaft_tip_y - coupler_start_y, " / encoder engage ",
             coupler_end_y - enc_tip_y));
    echo(str("Encoder OUTER-mounted (v4): face ", enc_face_y, " rear ",
             enc_rear_y, " | plate ", enc_plate_in_y, "..", enc_plate_out_y,
             " (as printed, untouched) | coupler end to plate ",
             enc_plate_in_y - coupler_end_y, " mm clear"));
    echo("Lower side: lower carbon bar passes through two bearings in robot lower body");
    echo(str("Robot L1/L2 [mm] = ", robot_l1, " / ", robot_l2,
             "  head top world z = ", axis_z - sag + robot_l1 + robot_l2));
    echo("Robot alpha: FREE (no stop; two plain lock collars) - inverted fall clears floor/braces");
    echo(str("Inverted hang (alpha=180, phi=0) floor clearance [mm] = ",
             axis_z - sag - usp_z1));
    robot_half_x = usp_x/2 + door_t + 2;  // v4: door face + screw heads
                                          // (was 34: old top tray corner)
    robot_hang_len = usp_z1;   // ankle -> head cap top (battery now inside)
    phi_hit_x = asin((frame_x - profile - axis_x - robot_half_x) / sag);
    phi_hit_z = acos(max(-1, min(1,
        (axis_z - (brace_z + profile) - robot_hang_len) / sag)));
    echo(phi_hit_z > phi_hit_x
        ? str("Y brace z=", brace_z, "..", brace_z + profile,
              ": inverted-hang robot hits it for phi in +/-[", phi_hit_x,
              ", ", phi_hit_z, "] deg -> hanging swing safe below +/-",
              phi_hit_x, " deg (upright balancing unaffected)")
        : str("Y brace z=", brace_z, "..", brace_z + profile,
              ": NO inverted-hang collision at any swing angle",
              " (hang lowest point ", axis_z - sag - robot_hang_len,
              " mm > brace top ", brace_z + profile, " mm)"));
    rigid_corner_r = sqrt(pow(sag + robot_hang_len, 2) + pow(robot_half_x, 2));
    brace_corner_r = sqrt(pow(axis_x - profile, 2)
                        + pow(axis_z - brace_z - profile, 2));
    echo(str("Rigid-rod sweep: robot swept corner radius ", rigid_corner_r,
             " mm vs brace inner corner at ", brace_corner_r,
             " mm from pivot -> min radial clearance ",
             brace_corner_r - rigid_corner_r, " mm (any phi)"));
    echo(str("OpenCR (IMU) center above hip [mm] = ", ocr_center_z - hip_z,
             "  -> firmware ELL_IMU candidate (v4 internal mount: update FW)"));
    echo("Robot mass estimate: lower ~230 g / upper ~550 g+ (re-weigh: measured 84x34x25 pack) -> ~30:70");
    echo("Printed robot parts: ankle_carrier, lower_spine, upper_body_a, upper_body_b, upper_body_door");
    echo("============================================================");
} else if (part_mode == "lower_socket") {
    lower_socket_part(
        side=is_undef(socket_side) ? "front" : socket_side,
        cutaway=is_undef(part_cutaway) ? false : part_cutaway
    );
} else if (part_mode == "upper_socket") {
    upper_socket_part(
        side=is_undef(socket_side) ? "front" : socket_side
    );
} else if (part_mode == "lock_collar") {
    lock_collar_part(
        dir=is_undef(collar_dir) ? 1 : collar_dir
    );
} else if (part_mode == "solid_coupler" || part_mode == "oldham_coupler") {
    solid_coupler_part(
        cutaway=is_undef(part_cutaway) ? false : part_cutaway
    );
} else if (part_mode == "encoder_bracket") {
    encoder_bracket_part();
} else if (part_mode == "bearing_mount") {
    bearing_mount_part(
        cutaway=is_undef(part_cutaway) ? false : part_cutaway
    );
} else if (part_mode == "ankle_carrier") {
    robot_ankle_carrier_part(
        cutaway=is_undef(part_cutaway) ? false : part_cutaway
    );
} else if (part_mode == "lower_spine") {
    // part_cutaway removes the (x<0, y>0) corner quarter - VIEWING ONLY.
    difference() {
        translate([0, 0, -lsp_z0]) robot_lower_spine_part();
        if (!is_undef(part_cutaway) && part_cutaway)
            translate([-60, 0, -1])
                cube([60, 60, lsp_z1 - lsp_z0 + 2]);
    }
} else if (part_mode == "upper_body_a") {
    // part_cutaway removes the (x<0, y>0) corner quarter - VIEWING ONLY.
    difference() {
        translate([0, 0, -usp_z0]) robot_upper_body_a_part();
        if (!is_undef(part_cutaway) && part_cutaway)
            translate([-50, 0, -1])
                cube([50, 60, usp_split_z - usp_z0 + 2]);
    }
} else if (part_mode == "upper_body_b") {
    // part_cutaway removes the (x<0, y>0) corner quarter so the central
    // bulkhead, OpenCR bosses, guide grooves and battery shelf show -
    // VIEWING ONLY, export the STL with part_cutaway=false.
    difference() {
        translate([0, 0, -(usp_split_z - usp_sleeve_len)])
            robot_upper_body_b_part();
        if (!is_undef(part_cutaway) && part_cutaway)
            translate([-50, 0, -1])
                cube([50, 60, usp_z1 - usp_split_z + usp_sleeve_len + 2]);
    }
} else if (part_mode == "upper_body_door") {
    // Flat on the bed, outer face down.
    translate([-(door_z0 + door_z1)/2, 0, -usp_x/2])
        rotate([0, 90, 0]) robot_upper_body_door_part();
}
